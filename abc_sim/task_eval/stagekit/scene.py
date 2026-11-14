"""Per-object measurements shared by every subtask scene: finger contacts, lift above rest, grasp-site distance."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from abc_sim.task_eval.stagekit.geometry import T
from abc_sim.task_eval.stagekit.robot import ARMS


def subtree(model: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in out:
            out.add(body)
    return out


def geoms_of(model: mujoco.MjModel, bodies: set[int]) -> np.ndarray:
    return np.asarray([g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies], dtype=np.int32)


def is_desc(model: mujoco.MjModel, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = int(model.body_parentid[b])
    return False


def free_joints_matching(model: mujoco.MjModel, pattern: str) -> list[tuple[str, int]]:
    """(object name, joint id) for joints named like ``pattern`` (a regex with one group: the object's index), sorted
    by that index; the object name is the joint name without a trailing ``_jnt`` / ``_joint``."""
    rx = re.compile(pattern)
    found = []
    for j in range(model.njnt):
        name = model.jnt(j).name or ""
        m = rx.fullmatch(name)
        if m:
            base = name[:-4] if name.endswith("_jnt") else name[:-6] if name.endswith("_joint") else name
            found.append((int(m.group(1)), base, j))
    found.sort()
    return [(n, j) for _, n, j in found]


class ObjectScene:
    """Free-jointed task objects of one MjModel and the fingers/grasp sites of both arms.

    ``observe`` returns arrays over objects (subclasses add task-specific entries):
      lift        height now minus the rest height (m)
      hits        (m, 4) bool: object touches finger arm*2+side
      fingers     fingers of the best arm on the object (0-2)
      grasp_arm   arm with both fingers on it (0 left, 1 right, 2 both, -1 none)
      grasped     both fingers of one arm on it;  touched: any finger on it
      dist        distance from the nearest grasp site to the object's CoM (m);  near_arm: that arm
    """

    def __init__(self, model: mujoco.MjModel, objects: list[tuple[str, int]]) -> None:
        self.model = model
        self.names = [n for n, _ in objects]
        self.joints = [int(j) for _, j in objects]
        self.qpos_adr = np.asarray([int(model.jnt_qposadr[j]) for j in self.joints], dtype=np.int64)
        self.dof_adr = np.asarray([int(model.jnt_dofadr[j]) for j in self.joints], dtype=np.int64)
        self.bodies = [int(model.jnt_bodyid[j]) for j in self.joints]
        self.owner = np.full((model.ngeom, 2), -1, dtype=np.int32)  # geom -> (0, object k) | (1, finger arm*2+side)
        for k, body in enumerate(self.bodies):
            self.owner[geoms_of(model, subtree(model, body))] = (0, k)
        for a, arm in enumerate(ARMS):
            for s, side in enumerate(("link_left_finger", "link_right_finger")):
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_{side}")
                if body < 0:
                    raise ValueError(f"body {arm}_{side} not found")
                self.owner[geoms_of(model, subtree(model, body))] = (1, a * 2 + s)
        self.sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grasp_site") for arm in ARMS]
        if min(self.sites) < 0:
            raise ValueError("left/right_grasp_site not found")

    def heights(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray([data.qpos[a + 2] for a in self.qpos_adr], dtype=np.float64)

    def finger_hits(self, data: mujoco.MjData) -> np.ndarray:
        m = len(self.names)
        hits = np.zeros((m, 4), dtype=bool)
        for i in range(int(data.ncon)):
            c = data.contact[i]
            o1, o2 = self.owner[int(c.geom1)], self.owner[int(c.geom2)]
            if o1[0] == 0 and o2[0] == 1:
                hits[o1[1], o2[1]] = True
            elif o1[0] == 1 and o2[0] == 0:
                hits[o2[1], o1[1]] = True
        return hits

    def observe(self, data: mujoco.MjData, rest_height: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays over objects. Needs mj_forward/mj_step to have run (contacts, site_xpos, subtree_com)."""
        return {"lift": self.heights(data) - rest_height, **self.contact_obs(data)}

    def contact_obs(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """The finger-contact and grasp-site entries of ``observe`` (everything but ``lift``)."""
        m = len(self.names)
        hits = self.finger_hits(data)
        per_arm = hits.reshape(m, 2, 2).sum(axis=2)  # fingers of each arm on the object
        both = per_arm == 2
        grasp_arm = np.where(both[:, 0] & both[:, 1], 2, np.where(both[:, 0], 0, np.where(both[:, 1], 1, -1)))
        sites = np.asarray(data.site_xpos[self.sites], dtype=np.float64)
        com = np.asarray(data.subtree_com[self.bodies], dtype=np.float64).reshape(m, 3)
        dist = np.linalg.norm(com[:, None, :] - sites[None, :, :], axis=-1)  # (m, 2 arms)
        return {
            "hits": hits,
            "fingers": per_arm.max(axis=1),
            "grasp_arm": grasp_arm,
            "grasped": grasp_arm >= 0,
            "touched": hits.any(axis=1),
            "dist": dist.min(axis=1),
            "near_arm": dist.argmin(axis=1),
        }

    def object_T(self, data: mujoco.MjData, k: int) -> np.ndarray:
        a = int(self.qpos_adr[k])
        return T(data.qpos[a:a + 3], data.qpos[a + 3:a + 7])
