"""Physics-defined stages for the approved put-bottles reach/pick/carry/drop chain."""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.task_eval.bottles import PutBottlesInBinEvaluator
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching, geoms_of, subtree
from abc_sim.task_eval.stagekit.tracker import StageTracker

STAGES = ("reach", "pick", "carry", "drop", "done")
R, P, C, D, DONE = range(5)
FALLEN = len(STAGES)
LABELS = STAGES + ("fallen",)
STAGE_PROMPTS = {
    "reach": "bottle subtask: move a gripper to the target bottle",
    "pick": "bottle subtask: grasp the target bottle and lift it",
    "carry": "bottle subtask: carry the target bottle over the bin opening",
    "drop": "bottle subtask: release the target bottle into the bin",
}


@dataclass(frozen=True)
class StageConfig:
    reach_m: float = 0.05
    reach_exit_m: float = 0.07
    lift_m: float = 0.03
    push_m: float = 0.05
    fallen_m: float = 0.05
    opening_margin_m: float = 0.02
    opening_exit_m: float = 0.02
    bin_moved_m: float = 0.05
    bin_tilt_rad: float = np.deg2rad(20)
    at_rest_speed: float = 0.02
    at_rest_angular_speed: float = 0.2
    hold: int = 5
    eps: dict = field(default_factory=lambda: dict(reach=0.005, pick=0.05, carry=0.005, drop=0.005))
    span: dict = field(default_factory=lambda: dict(reach=0.30, pick=1.0, carry=0.40, drop=0.10))


class BottleScene(ObjectScene):
    def __init__(self, model, cfg=None):
        super().__init__(model, free_joints_matching(model, r"bottle_(\d+)_joint"))
        self.cfg = cfg or StageConfig()
        self.geometry = PutBottlesInBinEvaluator(model=model, spec=None)
        self.bin_adr = self.geometry._bin_qpos_adr
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint")
        self.bin_body = int(model.jnt_bodyid[joint])
        self.bin_geoms = set(geoms_of(model, subtree(model, self.bin_body)))

    def heights(self, data):
        # Preserve the original bin anchor through exact snapshot restores as well as bottle rest positions.
        com = np.asarray(data.subtree_com[self.bodies]).reshape(-1, 3)
        return np.vstack((com, data.qpos[self.bin_adr:self.bin_adr + 3]))

    def observe(self, data, rest):
        rest = np.asarray(rest).reshape(-1, 3)
        pos = np.asarray(data.subtree_com[self.bodies]).reshape(-1, 3)
        bin_pos = data.qpos[self.bin_adr:self.bin_adr + 3]
        rotation = data.xmat[self.bin_body].reshape(3, 3)
        local = (pos - bin_pos) @ rotation
        radius = np.hypot(local[:, 0], local[:, 2])
        top = self.geometry._bin_top_y
        opening_radius = self.geometry._bin_radius_top - self.cfg.opening_margin_m
        radial_gap = np.maximum(radius - opening_radius, 0)
        gap = np.hypot(radial_gap, np.maximum(top - local[:, 1], 0))
        stock = self.geometry.evaluate_qpos_batch(data.qpos[None]).metrics["bottle_in_bin_mask"][0]
        speed = np.array([np.linalg.norm(data.qvel[a:a + 3]) for a in self.dof_adr])
        angular_speed = np.array([np.linalg.norm(data.qvel[a + 3:a + 6]) for a in self.dof_adr])
        contacts = self.contact_obs(data)
        direct = np.zeros(len(self.names), dtype=bool)
        neighbors = [set() for _ in self.names]
        resting_contact = np.zeros(len(self.names), dtype=bool)
        for i in range(data.ncon):
            contact = data.contact[i]
            for obj, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                kind, index = self.owner[obj]
                other_kind, other_index = self.owner[other]
                if kind == 0 and other_kind != 1:
                    resting_contact[index] = True
                if kind == 0 and other in self.bin_geoms:
                    direct[index] = True
                elif kind == 0 and other_kind == 0 and index != other_index:
                    neighbors[index].add(other_index)
        # Support must ultimately reach the fixture. Two airborne bottles touching each other are not settled.
        stable = (stock & ~contacts["touched"] & (speed <= self.cfg.at_rest_speed)
                  & (angular_speed <= self.cfg.at_rest_angular_speed))
        supported = direct & stable
        for _ in self.names:
            grown = supported | (stable & np.array([any(supported[j] for j in ns) for ns in neighbors], dtype=bool))
            if np.array_equal(grown, supported):
                break
            supported = grown
        return dict(xy=pos[:, :2], lift=pos[:, 2] - rest[:len(pos), 2],
                    speed=speed, angular_speed=angular_speed, radial=radius, radial_gap=radial_gap,
                    gap=gap, above_rim=local[:, 1] - top,
                    over=(radius <= opening_radius) & (local[:, 1] >= top), in_bin=stock, supported=supported,
                    resting_contact=resting_contact,
                    bin_moved=float(np.linalg.norm(bin_pos[:2] - rest[-1, :2])),
                    bin_tilt=float(np.arccos(np.clip(rotation[2, 1], -1, 1))),
                    **contacts)


def classify(obs, prev, cfg):
    near = (obs["dist"] <= np.where(prev == P, cfg.reach_exit_m, cfg.reach_m)) | obs["touched"]
    out = np.full(prev.shape, R, dtype=np.int64)
    out[near] = P
    out[obs["grasped"] & (obs["lift"] >= cfg.lift_m)] = C
    inside = obs["in_bin"] & (obs["above_rim"] <= 0)
    over = obs["over"] | inside | ((prev == D) & (obs["radial_gap"] <= cfg.opening_exit_m)
                              & (obs["above_rim"] >= -cfg.opening_exit_m))
    out[over & obs["touched"]] = D
    at_rest = (obs["speed"] <= cfg.at_rest_speed) & (obs["angular_speed"] <= cfg.at_rest_angular_speed)
    # Released bottles retain their stage throughout their fall, including the apex of a tumbling bottle.
    free = ~obs["touched"] & (~at_rest | ~obs["resting_contact"])
    out[free] = prev[free]
    out[obs["in_bin"] & obs["supported"] & ~obs["touched"] & at_rest] = DONE
    # The bin floor is close to tabletop height, so a true in-bin bottle must not become FALLEN.
    out[(obs["lift"] < -cfg.fallen_m) & ~obs["in_bin"]] = FALLEN
    return out


def progress(stage, obs):
    if stage == R:
        return -obs["dist"]
    if stage == P:
        return 0.25 * obs["fingers"] + 0.5 * np.clip(obs["lift"] / 0.03, 0, 1) * obs["grasped"]
    if stage == C:
        return -np.minimum(obs["gap"], 1.0)
    if stage == D:
        return -obs["radial_gap"] + 0.05 * ~obs["touched"]
    raise ValueError(stage)


class BottleTracker(StageTracker):
    def __init__(self, scene, cfg=None):
        super().__init__(scene, classify, cfg or scene.cfg, start_label=R)
