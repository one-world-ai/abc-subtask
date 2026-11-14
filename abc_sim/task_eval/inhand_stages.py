"""Stages of inhand_transfer_item_to_other_side, for single-stage RL (approved stage table: subtask_rewards/inhand).

One item per episode. Giver A = the arm on the item's spawn side (left arm for a left/+y spawn), receiver B = the
other arm; s = +1 for a left spawn, and "far side" means y·s <= -0.10 (the stock band). Labels come from the sim
state alone (contacts per arm, poses, speeds) plus hysteresis on the previous label:

    stage    the item is in it when
    reach    none of the below
    pick     A's pinch centre (PAD) within reach_m of the item's surface (exit reach_exit_m) or an A finger on it;
             entering also needs A's fingers >= 50 % open
    present  both A fingers on it, lifted >= lift_m (B may touch it)
    handoff  ``handoff_stage.handing``: in the middle region M, lifted, A holds it or B has a finger on it
    carry    B holds it alone and A is clear (entry: also lifted and touching nothing but B)
    place    over the far zone (origin y·s <= -far_m, exit +far_exit_m, on the table), A not touching, and either B
             holds it alone <= place_lift_m above rest (entry) or B still touches it (stay)
    done     the fixed stock success: far band, <= 10 cm up, on the table, no robot contact, at rest
    fallen   CoM below the tabletop or the item outside the table's xy bounds
A free item (no robot contact) that is moving keeps its previous label (dropped / sliding / bouncing).

Progress (higher is better; only shapes the reward):
    reach    -(A pinch centre -> nearest surface point)
    pick     0.25 x A fingers + 0.5 x lift fraction (of lift_m) while A holds it with both
    present  -(CoM distance to M)
    handoff  ``handoff_stage.progress``
    carry    -(gap of the origin to the far zone: y and table x)
    place    -(CoM height above rest, clipped to [0, 0.3]) + 0.05 once no robot touches it
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.inhand_transfer import InhandTransferOtherSideEvaluator
from abc_sim.task_eval.stagekit.scene import ObjectScene, geoms_of, subtree
from abc_sim.task_eval.stagekit.tracker import StageTracker

# Pinch centre in link_6 (finger pads span y -0.047..-0.017, z 0.07..0.15; the grasp site is at (0, 0, 0.1347),
# 3 cm off the pads). "Near" and "clear" are measured from it.
PAD = np.array([0.0, -0.031, 0.125])

STAGES = ("reach", "pick", "present", "handoff", "carry", "place", "done")
R, P, PR, H, C, PL, DONE = range(7)
FALLEN = len(STAGES)
LABELS = STAGES + ("fallen",)
STAGE_PROMPTS = {
    "reach": "inhand transfer subtask: reach the item with the near arm",
    "pick": "inhand transfer subtask: grasp and lift the item with the near arm",
    "present": "inhand transfer subtask: bring the item to the middle",
    "handoff": "inhand transfer subtask: hand the item to the other arm",
    "carry": "inhand transfer subtask: carry the item to the other side",
    "place": "inhand transfer subtask: set the item down on the other side",
}


@dataclass(frozen=True)
class StageConfig:
    # Pinch centre (PAD) -> nearest surface point. Open top-down pre-grasps put it 0-3.3 cm from the surface (a
    # can's axis is 3.3 cm from its side; raised up to 3 cm over thin items); checked on the pick starts (STATUS).
    reach_m: float = 0.04
    reach_exit_m: float = 0.06
    lift_m: float = 0.03  # lifted (as mug flip / pick template)
    push_m: float = 0.03  # reach/pick: item moved this far (xy) while not held by A = pushed (approved table)
    far_m: float = 0.13  # far zone: stock band (0.10) + 3 cm, so carry's goal leaves place room inside the band
    far_exit_m: float = 0.01
    band_m: float = 0.10  # stock band (other_side_min_y_m)
    place_lift_m: float = 0.15  # carry goal: at most this above rest when it enters the far zone
    release_low_m: float = 0.05  # decision 3: "set down" = released with its lowest point <= 5 cm above the table
    at_rest_speed: float = 0.02  # stock scorer fix
    below_table_m: float = 0.02  # CoM this far below the tabletop = fallen
    closed_open_frac: float = 0.5  # reach: touching it with A's fingers < 50 % open = failure (G5 repair round 1)
    hold: int = 5
    handoff: hs.HandoffConfig = field(default_factory=hs.HandoffConfig)
    eps: dict = field(default_factory=lambda: dict(reach=0.005, pick=0.02, present=0.005, handoff=0.02,
                                                   carry=0.005, place=0.002))
    span: dict = field(default_factory=lambda: dict(reach=0.30, pick=1.0, present=0.30, handoff=1.25,
                                                    carry=0.30, place=0.10))


def surface_points(model: mujoco.MjModel, body: int, n_samples: int = 1500, seed: int = 0) -> np.ndarray:
    """Points on the collision meshes of ``body``'s subtree, in ``body``'s frame: vertices plus area-weighted samples
    on the faces (long items' meshes have vertices only at their ends)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    Rb, pb = data.xmat[body].reshape(3, 3), data.xpos[body]
    vs, tris = [], []
    for g in geoms_of(model, subtree(model, body)):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or (model.geom_contype[g] == 0
                                                               and model.geom_conaffinity[g] == 0):
            continue
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        w = model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        local = (w - pb) @ Rb
        vs.append(local)
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        tris.append(local[model.mesh_face[fa:fa + fn]])
    if not vs:
        raise ValueError("item has no collision mesh")
    v, t = np.concatenate(vs), np.concatenate(tris)
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(t), size=n_samples, p=area / area.sum())
    r1, r2 = rng.random(n_samples), rng.random(n_samples)
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * t[i, 0] + (s * (1 - r2))[:, None] * t[i, 1] + (s * r2)[:, None] * t[i, 2]
    return np.concatenate([v, pts])


class InhandScene(ObjectScene):
    """The single item (``task_object_joint``). ``heights`` returns the CoM *position* (1, 3) so rest xy is kept for
    the push test. ``set_side`` (from the randomization metadata) fixes giver/receiver and the far side."""

    def __init__(self, model: mujoco.MjModel, cfg: StageConfig | None = None) -> None:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
        if j < 0:
            raise ValueError("task_object_joint not found")
        super().__init__(model, [("task_object", j)])
        self.cfg = cfg or StageConfig()
        self.stock = InhandTransferOtherSideEvaluator(model=model, spec=None)
        self.table_z = float(self.stock._table_z)
        self.table_x, self.table_y = self.stock._table_x_bounds, self.stock._table_y_bounds
        self.points = surface_points(model, self.bodies[0])
        self.arm_of = np.full(model.ngeom, -1, dtype=np.int64)  # geom -> 0 left arm, 1 right arm
        for a, root in enumerate(("left_arm", "right_arm")):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
            if b < 0:
                raise ValueError(f"{root} not found")
            self.arm_of[geoms_of(model, subtree(model, b))] = a
        self.finger_q = []
        for arm in ("left", "right"):
            js = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{f}_finger") for f in ("left", "right")]
            self.finger_q.append([int(model.jnt_qposadr[j]) for j in js])
        self.link6 = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_6") for arm in ("left", "right")]
        self.sign = 0.0
        self.giver = -1

    def set_side(self, side: str, randomization: object | None = None) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {side!r}")
        self.sign = 1.0 if side == "left" else -1.0
        self.giver = 0 if side == "left" else 1
        if randomization is not None:
            self.stock.configure_from_randomization(self.model, randomization)
        self.stock._start_sign = self.sign

    def heights(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.subtree_com[self.bodies[0]], dtype=np.float64).reshape(1, 3).copy()

    def world_points(self, data: mujoco.MjData) -> np.ndarray:
        b = self.bodies[0]
        return self.points @ data.xmat[b].reshape(3, 3).T + data.xpos[b]

    def observe(self, data: mujoco.MjData, rest: np.ndarray) -> dict[str, np.ndarray]:
        if self.giver < 0:
            raise RuntimeError("inhand scene: side unknown (reset the env with randomize=True)")
        cfg = self.cfg
        rest = np.asarray(rest, dtype=np.float64).reshape(1, 3)
        obs = self.contact_obs(data)
        com = self.heights(data)
        a = int(self.qpos_adr[0])
        pos = np.asarray(data.qpos[a:a + 3], dtype=np.float64).reshape(1, 3)
        d = int(self.dof_adr[0])
        speed = np.array([float(np.linalg.norm(data.qvel[d:d + 3]))])
        touch_arm = np.zeros((1, 2), dtype=bool)
        env_touch = np.zeros(1, dtype=bool)
        obj = self.owner[:, 0] == 0
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for o, other in ((g1, g2), (g2, g1)):
                if obj[o] and not obj[other]:
                    arm = self.arm_of[other]
                    if arm >= 0:
                        touch_arm[0, arm] = True
                    else:
                        env_touch[0] = True
        pts = self.world_points(data)
        centres = np.stack([data.xpos[l] + data.xmat[l].reshape(3, 3) @ PAD for l in self.link6])
        site_gap = np.linalg.norm(pts[None, :, :] - centres[:, None, :], axis=-1).min(axis=1)[None, :]  # (1, 2)
        per_arm = obs["hits"].reshape(1, 2, 2).sum(axis=2)
        A, B = self.giver, 1 - self.giver
        fq = self.finger_q[A]
        opening = 0.5 * (abs(float(data.qpos[fq[0]])) + abs(float(data.qpos[fq[1]]))) / 0.0475
        ys = pos[:, 1] * self.sign
        on_table = ((pos[:, 0] >= self.table_x[0]) & (pos[:, 0] <= self.table_x[1])
                    & (pos[:, 1] >= self.table_y[0]) & (pos[:, 1] <= self.table_y[1]))
        far_gap = (np.maximum(ys + cfg.far_m, 0.0) + np.maximum(self.table_x[0] - pos[:, 0], 0.0)
                   + np.maximum(pos[:, 0] - self.table_x[1], 0.0))
        stock = None
        if self.stock._start_sign is not None:  # the fixed stock rule, with this step's contacts and speed
            stock = self.stock._evaluate(np.asarray(data.qpos, np.float32)[None], touch_arm.any(axis=1), speed)
        obs.update(
            com=com, pos=pos, xy=com[:, :2], lift=com[:, 2] - rest[:, 2], speed=speed,
            arm_fingers=per_arm, touch_arm=touch_arm, env_touch=env_touch, site_gap=site_gap,
            f_give=per_arm[:, A], f_recv=per_arm[:, B], touch_give=touch_arm[:, A], touch_recv=touch_arm[:, B],
            gap_give=site_gap[:, A], gap_recv=site_gap[:, B], open_give=np.array([opening]),
            low=np.array([float(pts[:, 2].min()) - self.table_z]), com_above_table=com[:, 2] - self.table_z,
            ys=ys, on_table=on_table, far_gap=far_gap,
            m_gap=hs.region_gap(com, cfg.handoff.region_lo, cfg.handoff.region_hi),
            stock_done=np.asarray(stock.success, dtype=bool) if stock is not None else np.zeros(1, dtype=bool),
        )
        return obs


def classify(obs: dict[str, np.ndarray], prev: np.ndarray, cfg: StageConfig) -> np.ndarray:
    h = cfg.handoff
    out = np.full(prev.shape, R, dtype=np.int64)
    touch_or_close = lambda r: (obs["gap_give"] <= r) | (obs["f_give"] > 0)  # noqa: E731
    # Entering "near" also needs A's fingers >= 50 % open (ready to grasp; G5 repair round 2: a closed fist pushing
    # the item was "near" before it touched it). Staying near does not, so closing on the item in pick keeps it.
    near = np.where(prev == P, touch_or_close(cfg.reach_exit_m),
                    touch_or_close(cfg.reach_m) & (obs["open_give"] >= cfg.closed_open_frac))
    out[near] = P
    lifted = obs["lift"] >= cfg.lift_m
    out[(obs["f_give"] == 2) & lifted] = PR
    out[hs.handing(obs, prev == H, h)] = H
    recv = hs.received(obs, h)
    out[hs.received_in_air(obs, h) | (recv & (prev >= C) & (prev <= DONE))] = C
    far = (obs["ys"] <= np.where(prev == PL, -cfg.far_m + cfg.far_exit_m, -cfg.far_m)) & obs["on_table"]
    enter = recv & (obs["lift"] <= cfg.place_lift_m) & (prev >= C)
    stay = (prev == PL) & obs["touch_recv"]
    out[far & ~obs["touch_give"] & (enter | stay)] = PL
    free = ~(obs["touch_give"] | obs["touch_recv"]) & (obs["speed"] > cfg.at_rest_speed)
    out[free] = prev[free]
    out[obs["stock_done"]] = DONE
    off = (obs["com_above_table"] < -cfg.below_table_m) | ~obs["on_table"]
    out[off] = FALLEN
    return out


def progress(stage: int, obs: dict[str, np.ndarray], cfg: StageConfig | None = None) -> np.ndarray:
    cfg = cfg or StageConfig()
    if stage == R:
        return -obs["gap_give"]
    if stage == P:
        return 0.25 * obs["f_give"] + 0.5 * np.clip(obs["lift"] / cfg.lift_m, 0, 1) * (obs["f_give"] == 2)
    if stage == PR:
        return -obs["m_gap"]
    if stage == H:
        return hs.progress(obs, cfg.handoff)
    if stage == C:
        return -np.minimum(obs["far_gap"], 1.0)
    if stage == PL:
        return -np.clip(obs["lift"], 0, 0.3) + 0.05 * ~(obs["touch_give"] | obs["touch_recv"])
    raise ValueError(f"no progress for stage {stage}")


class InhandTracker(StageTracker):
    def __init__(self, scene: InhandScene, cfg: StageConfig | None = None) -> None:
        super().__init__(scene, classify, cfg or scene.cfg, start_label=R)
