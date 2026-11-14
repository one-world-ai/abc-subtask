"""Physics-defined stages for the approved sweep hand-pick chain H: reach -> pick -> carry -> drop per target scrap.

Same contract as the approved bottles chain (``bottles_stages.py``); the bin geometry and the in-bin rule come from the
fixed ``SweepAwayEvaluator`` (tapered inner wall, bin upright within 45 deg). Only the episode's *active* scraps take
part (the randomizer parks the others below the floor); inactive scraps are never targets and never fail a stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.task_eval.stagekit.robot import ARMS, arm_ids
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching, geoms_of, subtree
from abc_sim.task_eval.stagekit.tracker import StageTracker
from abc_sim.task_eval.sweep import SweepAwayEvaluator

STAGES = ("reach", "pick", "carry", "drop", "done")
R, P, C, D, DONE = range(5)
FALLEN = len(STAGES)
LABELS = STAGES + ("fallen",)
# Centre of the fingertip pinch zone in the link_6 frame. The finger pads close 4.4 cm from ``<arm>_grasp_site``
# along link_6 -y (pads y -0.050..-0.038, z 0.067..0.147; site at (0, 0, 0.1347)); measured with
# ``subtask_rewards/sweep/probes/sw_hold.py``. ``dist`` is measured from here, not from the site: a 3.5 cm scrap
# 4.4 cm from the site is missed by both fingers.
PINCH = (0.0, -0.044, 0.115)  # z: 0.125 let 2/8 drop holds slip out in 10 idle chunks, 0.115 kept 8/8
STAGE_PROMPTS = {
    "reach": "sweep subtask: move a gripper to the target paper scrap",
    "pick": "sweep subtask: grasp the target paper scrap and lift it",
    "carry": "sweep subtask: carry the target paper scrap over the bin opening",
    "drop": "sweep subtask: release the target paper scrap into the bin",
}


@dataclass(frozen=True)
class StageConfig:
    # Thresholds of the approved bottles table unless noted.
    reach_m: float = 0.05            # grasp site -> CoM; the fingertip pinch zone is 4.4 cm from the site (measured)
    reach_exit_m: float = 0.07
    lift_m: float = 0.03
    push_m: float = 0.05             # bottles roll too; scraps (friction 0.15) as well
    fallen_m: float = 0.05           # 5 cm below rest = off the table top (table 0.74, floor far below)
    opening_margin_m: float = 0.02   # "over the opening" = r_top - 2 cm
    opening_exit_m: float = 0.02
    bin_moved_m: float = 0.05        # decision 7: stages fail at > 5 cm / > 20 deg (scorer keeps 45 deg)
    bin_tilt_rad: float = np.deg2rad(20)
    at_rest_speed: float = 0.02
    at_rest_angular_speed: float = 0.2
    hold: int = 5
    eps: dict = field(default_factory=lambda: dict(reach=0.005, pick=0.05, carry=0.005, drop=0.005))
    span: dict = field(default_factory=lambda: dict(reach=0.30, pick=1.0, carry=0.40, drop=0.10))


class SweepScene(ObjectScene):
    def __init__(self, model, cfg=None):
        super().__init__(model, free_joints_matching(model, r"trash_(\d+)_jnt"))
        self.cfg = cfg or StageConfig()
        self.geometry = SweepAwayEvaluator(model=model, spec=None)
        if self.geometry.trash_names != self.names:
            raise RuntimeError(f"scrap order differs: {self.geometry.trash_names} vs {self.names}")
        self.bin_adr = self.geometry._bin_qpos_adr
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint")
        self.bin_body = int(model.jnt_bodyid[joint])
        self.bin_geoms = set(geoms_of(model, subtree(model, self.bin_body)))
        self.link6 = [arm_ids(model, arm).link6 for arm in ARMS]
        # Until the randomizer reports the active set: every scrap above the active height counts (stock rule).
        self.active = None

    def set_active(self, mask):
        self.active = np.asarray(mask, dtype=bool).copy()
        self.geometry._active_trash_mask = self.active[None].copy()
        self.geometry._fixed_active_trash_mask = True

    def active_mask(self, data):
        if self.active is None:
            z = np.asarray([data.qpos[a + 2] for a in self.qpos_adr])
            return z > self.geometry.active_z_min
        return self.active

    def pinch_points(self, data):
        return np.asarray([data.xpos[b] + data.xmat[b].reshape(3, 3) @ PINCH for b in self.link6])

    def contact_obs(self, data):
        out = super().contact_obs(data)
        com = np.asarray(data.subtree_com[self.bodies], dtype=np.float64).reshape(-1, 3)
        dist = np.linalg.norm(com[:, None, :] - self.pinch_points(data)[None], axis=-1)
        out["dist"], out["near_arm"] = dist.min(axis=1), dist.argmin(axis=1)
        return out

    def heights(self, data):
        # Rest positions of the scraps plus the bin's original anchor (kept through exact snapshot restores).
        com = np.asarray(data.subtree_com[self.bodies]).reshape(-1, 3)
        return np.vstack((com, data.qpos[self.bin_adr:self.bin_adr + 3]))

    def in_bin(self, data):
        active = self.active_mask(data)
        if self.active is None:
            self.geometry._active_trash_mask = active[None].copy()
            self.geometry._fixed_active_trash_mask = True
        self.geometry._nworld = 1
        return self.geometry.evaluate_qpos_batch(np.asarray(data.qpos, np.float32)[None]).metrics[
            "trash_in_bin_mask"][0] & active

    def observe(self, data, rest):
        rest = np.asarray(rest).reshape(-1, 3)
        active = self.active_mask(data)
        pos = np.asarray(data.subtree_com[self.bodies]).reshape(-1, 3)
        bin_pos = data.qpos[self.bin_adr:self.bin_adr + 3]
        rotation = data.xmat[self.bin_body].reshape(3, 3)
        local = (pos - bin_pos) @ rotation
        radius = np.hypot(local[:, 0], local[:, 2])
        top = self.geometry._bin_top_y
        opening_radius = self.geometry._bin_radius_top - self.cfg.opening_margin_m
        radial_gap = np.maximum(radius - opening_radius, 0)
        gap = np.hypot(radial_gap, np.maximum(top - local[:, 1], 0))
        stock = self.in_bin(data)
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
        # Support must ultimately reach the bin (bottles rule): two airborne scraps touching are not settled.
        stable = (stock & ~contacts["touched"] & (speed <= self.cfg.at_rest_speed)
                  & (angular_speed <= self.cfg.at_rest_angular_speed))
        supported = direct & stable
        for _ in self.names:
            grown = supported | (stable & np.array([any(supported[j] for j in ns) for ns in neighbors], dtype=bool))
            if np.array_equal(grown, supported):
                break
            supported = grown
        lift = pos[:, 2] - rest[:len(pos), 2]
        return dict(xy=pos[:, :2], lift=lift, active=active,
                    speed=speed, angular_speed=angular_speed, radial=radius, radial_gap=radial_gap,
                    gap=gap, above_rim=local[:, 1] - top,
                    over=(radius <= opening_radius) & (local[:, 1] >= top), in_bin=stock, supported=supported,
                    resting_contact=resting_contact,
                    off_table=active & (lift < -self.cfg.fallen_m) & ~stock,
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
    # A released scrap keeps its stage while it falls or rolls.
    free = ~obs["touched"] & (~at_rest | ~obs["resting_contact"])
    out[free] = prev[free]
    out[obs["in_bin"] & obs["supported"] & ~obs["touched"] & at_rest] = DONE
    # The bin floor is near table height, so an in-bin scrap never becomes FALLEN.
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


class SweepTracker(StageTracker):
    def __init__(self, scene, cfg=None):
        super().__init__(scene, classify, cfg or scene.cfg, start_label=R)
