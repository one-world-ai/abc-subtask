"""Stages of turn_mug_right_side_up, for single-stage RL: what each stage is, when a state is in it, and its reward.

Each mug is in exactly one stage at any time, computed from the sim state alone (plus hysteresis on the previous
stage). The goal of a stage is entering any later one, so stages chain: a state that finishes "pick" is a valid
"rotate" start.

    stage   prompt                                   a mug is in it when
    reach   move the closest gripper to the mug      none of the below (start of the task)
    pick    grasp the mug and lift it                a grasp point is within reach_m of the mug or a finger touches
                                                     it, or it is grasped
                                                     but not yet lifted lift_m
    rotate  rotate the mug upright in the gripper    grasped, lifted >= lift_m, not upright
    place   put the mug down upright and let go      upright and still grasped/touched (in the air or on the table)
    done    -                                        upright, back at rest height, no finger touching (stock check)
    fallen  -                                        lifted < -fallen_m (off the table); never a valid start

"grasped" = both fingers of one gripper touch the mug (MuJoCo contacts). "upright" = the mug axis has world z
>= upright_z (0.95, ~18 deg, as the stock evaluator); a mug already upright stays so until z < upright_z_exit.
``MugStageTracker`` debounces: a new stage counts once it has held for ``hold`` consecutive updates.

Per-chunk reward for an RL episode of stage S (``chunk_reward``): +1 when the mug reaches a later stage (goal),
-1 when it falls back to an earlier stage or off the table, otherwise the sign of the change in S's progress
number, with a dead band ``eps[S]``:

    reach   -(distance from the closest grasp site to the mug CoM)
    pick    0.25 x (fingers of the best gripper on the mug, 0-2) + 0.5 x lift fraction while grasped
    rotate  mug axis z (upright_z)
    place   -(height above rest, clipped at 0.3 m) + 0.05 if no finger touches the mug
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching
from abc_sim.task_eval.stagekit.scene import geoms_of as _geoms_of  # noqa: F401  (kept for old imports)
from abc_sim.task_eval.stagekit.scene import subtree as _subtree  # noqa: F401
from abc_sim.task_eval.stagekit.tracker import StageTracker

STAGES = ("reach", "pick", "rotate", "place", "done")
FALLEN = len(STAGES)  # label index for a mug off the table
LABELS = STAGES + ("fallen",)
# Also task-spec prompts, which are lookup keys: each must normalize to something no other task uses.
STAGE_PROMPTS = {
    "reach": "move the closest gripper to the mug",
    "pick": "grasp the mug and lift it",
    "rotate": "rotate the mug upright in the gripper",
    "place": "put the mug down upright and let go",
}
ARMS = ("left", "right")
R, P, ROT, PL, DONE = range(5)


@dataclass(frozen=True)
class StageConfig:
    upright_z: float = 0.95  # enter upright (stock threshold)
    upright_z_exit: float = 0.90  # leave upright
    rest_lift_m: float = 0.03  # "back at rest" (stock max_lift_m)
    lift_m: float = 0.03  # lifted
    # Grasp site close enough to start grasping. When the policy grasps a mug its grasp site is 5-10 cm from the mug
    # CoM (median 7.6 cm over 576 recorded grasps), so "near" is 10 cm, or any finger touching.
    reach_m: float = 0.10
    reach_exit_m: float = 0.12
    fallen_m: float = 0.05
    hold: int = 5  # updates a new stage must persist
    eps: dict = field(default_factory=lambda: {"reach": 0.005, "pick": 0.05, "rotate": 0.02, "place": 0.005})
    # Typical progress gained over a whole stage (start -> goal): reach ~30 cm closer, pick 0 -> 1, rotate upside down
    # -> upright (-1 -> 0.95), place ~12 cm down + release. Shaping rewards are divided by this, so every stage's
    # shaping adds up to about ``shaping_total`` however many chunks it takes.
    span: dict = field(default_factory=lambda: {"reach": 0.30, "pick": 1.0, "rotate": 1.95, "place": 0.15})


class MugScene(ObjectScene):
    """Per-mug observations of a mug-flip MjData: tilt, lift, contacts per finger, grasp-site distance."""

    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model, free_joints_matching(model, r"mug_(\d+)_jnt"))
        self.mug_names = self.names
        self.mug_bodies = self.bodies

    def observe(self, data: mujoco.MjData, rest_height: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays over mugs. Needs mj_forward/mj_step to have run (contacts, site_xpos, subtree_com)."""
        m = len(self.names)
        quat = np.stack([data.qpos[a + 3:a + 7] for a in self.qpos_adr]) if m else np.zeros((0, 4))
        w, x, y, _z = quat.T if m else (np.zeros(0),) * 4
        upz = 1.0 - 2.0 * (x * x + y * y)  # world z of the mug's local z axis
        return {"upz": upz, **super().observe(data, rest_height)}


def classify(obs: dict[str, np.ndarray], prev: np.ndarray, cfg: StageConfig) -> np.ndarray:
    """Stage label per mug (index into LABELS) from one observation and the previous labels (for hysteresis)."""
    was_up = (prev == PL) | (prev == DONE)
    upright = obs["upz"] >= np.where(was_up, cfg.upright_z_exit, cfg.upright_z)
    at_rest = obs["lift"] <= cfg.rest_lift_m
    lifted = obs["lift"] >= cfg.lift_m
    near = (obs["dist"] <= np.where(prev == P, cfg.reach_exit_m, cfg.reach_m)) | obs["touched"]
    g, t = obs["grasped"], obs["touched"]
    out = np.full(prev.shape, R, dtype=np.int64)
    out[near | (g & ~lifted)] = P
    out[g & lifted & ~upright] = ROT
    out[upright & (g | (t & at_rest))] = PL
    out[upright & at_rest & ~t] = DONE
    # Upright in the air, not grasped (being dropped, or balanced on one finger): keep the previous stage.
    limbo = upright & ~at_rest & ~g
    out[limbo] = np.where(prev[limbo] == FALLEN, R, prev[limbo])
    out[obs["lift"] < -cfg.fallen_m] = FALLEN
    return out


def progress(stage: int, obs: dict[str, np.ndarray]) -> np.ndarray:
    """Progress number of ``stage`` for every mug (higher is better)."""
    if stage == R:
        return -obs["dist"]
    if stage == P:
        return 0.25 * obs["fingers"] + 0.5 * np.clip(obs["lift"] / 0.03, 0, 1) * obs["grasped"]
    if stage == ROT:
        return obs["upz"].astype(np.float64)
    if stage == PL:
        return -np.clip(obs["lift"], 0, 0.3) + 0.05 * ~obs["touched"]
    raise ValueError(f"no progress for stage {stage}")


def chunk_reward(stage: int, phi0: float, phi1: float, label1: int, cfg: StageConfig) -> tuple[int, bool]:
    """(reward, terminal) for one chunk of a stage-``stage`` episode on one mug."""
    if label1 == FALLEN or label1 < stage:
        return -1, True
    if label1 > stage:
        return 1, True
    d = phi1 - phi0
    eps = cfg.eps[STAGES[stage]]
    return (1 if d > eps else -1 if d < -eps else 0), False


class MugStageTracker(StageTracker):
    """Debounced per-mug stage labels over an episode. ``update`` returns the mugs whose committed stage changed."""

    def __init__(self, scene: MugScene, cfg: StageConfig | None = None) -> None:
        super().__init__(scene, classify, cfg or StageConfig(), start_label=R)
