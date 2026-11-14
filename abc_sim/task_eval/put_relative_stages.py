"""Stages of put_relative, for single-stage RL: what each stage is, when a state is in it, and its progress number.

The episode's directive ("put the lemon behind the coffee cup") names a mover, a reference and a direction; only
the mover is staged. Its stage comes from the sim state alone (plus hysteresis on the previous stage):

    stage    a mover is in it when
    reach    none of the below (start of the task)
    pick     a grasp site is within reach_m of its CoM or a finger touches it, or it is grasped but lifted < lift_m
    handoff  cross-table episodes only (below): lifted and held by the giver, or the receiver has a finger on it,
             until it is received
    carry    grasped and lifted >= lift_m, not yet over the goal zone (cross-table: received in the air by the
             receiver, giver clear)
    place    xy inside the goal zone and held or touched (cross-table: by the receiver, giver not touching it)
    done     the stock success (zone + no higher than rest_lift_m above rest) with no finger touching, at rest
    fallen   more than fallen_m below its rest height (off the table); never a valid start

Cross-table episodes (amendment 2026-11-14, subtask_rewards/put_relative/stage_table.md): the zone point (14 cm along
the prompted direction from the reference's rest spot) has the other y sign than the mover's rest spot, so one arm
cannot carry it there. Giver G = the arm on the mover's side (left for y > 0), receiver V = the other one. Their
handoff stage is the task-independent ``handoff_stage`` (region = the whole work space, so the pick goal "lifted
and held by G" is already a handoff state; "received" = V holds it with both fingers, G touches nothing of it and
G's pinch centre is >= 6 cm from its collision box). Same-side episodes have no handoff label and are classified
exactly as before.

Goal zone = the stock ``PutRelativeEvaluator`` cone, measured from the reference's *live* position: along the
prompted axis between along_min_m and along_max_m, within perp_max_m across it. A mover moving freely (no finger
on it and not at rest: being dropped or sliding) keeps its previous stage.

Progress (higher is better; only shapes the reward):
    reach   -(distance from the nearest grasp site to the mover's CoM)
    pick    0.25 x fingers on it (0-2) + 0.5 x lift fraction while grasped
    handoff ``handoff_stage.progress`` + 0.25 x (1 - min(distance of the CoM to the middle box / 0.25 m, 1))
    carry   -(xy distance to the goal zone)
    place   -(height above rest, clipped at 0.3 m) + 0.05 if no finger touches it
Thresholds: see ``StageConfig``; the design and its checks are in subtask_rewards/put_relative/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.put_relative_handoff import PAD, arm_map, box_gap, collision_geoms, sides
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching
from abc_sim.task_eval.stagekit.tracker import StageTracker

STAGES = ("reach", "pick", "handoff", "carry", "place", "done")
FALLEN = len(STAGES)
LABELS = STAGES + ("fallen",)
R, P, H, C, PL, DONE = range(6)
# Label numbers before the handoff stage existed (start sets saved then): reach 0, pick 1, carry 2, place 3, done 4,
# fallen 5.
OLD_LABELS = {0: R, 1: P, 2: C, 3: PL, 4: DONE, 5: FALLEN}
# Task-spec prompts are lookup keys, so they must be unique; the live prompt of an episode is ``live_prompt``.
STAGE_PROMPTS = {
    "reach": "put relative subtask: move a gripper to the object to move",
    "pick": "put relative subtask: grasp the object to move and lift it",
    "handoff": "put relative subtask: hand the object to the other arm",
    "carry": "put relative subtask: carry the object to its goal",
    "place": "put relative subtask: put the object down at its goal",
}
_LIVE = {
    "reach": "move a gripper to the {mover}",
    "pick": "grasp the {mover} and lift it",
    "handoff": "hand the {mover} over to the other arm",
    "carry": "carry the {mover} {phrase} the {reference}",
    "place": "put the {mover} down {phrase} the {reference}",
}
PHRASES = {"front": "in front of", "behind": "behind", "left": "to the left of", "right": "to the right of"}


def live_prompt(stage: str, goal: dict) -> str:
    return _LIVE[stage].format(mover=goal.get("mover_label", "object"), reference=goal.get("reference_label", "object"),
                               phrase=PHRASES.get(goal.get("direction", ""), "next to"))


@dataclass(frozen=True)
class StageConfig:
    # Top-down grasps put the grasp site at the mover's CoM (0-1 cm); 5 cm leaves room for side grasps of small
    # objects (all movers are 12-30 g household items, 3-8 cm). Re-measure once solver grasps exist.
    reach_m: float = 0.05
    reach_exit_m: float = 0.07
    lift_m: float = 0.03  # lifted (as mug flip)
    rest_lift_m: float = 0.03  # "placed": the stock placed_z_tolerance_m
    fallen_m: float = 0.05
    at_rest_speed: float = 0.02  # m/s, linear speed of a released mover that counts as at rest
    zone_exit_m: float = 0.01  # a mover in the zone stays in until it is this far outside the cone
    along_min_m: float = 0.06  # the stock cone
    along_max_m: float = 0.30
    perp_max_m: float = 0.08
    push_m: float = 0.03  # reach/pick: mover moved this far (xy) while not held = pushed
    ref_moved_m: float = 0.03  # reference moved this far (xy) from its rest spot = failure
    hold: int = 5
    # Cross-table handoff (``handoff_stage``): the region is the whole work space above the table (the handoff can
    # happen wherever the arms meet; the pick goal is already "handing"); thresholds as inhand (clear 6 cm from the
    # surface, lifted 3 cm, dropped = free and > 4 cm up = put_relative's rest_lift_m + drop margin).
    handoff: hs.HandoffConfig = field(default_factory=lambda: hs.HandoffConfig(
        region_lo=(0.20, -0.60, 0.70), region_hi=(1.00, 0.60, 1.60), region_exit_m=0.0))
    # Middle box for the handoff progress term (inhand's handoff region: scripted handoffs meet there).
    mid_lo: tuple = (0.45, -0.10, 0.83)
    mid_hi: tuple = (0.70, 0.10, 1.15)
    mid_span_m: float = 0.25
    eps: dict = field(default_factory=lambda: {"reach": 0.005, "pick": 0.05, "handoff": 0.02, "carry": 0.005,
                                               "place": 0.005})
    # Typical progress over a whole stage: reach ~30 cm closer, pick 0 -> 1, handoff 0 -> 1.5, carry ~30 cm,
    # place ~10 cm down.
    span: dict = field(default_factory=lambda: {"reach": 0.30, "pick": 1.0, "handoff": 1.5, "carry": 0.30,
                                                "place": 0.10})


class PutRelativeScene(ObjectScene):
    """The put_relative objects (``relative_obj_<i>_joint``). ``heights`` returns full rest *positions* (m, 3) so the
    rest reference carries xy too (for push / reference-moved checks); ``observe`` adds the goal-zone geometry of
    every object relative to the reference, once ``set_goal`` was called."""

    def __init__(self, model: mujoco.MjModel, cfg: StageConfig | None = None) -> None:
        super().__init__(model, free_joints_matching(model, r"relative_obj_(\d+)_joint"))
        self.cfg = cfg or StageConfig()
        self.goal: dict | None = None
        self.mover = self.reference = -1
        self.arm_of = arm_map(model)
        self.obj_geoms = [collision_geoms(model, b) for b in self.bodies]
        self.link6 = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{a}_link_6") for a in ("left", "right")]

    def set_goal(self, goal: dict) -> None:
        self.goal = goal
        self.mover = self.names.index(goal["mover"])
        self.reference = self.names.index(goal["reference"])

    def heights(self, data: mujoco.MjData) -> np.ndarray:  # rest reference = positions
        return np.asarray([data.qpos[a:a + 3] for a in self.qpos_adr], dtype=np.float64).reshape(-1, 3)

    def observe(self, data: mujoco.MjData, rest: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.cfg
        rest = np.asarray(rest, dtype=np.float64).reshape(-1, 3)
        pos = self.heights(data)
        speed = np.asarray([np.linalg.norm(data.qvel[d:d + 3]) for d in self.dof_adr], dtype=np.float64)
        obs = {"xy": pos[:, :2], "lift": pos[:, 2] - rest[:, 2], "moved_xy": np.linalg.norm(pos[:, :2] - rest[:, :2], axis=1),
               "speed": speed, **self.contact_obs(data)}
        m = len(self.names)
        if self.goal is None or self.reference < 0:
            obs.update(along=np.zeros(m), perp=np.zeros(m), gap=np.full(m, np.inf), ref_moved=0.0)
            return obs
        axis = np.asarray(self.goal["axis"], dtype=np.float64)
        rel = pos[:, :2] - pos[self.reference, :2]
        along = rel @ axis
        perp = np.abs(rel @ np.array([-axis[1], axis[0]]))
        gap = np.hypot(along - np.clip(along, cfg.along_min_m, cfg.along_max_m),
                       perp - np.minimum(perp, cfg.perp_max_m))
        obs.update(along=along, perp=perp, gap=gap, ref_moved=float(obs["moved_xy"][self.reference]))
        cross, giver, _ = sides(rest, self.mover, self.reference, axis)
        obs["cross"] = np.full(m, cross)
        if cross:
            obs.update(self.handoff_obs(data, giver, obs["hits"]))
        return obs

    def handoff_obs(self, data: mujoco.MjData, giver: int, hits: np.ndarray) -> dict[str, np.ndarray]:
        """The ``handoff_stage`` observation arrays (G = ``giver``, V = the other arm), plus the CoM's distance to
        the middle box."""
        m = len(self.names)
        touch_arm = np.zeros((m, 2), dtype=bool)
        env_touch = np.zeros(m, dtype=bool)
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for o, other in ((g1, g2), (g2, g1)):
                if self.owner[o, 0] != 0:
                    continue
                k = int(self.owner[o, 1])
                if self.owner[other, 0] == 0 and int(self.owner[other, 1]) == k:
                    continue
                arm = int(self.arm_of[other])
                if arm >= 0:
                    touch_arm[k, arm] = True
                else:
                    env_touch[k] = True
        per_arm = hits.reshape(m, 2, 2).sum(axis=2)
        pad = data.xpos[self.link6[giver]] + data.xmat[self.link6[giver]].reshape(3, 3) @ PAD
        gap_give = np.array([float(box_gap(self.model, data, g, pad[None])[0]) for g in self.obj_geoms])
        com = np.asarray(data.subtree_com[self.bodies], dtype=np.float64).reshape(m, 3)
        G, V = giver, 1 - giver
        return {"com": com, "f_give": per_arm[:, G], "f_recv": per_arm[:, V], "touch_give": touch_arm[:, G],
                "touch_recv": touch_arm[:, V], "gap_give": gap_give, "env_touch": env_touch,
                "mid_gap": hs.region_gap(com, self.cfg.mid_lo, self.cfg.mid_hi)}


def classify(obs: dict[str, np.ndarray], prev: np.ndarray, cfg: StageConfig) -> np.ndarray:
    """Stage label per object (index into LABELS) from one observation and the previous labels (hysteresis)."""
    if obs.get("cross") is not None and bool(np.any(obs["cross"])):
        return classify_cross(obs, prev, cfg)
    was_in = (prev == PL) | (prev == DONE)
    in_zone = obs["gap"] <= np.where(was_in, cfg.zone_exit_m, 0.0)
    strictly_in = obs["gap"] <= 0.0
    lifted = obs["lift"] >= cfg.lift_m
    low = obs["lift"] <= cfg.rest_lift_m
    near = (obs["dist"] <= np.where(prev == P, cfg.reach_exit_m, cfg.reach_m)) | obs["touched"]
    g, t = obs["grasped"], obs["touched"]
    at_rest = obs["speed"] <= cfg.at_rest_speed
    out = np.full(prev.shape, R, dtype=np.int64)
    out[near | (g & ~lifted)] = P
    out[g & lifted] = C
    out[in_zone & (g | t)] = PL
    out[strictly_in & low & ~t & at_rest] = DONE
    # Moving freely (dropped, sliding): keep the previous stage until it is held or comes to rest.
    free = ~t & ~at_rest
    out[free] = np.where(prev[free] == FALLEN, R, prev[free])
    out[obs["lift"] < -cfg.fallen_m] = FALLEN
    return out


def classify_cross(obs: dict[str, np.ndarray], prev: np.ndarray, cfg: StageConfig) -> np.ndarray:
    """Cross-table episodes: reach and pick as in ``classify``; lifted and held by G (or V has a finger on it) =
    handoff; received in the air = carry (stays carry while received); over the zone with a V finger on it and G not
    touching it = place; done as same-side."""
    h = cfg.handoff
    was_in = (prev == PL) | (prev == DONE)
    in_zone = obs["gap"] <= np.where(was_in, cfg.zone_exit_m, 0.0)
    strictly_in = obs["gap"] <= 0.0
    lifted = obs["lift"] >= cfg.lift_m
    low = obs["lift"] <= cfg.rest_lift_m
    near = (obs["dist"] <= np.where(prev == P, cfg.reach_exit_m, cfg.reach_m)) | obs["touched"]
    g, t = obs["grasped"], obs["touched"]
    robot = obs["touch_give"] | obs["touch_recv"]
    at_rest = obs["speed"] <= cfg.at_rest_speed
    out = np.full(prev.shape, R, dtype=np.int64)
    out[near | (g & ~lifted)] = P
    out[g & lifted] = H
    out[hs.handing(obs, prev == H, h)] = H
    recv = hs.received(obs, h)
    out[hs.received_in_air(obs, h) | (recv & (prev >= C) & (prev <= DONE))] = C
    out[in_zone & ~obs["touch_give"] & (obs["f_recv"] >= 1)] = PL
    out[strictly_in & low & ~t & ~robot & at_rest] = DONE
    free = ~t & ~robot & ~at_rest
    out[free] = np.where(prev[free] == FALLEN, R, prev[free])
    out[obs["lift"] < -cfg.fallen_m] = FALLEN
    return out


def progress(stage: int, obs: dict[str, np.ndarray], cfg: StageConfig | None = None) -> np.ndarray:
    if stage == R:
        return -obs["dist"]
    if stage == P:
        return 0.25 * obs["fingers"] + 0.5 * np.clip(obs["lift"] / 0.03, 0, 1) * obs["grasped"]
    if stage == H:
        if "f_recv" not in obs:  # same-side episode: no handoff
            return np.zeros(len(obs["lift"]))
        cfg = cfg or StageConfig()
        return hs.progress(obs, cfg.handoff) + 0.25 * (1.0 - np.minimum(obs["mid_gap"] / cfg.mid_span_m, 1.0))
    if stage == C:
        return -np.minimum(obs["gap"], 1.0)
    if stage == PL:
        return -np.clip(obs["lift"], 0, 0.3) + 0.05 * ~obs["touched"]
    raise ValueError(f"no progress for stage {stage}")


class PutRelativeTracker(StageTracker):
    def __init__(self, scene: PutRelativeScene, cfg: StageConfig | None = None) -> None:
        super().__init__(scene, classify, cfg or scene.cfg, start_label=R)
