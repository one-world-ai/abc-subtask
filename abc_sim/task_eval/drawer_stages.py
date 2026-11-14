"""Stages of open_drawer / close_drawer for single-stage RL (stage table approved 2026-11-14,
``subtask_rewards/drawer/stage_table.md``).

open_drawer:  reach -> grasp -> pull -> release -> done
close_drawer: reach -> push -> release -> done

The "objects" of the kit's tracker are the three drawers (top, middle, bottom); the episode's target is the drawer the
randomizer names. Everything is read from sim state: slide joint q, handle/finger contacts, grasp-site positions,
the gripper command. Notation (stage table):
  bar     crossbar centre = ``*_handle`` site + 1.9 cm along the pull axis
  pinch   grasp site + R (0.044, 0, -0.005): the finger-pad centre (overlay measurement)
  q       slide joint (m, + = open); others = max q of the two other drawers
  on handle      a finger of one gripper touches the drawer's stem/bar in any of the last ``contact_window`` steps
  hand on drawer any robot geom touches any geom of the drawer in any of the last ``contact_window`` steps
Contacts read at control-step boundaries chatter during quasi-static pushes (phase 0: 2 of 6 steps), hence windows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.task_eval.stagekit.robot import ARMS, GRIP_MAX
from abc_sim.task_eval.stagekit.scene import geoms_of, subtree
from abc_sim.task_eval.stagekit.tracker import StageTracker

DRAWERS = ("top", "middle", "bottom")
JOINTS = {"top": "joint_top_drawer_low", "middle": "joint_middle_drawer_low", "bottom": "joint_bottom_drawer_low"}
LINKS = {"top": "link_top_drawer_low", "middle": "link_middle_drawer_low", "bottom": "link_bottom_drawer_low"}
HANDLE_SITES = {"top": "top_handle", "middle": "mid_handle", "bottom": "bot_handle"}
BAR_OFFSET_M = 0.019  # crossbar centre in front of the handle site (scene XML, overlay drawer_geom)
PINCH = np.array([0.044, 0.0, -0.005])  # finger-pad centre in the grasp-site frame (overlay measurement)
HAND_X = (0.0, 0.065)  # gripper extent along the grasp-site x (finger) axis: site (palm) .. fingertips (1.7 cm past
# the pinch, overlay close solver). Release "away" is measured from this segment, not from the pinch: with an open
# gripper the bar can sit at the palm 5 cm behind the pinch (scripted push_back_in, repair round 2).

OPEN_STAGES = ("reach", "grasp", "pull", "release", "done")
CLOSE_STAGES = ("reach", "push", "release", "done")
OR, OG, OP, ORL, ODONE = range(5)
CR, CP, CRL, CDONE = range(4)

# Prompts are lookup keys: the fixed ones (spec prompts) carry a task prefix, the live ones name the drawer and use
# phrasings no other task spec uses (checked in tests/test_drawer_subtasks.py).
OPEN_PROMPTS = {
    "reach": "open drawer subtask: move a gripper to the drawer handle",
    "grasp": "open drawer subtask: grasp the drawer handle",
    "pull": "open drawer subtask: pull the drawer open",
    "release": "open drawer subtask: let go of the drawer",
}
CLOSE_PROMPTS = {
    "reach": "close drawer subtask: move a gripper to the front of the drawer",
    "push": "close drawer subtask: push the drawer closed",
    "release": "close drawer subtask: move the gripper away from the drawer",
}
OPEN_LIVE = {
    "reach": "move a gripper to the {target} drawer handle",
    "grasp": "grasp the {target} drawer handle",
    "pull": "pull the {target} drawer open",
    "release": "let go of the {target} drawer",
}
CLOSE_LIVE = {
    "reach": "move a gripper to the front of the {target} drawer",
    "push": "push the {target} drawer closed",
    "release": "move the gripper away from the {target} drawer",
}


@dataclass(frozen=True)
class StageConfig:
    # stock evaluator options (eval set v5): goals reuse them verbatim
    open_m: float = 0.12
    closed_m: float = 0.015
    other_max_m: float = 0.05
    # release-stage hysteresis (decision, 1 cm band; the drawer is quasi-static so it only moves while pushed)
    open_exit_m: float = 0.11
    closed_exit_m: float = 0.025
    # near: pinch within 3 cm of the bar (exit 5 cm). Overlay grasp puts the pinch 0.4 cm from the bar.
    near_m: float = 0.03
    near_exit_m: float = 0.05
    # pull entry without a grasp (approved D2: hooks count): q >= 2 cm with a hand on the drawer
    hook_open_m: float = 0.02
    # grasped needs the gripper commanded closed (env action < 0.5)
    grip_closed_cmd: float = 0.5
    # release goal: no robot contact with the target for 5 steps and the gripper (palm..fingertips) >= 5 cm from the bar
    release_far_m: float = 0.05
    release_window: int = 5
    contact_window: int = 3  # phase 0: close push at 8 cm/s had contact at 2 of 6 step boundaries
    # close: the target pulled > 3 cm past its start opening -> failure (approved D5)
    pulled_past_m: float = 0.03
    hold: int = 5  # debounce, as mug / bottles
    span: dict = field(default_factory=lambda: dict(reach=0.45, grasp=1.0, pull=1.0, push=1.0, release=1.0))
    eps: dict = field(default_factory=lambda: dict(reach=0.005, grasp=0.05, pull=0.01, push=0.01, release=0.05))


class DrawerScene:
    """The three drawers of the cabinet and the robot's fingers / grasp sites. ``observe`` returns arrays over the
    drawers; contact windows are kept here (one entry per sim time, so a repeated evaluation of the same state does
    not shorten them). ``clear()`` drops the history (episode start)."""

    def __init__(self, model: mujoco.MjModel, cfg: StageConfig) -> None:
        self.model, self.cfg = model, cfg
        self.names = list(DRAWERS)
        self.joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINTS[d]) for d in DRAWERS]
        if min(self.joints) < 0:
            raise ValueError("drawer joints not found")
        self.qpos_adr = np.asarray([int(model.jnt_qposadr[j]) for j in self.joints])
        self.dof_adr = np.asarray([int(model.jnt_dofadr[j]) for j in self.joints])
        self.bodies = [int(model.jnt_bodyid[j]) for j in self.joints]
        self.handle_sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, HANDLE_SITES[d]) for d in DRAWERS]
        # geom owner: (0, drawer k) any drawer geom, (1, finger arm*2+side), (2, arm) other robot geom
        self.owner = np.full((model.ngeom, 2), -1, dtype=np.int32)
        self.handle = np.zeros(model.ngeom, dtype=bool)  # stem + crossbar: the first two box geoms of each link
        for k, body in enumerate(self.bodies):
            self.owner[geoms_of(model, subtree(model, body))] = (0, k)
            boxes = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body
                     and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
            self.handle[boxes[:2]] = True
        for a, arm in enumerate(ARMS):
            for b in range(model.nbody):
                if (model.body(b).name or "").startswith(f"{arm}_"):
                    self.owner[geoms_of(model, {b})] = (2, a)
            for s, side in enumerate(("link_left_finger", "link_right_finger")):
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_{side}")
                self.owner[geoms_of(model, subtree(model, body))] = (1, a * 2 + s)
        self.sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grasp_site") for arm in ARMS]
        self.grip_act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_gripper") for arm in ARMS]
        self.clear()

    def clear(self) -> None:
        n = max(self.cfg.contact_window, self.cfg.release_window)
        self._hist: deque = deque(maxlen=n)
        self._hist_time = None

    def heights(self, data: mujoco.MjData) -> np.ndarray:
        """Slide joint q of each drawer (the tracker's "rest height": the episode's start openings)."""
        return np.asarray(data.qpos[self.qpos_adr], dtype=np.float64)

    def bar(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(3, 3) crossbar centres and (3, 3) pull directions (unit, world)."""
        pull = np.stack([data.xmat[b].reshape(3, 3) @ self.model.jnt_axis[j] for b, j in zip(self.bodies, self.joints)])
        return np.asarray(data.site_xpos[self.handle_sites]) + BAR_OFFSET_M * pull, pull

    def pinch(self, data: mujoco.MjData) -> np.ndarray:
        """(2, 3) finger-pad centres of the left / right gripper."""
        return np.stack([data.site_xpos[s] + data.site_xmat[s].reshape(3, 3) @ PINCH for s in self.sites])

    def hand_dist(self, data: mujoco.MjData, bar: np.ndarray) -> np.ndarray:
        """(3,) distance from each drawer's bar centre to the nearer gripper's palm-to-fingertip segment."""
        out = np.full(len(bar), np.inf)
        for s in self.sites:
            R = data.site_xmat[s].reshape(3, 3)
            a = data.site_xpos[s] + R @ np.array([HAND_X[0], 0.0, PINCH[2]])
            b = data.site_xpos[s] + R @ np.array([HAND_X[1], 0.0, PINCH[2]])
            ab = b - a
            t = np.clip(((bar - a) @ ab) / (ab @ ab), 0.0, 1.0)
            out = np.minimum(out, np.linalg.norm(bar - (a + t[:, None] * ab), axis=1))
        return out

    def contacts(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """This step: (3, 4) finger arm*2+side on drawer k's handle, (3, 2) any geom of arm a on drawer k."""
        handle = np.zeros((3, 4), dtype=bool)
        robot = np.zeros((3, 2), dtype=bool)
        for i in range(int(data.ncon)):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for d, r in ((g1, g2), (g2, g1)):
                od, orr = self.owner[d], self.owner[r]
                if od[0] != 0 or orr[0] not in (1, 2):
                    continue
                arm = int(orr[1]) // 2 if orr[0] == 1 else int(orr[1])
                robot[od[1], arm] = True
                if orr[0] == 1 and self.handle[d]:
                    handle[od[1], orr[1]] = True
        return handle, robot

    def observe(self, data: mujoco.MjData, rest: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.cfg
        handle, robot = self.contacts(data)
        t = float(data.time)
        if self._hist and self._hist_time == t:
            self._hist[-1] = (handle, robot)
        else:
            self._hist.append((handle, robot))
        self._hist_time = t
        hist = list(self._hist)
        win = hist[-cfg.contact_window:]
        hits = np.any([h for h, _ in win], axis=0)  # (3, 4)
        touch_arm = np.any([r for _, r in win], axis=0)  # (3, 2)
        touch_long = np.any([r for _, r in hist[-cfg.release_window:]], axis=0).any(axis=1)
        released = ~touch_long & (len(hist) >= cfg.release_window)
        q = self.heights(data)
        bar, pull = self.bar(data)
        pinch = self.pinch(data)
        dist_arm = np.linalg.norm(bar[:, None, :] - pinch[None, :, :], axis=-1)  # (3 drawers, 2 arms)
        per_arm = hits.reshape(3, 2, 2).sum(axis=2)
        grip_cmd = np.asarray([data.ctrl[a] / GRIP_MAX for a in self.grip_act])  # env action: 1 open, 0 closed
        closed = grip_cmd < cfg.grip_closed_cmd
        both = (per_arm == 2) & closed[None, :]
        others = np.asarray([max(q[j] for j in range(3) if j != k) for k in range(3)])
        return {
            "q": q, "q0": np.asarray(rest, dtype=np.float64), "others": others,
            "dist": dist_arm.min(axis=1), "near_arm": dist_arm.argmin(axis=1), "hand_dist": self.hand_dist(data, bar),
            "hits": hits, "fingers": per_arm.max(axis=1), "on_handle": hits.any(axis=1),
            "grasped": both.any(axis=1), "grasp_arm": np.where(both[:, 0] & both[:, 1], 2,
                                                            np.where(both[:, 0], 0, np.where(both[:, 1], 1, -1))),
            "hand_on": touch_arm.any(axis=1), "released": released, "grip_cmd": grip_cmd,
            "closure": np.clip((1.0 - grip_cmd) / (1.0 - cfg.grip_closed_cmd), 0.0, 1.0),
        }


def classify_open(obs: dict, prev: np.ndarray, cfg: StageConfig) -> np.ndarray:
    q, others_ok = obs["q"], obs["others"] <= cfg.other_max_m
    # a finger on the handle counts only with the pinch inside the exit band: an open finger of a gripper at the
    # neighbouring handle grazes the target's bar with the pinch 6.8 cm away (scripted wrong_handle, repair round 1)
    near = ((obs["dist"] <= np.where(prev >= OG, cfg.near_exit_m, cfg.near_m))
            | (obs["on_handle"] & (obs["dist"] <= cfg.near_exit_m)))
    out = np.full(prev.shape, OR, dtype=np.int64)
    out[near] = OG
    pull = obs["grasped"] | (obs["hand_on"] & ((q >= cfg.hook_open_m) | (prev >= OP)))
    out[pull] = OP
    stock = (q >= cfg.open_m) & others_ok
    release = stock | ((prev >= ORL) & (q >= cfg.open_exit_m))
    out[release] = ORL
    out[stock & obs["released"] & (obs["hand_dist"] >= cfg.release_far_m)] = ODONE
    return out


def classify_close(obs: dict, prev: np.ndarray, cfg: StageConfig) -> np.ndarray:
    q, others_ok = obs["q"], obs["others"] <= cfg.other_max_m
    # entering push by contact needs the pinch inside the exit band (fingers reaching for a neighbour hit the open
    # target's bar with the pinch 7-9 cm away, repair round 1); once pushing, any hand contact keeps it
    near = ((obs["dist"] <= np.where(prev >= CP, cfg.near_exit_m, cfg.near_m))
            | (obs["hand_on"] & ((obs["dist"] <= cfg.near_exit_m) | (prev >= CP))))
    out = np.full(prev.shape, CR, dtype=np.int64)
    out[near] = CP
    stock = (q <= cfg.closed_m) & others_ok
    release = stock | ((prev >= CRL) & (q <= cfg.closed_exit_m))
    out[release] = CRL
    out[stock & obs["released"] & (obs["hand_dist"] >= cfg.release_far_m)] = CDONE
    return out


def progress_open(stage: int, obs: dict, cfg: StageConfig) -> np.ndarray:
    if stage == OR:
        return -obs["dist"]
    if stage == OG:
        both = obs["fingers"] == 2
        return 0.25 * obs["fingers"] + 0.5 * obs["closure"][obs["near_arm"]] * both
    if stage == OP:
        return np.clip(obs["q"] / cfg.open_m, 0.0, 1.0)
    if stage == ORL:
        return np.minimum(obs["hand_dist"], cfg.release_far_m) / cfg.release_far_m
    raise ValueError(stage)


def progress_close(stage: int, obs: dict, cfg: StageConfig, start: np.ndarray) -> np.ndarray:
    if stage == CR:
        return -obs["dist"]
    if stage == CP:  # the stock progress: share of the way from the episode's start opening to closed
        return np.clip((start - obs["q"]) / np.maximum(start - cfg.closed_m, 1e-6), 0.0, 1.0)
    if stage == CRL:
        return np.minimum(obs["hand_dist"], cfg.release_far_m) / cfg.release_far_m
    raise ValueError(stage)


class DrawerTracker(StageTracker):
    def __init__(self, scene: DrawerScene, classify, cfg: StageConfig) -> None:
        super().__init__(scene, classify, cfg, start_label=0)

    def reset(self, data, rest_height=None, labels=None):
        self.scene.clear()
        return super().reset(data, rest_height=rest_height, labels=labels)


__all__ = ["DRAWERS", "JOINTS", "OPEN_STAGES", "CLOSE_STAGES", "OPEN_PROMPTS", "CLOSE_PROMPTS", "OPEN_LIVE",
           "CLOSE_LIVE", "StageConfig", "DrawerScene", "DrawerTracker", "classify_open", "classify_close",
           "progress_open", "progress_close", "PINCH", "BAR_OFFSET_M"]
