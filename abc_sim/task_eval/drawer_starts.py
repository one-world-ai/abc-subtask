"""Start states of the open_drawer / close_drawer subtasks (seeds SEED_BASE[mode] + i).

Drawers are articulated (one slide joint each, quasi-static: damping 5, no spring, 53-62 g), so a start is built by
setting the slide q directly (no velocity) and IK-teleporting a gripper to a pose relative to the crossbar:
  open  reach    the reset scene
        grasp    open gripper, finger pads straddling the bar (overlay approach 15 / 30 / 22 deg below horizontal),
                 pinch 0.4-1.5 cm in front of the bar centre; no contact
        pull     as grasp, then close for 20 actions on the fixed handle: both fingers on it, q < 1 cm
        release  target q ~ U(12.5, 18) cm, then grasped as pull; stock success holds
  close reach    the reset scene (target 14-19 cm open)
        push     closed fingertips on the bar's front face (overlay pose, 30 deg, fingers stacked vertically),
                 at the reset opening
        release  target q ~ U(0, 1.4) cm, fingertips on the bar face
Each pose is kept only if the stage's own motion from it has IK (``follow_points``: pull to 15 cm, back off 10 cm,
push to closed; wrist pitch up to 20 deg allowed). The teleported arm is then held at its joint targets for ``SETTLE`` actions (sets the controller targets), and the
start is kept only if it classifies as its stage.
"""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.drawer_stages import (CLOSE_STAGES, DRAWERS, OPEN_STAGES, PINCH, DrawerScene, StageConfig,
                                             classify_close, classify_open)
from abc_sim.task_eval.stagekit import (ARMS, GRIP_MAX, STATE_SPEC, T, actuator_limits, arm_ids, axis_angle_quat,
                                       ik_link6, inv)

SEED_BASE = {"open": 980_000_000, "close": 985_000_000}
DOWN = np.array([0.0, 0.0, -1.0])
SETTLE = 6  # actions holding the teleported pose before the state is saved
CLOSE_ACTIONS = 20  # 0.7 s of closing (overlay grip time)


def frame(approach, closing) -> np.ndarray:
    """Grasp-site rotation: z = approach direction, y = finger closing direction (overlay convention)."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(closing, float) - np.dot(closing, z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def approach_dir(pull: np.ndarray, deg: float) -> np.ndarray:
    """Approach towards the cabinet (against the pull axis), tilted ``deg`` below horizontal."""
    return -pull * np.cos(np.radians(deg)) + DOWN * np.sin(np.radians(deg))


def side_dir(pull: np.ndarray) -> np.ndarray:
    s = np.cross([0.0, 0.0, 1.0], pull)
    return s / np.linalg.norm(s)


def link6_for_pinch(model: mujoco.MjModel, arm: int, pinch: np.ndarray, R: np.ndarray) -> np.ndarray:
    """link_6 pose that puts the finger-pad centre (pinch) at ``pinch`` with grasp-site rotation ``R``."""
    s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{ARMS[arm]}_grasp_site")
    T_site = np.eye(4)
    T_site[:3, :3], T_site[:3, 3] = R, pinch - R @ PINCH
    return T_site @ inv(T(model.site_pos[s], model.site_quat[s]))


def ik_pinch(model, data, arm: int, pinch, R, rng=None):
    """-> (q, pos err, rot err) for the pinch pose, within what the position actuators can command."""
    return ik_link6(model, data, ARMS[arm], link6_for_pinch(model, arm, np.asarray(pinch, float), R), rng=rng,
                    limits=actuator_limits(model, ARMS[arm]))


def pitched(R: np.ndarray, axis: np.ndarray, degs=(0, 10, -10, 20, -20)) -> list[np.ndarray]:
    """R rotated about ``axis`` (the bar's long axis) by each angle: a wrist pitch that keeps the pads on the bar."""
    out = []
    for d in degs:
        M = np.zeros(9)
        mujoco.mju_quat2Mat(M, axis_angle_quat(np.asarray(axis, float), np.radians(d)))
        out.append(M.reshape(3, 3) @ R)
    return out


def reachable(model, data, arm: int, pinch, Rs, rng=None, rot_tol: float = 0.05) -> bool:
    for R in Rs:
        q, pe, re = ik_pinch(model, data, arm, pinch, R, rng=rng)
        if q is not None and pe < 3e-3 and re < rot_tol:
            return True
    return False


def follow_points(mode: str, stage: str, pinch, R, bar, pull, q_target: float):
    """Pinch points the stage's own motion must reach from a start pose (checked when building the start, so a start
    is not a kinematic dead end): open grasp / pull - the bar pulled to 15 cm; open release, close release - backed
    off 10 cm; close push - the bar pushed 1.2 cm past closed. Each with the wrist pitched up to 20 deg."""
    a = R[:, 2]
    if mode == "open" and stage in ("grasp", "pull"):
        return [bar - a * 0.004 + pull * (0.15 - q_target)]
    if stage == "release":
        return [pinch - a * 0.10]
    if mode == "close" and stage == "push":
        return [pinch - pull * (q_target + 0.012)]
    return []


def arm_action(env, arm: int, q: np.ndarray, grip: float) -> np.ndarray:
    """``arm`` at joints q with gripper ``grip`` (1 open, 0 closed); the other arm held at its current joints."""
    act = np.zeros(14, np.float32)
    for a in range(2):
        ids = arm_ids(env.model, ARMS[a])
        act[a * 7:a * 7 + 6] = env.data.qpos[ids.qadr]
        act[a * 7 + 6] = env.data.ctrl[env._ctrl_indices[a * 7 + 6]] / GRIP_MAX
    act[arm * 7:arm * 7 + 6], act[arm * 7 + 6] = q, grip
    return act


def idle_action(env) -> np.ndarray:
    """Both arms at their current joints, grippers at their current commands (ctrl is zero right after a reset, so
    repeating the controller targets is not idle there)."""
    act = np.zeros(14, np.float32)
    for a in range(2):
        act[a * 7:a * 7 + 6] = env.data.qpos[arm_ids(env.model, ARMS[a]).qadr]
        act[a * 7 + 6] = env.data.ctrl[env._ctrl_indices[a * 7 + 6]] / GRIP_MAX
    return act


def set_drawer(env, scene: DrawerScene, k: int, q: float) -> None:
    env.data.qpos[scene.qpos_adr[k]] = q
    env.data.qvel[scene.dof_adr[k]] = 0.0
    mujoco.mj_forward(env.model, env.data)


def teleport(env, arm: int, q: np.ndarray, grip: float) -> None:
    """Put ``arm`` at joints q (velocity 0) with the fingers at the opening of ``grip`` (1 open, 0 closed)."""
    model, data = env.model, env.data
    ids = arm_ids(model, ARMS[arm])
    data.qpos[ids.qadr], data.qvel[ids.dadr] = q, 0.0
    f = GRIP_MAX * float(grip)
    data.qpos[ids.finger_qadr[0]], data.qpos[ids.finger_qadr[1]] = f, -f
    data.qvel[list(ids.finger_dadr)] = 0.0
    mujoco.mj_forward(model, data)


def arm_order(bar: np.ndarray) -> tuple[int, int]:
    """Right arm first for handles on the right half (y < 0), as the overlay scripts; the other arm as fallback."""
    return (1, 0) if bar[1] < 0 else (0, 1)


def handle_pose(scene: DrawerScene, data, k: int, mode: str, rng: np.random.Generator, *, jitter: bool = True):
    """Candidate (pinch, R, grip) poses on drawer k's crossbar: open - pads straddling the bar; close - closed
    fingertips on its front face. ``jitter`` adds the start-set variation."""
    bar, pull = scene.bar(data)
    bar, pull = bar[k], pull[k]
    side = side_dir(pull)
    out = []
    if mode == "open":
        for deg in (15, 30, 22):
            a = approach_dir(pull, deg)
            back = rng.uniform(0.004, 0.015) if jitter else 0.004
            lat = rng.uniform(-0.005, 0.005) if jitter else 0.0
            out.append((bar - a * back + side * lat, frame(a, [0, 0, 1]), 1.0))
    else:
        # fingers closing vertically (as the open grasp): the overlay's horizontal closing (wrist rolled 90 deg) has no
        # IK once the drawer is within ~8 cm of closed (rot err 0.1-1 rad at q = 0 on 6/6 seeds; vertical: 0)
        for deg in (30, 22, 40, 15):
            a = approach_dir(pull, deg)
            lat = rng.uniform(-0.008, 0.008) if jitter else 0.0
            gap = rng.uniform(0.0, 0.003) if jitter else 0.0
            # fingertips reach 1.7 cm past the pinch: on the bar's front face (overlay close solver)
            out.append((bar + pull * (0.006 + gap) - a * 0.019 + side * lat, frame(a, [0, 0, 1]), 0.0))
    return out


def place_on_handle(env, scene: DrawerScene, k: int, mode: str, rng: np.random.Generator, *, jitter: bool = True,
                    stage: str = ""):
    """IK-teleport a gripper to a handle pose whose stage motion (``follow_points``) is reachable; hold it SETTLE
    actions. -> (ok, info)."""
    model, data = env.model, env.data
    bars, pulls = scene.bar(data)
    bar, pull = bars[k], pulls[k]
    q_target = float(data.qpos[scene.qpos_adr[k]])
    snap = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, snap, STATE_SPEC)
    why = "no IK"
    for arm in arm_order(bar):
        for pinch, R, grip in handle_pose(scene, data, k, mode, rng, jitter=jitter):
            q, pe, re = ik_pinch(model, data, arm, pinch, R, rng=rng)
            if q is None or pe > 3e-3 or re > 0.05:
                why = f"no IK (pos err {pe:.4f} m, rot err {re:.3f} rad)"
                continue
            follow = follow_points(mode, stage, pinch, R, bar, pull, q_target)
            if not all(reachable(model, data, arm, p, pitched(R, side_dir(pull)), rng=rng) for p in follow):
                why = "stage motion from this pose has no IK"
                continue
            # the bottom handle is 4 cm above the table: a fully open lower finger hits it and the hand is pushed
            # 3-4 cm off the bar while settling, so try narrower openings (the bar is 2 cm tall)
            for g in ((grip, 0.6, 0.4) if grip > 0 else (grip,)):
                mujoco.mj_setState(model, data, snap, STATE_SPEC)
                teleport(env, arm, q, g)
                handle, robot = scene.contacts(data)
                if robot.any() and mode == "open":
                    why = "open gripper pose touches a drawer"
                    continue
                if robot[[j for j in range(3) if j != k]].any():
                    why = "gripper touches another drawer"
                    continue
                if touches_world(scene, data):
                    why = "gripper touches the table / cabinet"
                    continue
                act = arm_action(env, arm, q, g)
                for _ in range(SETTLE):
                    env.step(act, render_obs=False)
                return True, dict(arm=arm, q=q, ik_pos_err=round(pe, 4), ik_rot_err=round(re, 3), grip=g)
    mujoco.mj_setState(model, data, snap, STATE_SPEC)
    mujoco.mj_forward(model, data)
    return False, dict(why=why)


def touches_world(scene: DrawerScene, data) -> bool:
    """A robot geom touches something that is neither the robot nor a drawer (table, cabinet body)."""
    for i in range(int(data.ncon)):
        o1, o2 = scene.owner[int(data.contact[i].geom1)], scene.owner[int(data.contact[i].geom2)]
        if (o1[0] in (1, 2)) != (o2[0] in (1, 2)) and -1 in (o1[0], o2[0]):
            return True
    return False


def close_on_handle(env, arm: int, steps: int = CLOSE_ACTIONS) -> None:
    act = arm_action(env, arm, env.data.qpos[arm_ids(env.model, ARMS[arm]).qadr].copy(), 0.0)
    act[arm * 7:arm * 7 + 6] = env.data.ctrl[[env._ctrl_indices[i] for i in range(arm * 7, arm * 7 + 6)]]
    for _ in range(steps):
        env.step(act, render_obs=False)


def build_start(env, seed: int, mode: str, stage: str, *, rng: np.random.Generator | None = None, tries: int = 4):
    """-> (start | None, log). start: seed, stage, target, arm, state, rest_height (episode start openings), labels."""
    rng = rng or np.random.default_rng(seed)
    stages = OPEN_STAGES if mode == "open" else CLOSE_STAGES
    classify = classify_open if mode == "open" else classify_close
    want = stages.index(stage)
    cfg = StageConfig()
    logs = []
    for attempt in range(1 if stage == "reach" else tries):
        env.forget_arm_state()
        env.reset(seed=seed, randomize=True)
        ev = env._task_evaluator
        ev.reset()
        k = ev.target_index()
        scene = ev.scene
        rest = ev.start_opening()
        info = dict(attempt=attempt, target=DRAWERS[k])
        arm = -1
        if stage != "reach":
            if stage == "release":
                set_drawer(env, scene, k, rng.uniform(0.125, 0.18) if mode == "open" else rng.uniform(0.0, 0.014))
            ok, detail = place_on_handle(env, scene, k, mode, rng, stage=stage)
            info.update({key: v for key, v in detail.items() if key != "q"})
            if not ok:
                logs.append(info)
                continue
            arm = int(detail["arm"])
            if mode == "open" and stage in ("pull", "release"):
                close_on_handle(env, arm)
        scene.clear()
        obs = scene.observe(env.data, rest)
        info.update(q=round(float(obs["q"][k]), 4), pinch_bar_m=round(float(obs["dist"][k]), 4),
                    fingers=int(obs["fingers"][k]), hand_on=bool(obs["hand_on"][k]))
        if mode == "open" and stage == "pull" and not (obs["grasped"][k] and obs["q"][k] < 0.01):
            logs.append({**info, "why": "handle not gripped by both fingers with q < 1 cm"})
            continue
        if mode == "open" and stage == "release" and not obs["grasped"][k]:
            logs.append({**info, "why": "handle not gripped"})
            continue
        if mode == "close" and stage != "reach" and not (obs["hand_on"][k] or obs["dist"][k] <= cfg.near_m):
            logs.append({**info, "why": "fingertips not on the bar face"})
            continue
        labels = np.zeros(3, dtype=np.int64)
        for _ in range(3):
            labels = classify(obs, labels, cfg)
        if labels[k] != want:
            logs.append({**info, "why": f"classified as {stages[labels[k]]}"})
            continue
        if obs["others"][k] > cfg.other_max_m:
            logs.append({**info, "why": "another drawer is open"})
            continue
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
        mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
        return (dict(seed=int(seed), stage=stage, target=int(k), arm=arm, state=state, rest_height=rest,
                     labels=labels, info=info), logs + [{**info, "ok": True}])
    return None, logs


__all__ = ["SEED_BASE", "build_start", "pitched", "reachable", "follow_points", "idle_action", "arm_action", "ik_pinch", "frame", "approach_dir",
           "side_dir", "handle_pose", "place_on_handle", "set_drawer", "teleport", "link6_for_pinch"]
