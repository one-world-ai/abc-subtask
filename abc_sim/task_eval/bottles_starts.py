"""Reproducible bottle starts; over-bin starts are chained from a physical carry."""
from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.bottles_stages import R, STAGES, StageConfig, classify
from abc_sim.task_eval.put_relative_starts import topdown_grasp
from abc_sim.task_eval.stagekit import (ARMS, STATE_SPEC, T, actuator_limits, arm_ids, hold_action,
                                       hold_object, ik_link6, place_gripper, set_object_pose, subtree)

SEED_BASE = 920_000_000


def bottle_yaw(model, data, scene, k):
    bodies = [b for b in subtree(model, scene.bodies[k]) if model.body_mass[b] > 0]
    b = max(bodies, key=lambda b: model.body_mass[b])
    axis = data.ximat[b].reshape(3, 3)[:, int(np.argmin(model.body_inertia[b]))]
    return float(np.arctan2(axis[1], axis[0]) + np.pi / 2)


def idle_action(env):
    action = np.zeros(14, np.float32)
    for a, arm in enumerate(ARMS):
        action[a * 7:a * 7 + 6] = env.data.qpos[arm_ids(env.model, arm).qadr]
        action[a * 7 + 6] = env.data.ctrl[env._ctrl_indices[a * 7 + 6]] / 0.0475
    return action


def move_link(env, arm, target, *, actions=64, grip=0.0):
    """Physical Cartesian move; returns False on an unreachable waypoint, never teleports the held object."""
    model, data = env.model, env.data
    ids = arm_ids(model, ARMS[arm])
    start = T(data.xpos[ids.link6], data.xquat[ids.link6])
    qa, qb = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(qa, start[:3, :3].ravel())
    mujoco.mju_mat2Quat(qb, target[:3, :3].ravel())
    if qa @ qb < 0:
        qb = -qb
    waypoints = []
    planning = mujoco.MjData(model)
    planning.qpos[:] = data.qpos
    for fraction in np.linspace(0, 1, 9)[1:]:
        pose = np.eye(4)
        quat = (1 - fraction) * qa + fraction * qb
        quat /= np.linalg.norm(quat)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quat)
        pose[:3, :3] = matrix.reshape(3, 3)
        pose[:3, 3] = start[:3, 3] * (1 - fraction) + target[:3, 3] * fraction
        q, pe, re = ik_link6(model, planning, ARMS[arm], pose, limits=actuator_limits(model, ARMS[arm]))
        if q is None or pe > 0.008 or re > 0.2:
            return False, f"IK path unreachable (position {pe:.4f}, rotation {re:.3f})"
        waypoints.append(q)
        planning.qpos[ids.qadr] = q
    other = idle_action(env)
    for q in waypoints:
        action = other.copy()
        action[arm * 7:arm * 7 + 6] = q
        action[arm * 7 + 6] = grip
        for _ in range(max(1, actions // 8)):
            env.step(action, render_obs=False)
    return True, ""


def carry_to_opening(env, scene, k, arm, *, tilt_deg=50):
    """Lift, lean towards the bin, then traverse above the rim, keeping the grasp rigid in the plan."""
    from abc_sim.task_eval.stagekit import axis_angle_quat, grasp_from_state

    data, model = env.data, env.model
    ids = arm_ids(model, ARMS[arm])
    grasp = grasp_from_state(model, data, scene, k, arm)
    com = data.subtree_com[scene.bodies[k]].copy()
    obj = scene.object_T(data, k)
    bin_pos = data.qpos[scene.bin_adr:scene.bin_adr + 3]
    rotation = data.xmat[scene.bin_body].reshape(3, 3)
    rim = bin_pos + rotation @ [0, scene.geometry._bin_top_y, 0]
    # Lean away from the arm's base so link_6 can reach a bottle held above the high rim.
    base = np.array([0.252, 0.31 if arm == 0 else -0.31, 0])
    towards = rim - base
    axis = np.array([-towards[1], towards[0], 0.0])
    axis /= np.linalg.norm(axis)
    qr = axis_angle_quat(axis, np.deg2rad(tilt_deg))
    rflat = np.empty(9)
    mujoco.mju_quat2Mat(rflat, qr)
    rotated = rflat.reshape(3, 3) @ obj[:3, :3]
    offset = obj[:3, :3].T @ (obj[:3, 3] - com)
    # Clear the bottle's lower end before moving over the wall.
    clearance = float(scene.geometry._bottle_reaches[k]) + 0.025
    high = max(com[2], rim[2] + clearance)
    for location, orientation in ((np.array([com[0], com[1], min(high, com[2] + 0.08)]), rotated),
                                   (np.array([rim[0], rim[1], high]), rotated)):
        desired = np.eye(4)
        desired[:3, :3] = orientation
        desired[:3, 3] = location + orientation @ offset
        ok, why = move_link(env, arm, desired @ grasp["T_obj_link6"], actions=96)
        if not ok:
            return False, why
        if not scene.contact_obs(data)["grasped"][k]:
            return False, "bottle slipped during carry"
    action = idle_action(env)
    for _ in range(32):
        env.step(action, render_obs=False)
    obs = scene.observe(data, env._task_evaluator.tracker.rest_height)
    return bool(obs["over"][k] and obs["grasped"][k]), "carry end is not held over opening"


def sampled_drop_hold(env, scene, k, *, tilt_deg, rng):
    """Fallback grasp source 3: synthesize an over-bin grasp, then verify it under gravity."""
    from abc_sim.task_eval.stagekit import axis_angle_quat, qmul
    data = env.data
    adr = scene.qpos_adr[k]
    bin_pos = data.qpos[scene.bin_adr:scene.bin_adr + 3].copy()
    arm = 0 if bin_pos[1] > 0 else 1
    yaw = bottle_yaw(env.model, data, scene, k) + (np.pi if tilt_deg > 0 else 0)
    grasp = topdown_grasp(data, scene, k, yaw, arm)
    base = np.array([0.252, 0.31 if arm == 0 else -0.31])
    to_bin = bin_pos[:2] - base
    axis = np.array([-to_bin[1], to_bin[0], 0.0])
    axis /= np.linalg.norm(axis)
    qr = axis_angle_quat(axis, np.deg2rad(tilt_deg))
    flat = np.empty(9)
    mujoco.mju_quat2Mat(flat, qr)
    rotation = flat.reshape(3, 3)
    com = data.subtree_com[scene.bodies[k]].copy()
    rim = bin_pos + data.xmat[scene.bin_body].reshape(3, 3) @ [0, scene.geometry._bin_top_y, 0]
    desired = np.array([rim[0], rim[1], rim[2] + rng.uniform(0.06, 0.10)])
    origin = desired + rotation @ (data.qpos[adr:adr + 3] - com)
    quat = qmul(qr, data.qpos[adr + 3:adr + 7].copy())
    set_object_pose(env, scene, k, origin, quat)
    ok, detail = hold_object(env, scene, k, grasp, rng=rng)
    detail["source"] = "sampled over-bin grasp (fallback after carry chain)"
    detail["arm"] = arm
    return ok, detail, arm


def build_start(env, seed, stage, *, rng=None, tries=8):
    rng = rng or np.random.default_rng(seed)
    logs = []
    cfg = StageConfig()
    for attempt in range(1 if stage == "reach" else tries):
        env.forget_arm_state()
        env.reset(seed=seed, randomize=True)
        env.evaluate_task()
        scene = env._task_evaluator.scene
        rest = scene.heights(env.data)
        obs = scene.observe(env.data, rest)
        k = env._task_evaluator.pick_target(obs)
        arm = 0 if obs["xy"][k, 1] > 0 else 1
        info = dict(attempt=attempt, target=k, arm=arm)
        if stage != "reach":
            # A parallel-jaw grasp has two equivalent yaw directions. Try both wrist configurations.
            yaw = bottle_yaw(env.model, env.data, scene, k) + rng.uniform(-0.15, 0.15) + (np.pi if attempt >= 4 else 0)
            grasp = topdown_grasp(env.data, scene, k, yaw, arm,
                                  below_com_m=0.0 if stage == "pick" else 0.01)
            if stage == "pick":
                ok, detail = place_gripper(env, scene, k, grasp, rng=rng)
                if ok:
                    action = idle_action(env)
                    action[arm * 7 + 6] = 1
                    for _ in range(10):
                        env.step(action, render_obs=False)
                    ok = not scene.contact_obs(env.data)["touched"][k]
                    if not ok:
                        detail["why"] = "open fingers touch the bottle"
            else:
                adr = scene.qpos_adr[k]
                pos = env.data.qpos[adr:adr + 3].copy()
                pos[2] += rng.uniform(0.06, 0.12)
                set_object_pose(env, scene, k, pos, env.data.qpos[adr + 3:adr + 7].copy())
                ok, detail = hold_object(env, scene, k, grasp, rng=rng)
                if ok and stage == "drop":
                    # The previous stage's physical solver is the source, not an over-bin teleport.
                    ok, why = carry_to_opening(env, scene, k, arm,
                                              tilt_deg=(40, -40, 50, -50, 60, -60, 30, -30)[attempt % 8])
                    if not ok:
                        detail["why"] = why
                        # Chaining failed; use the skill's next available source rather than weakening the goal.
                        chain_why = why
                        ok, detail, arm = sampled_drop_hold(env, scene, k,
                                                            tilt_deg=(-40, 40, -50, 50)[attempt % 4], rng=rng)
                        detail["chain_failure"] = chain_why
            info.update({key: value for key, value in detail.items() if key != "q"})
            if not ok:
                logs.append(info)
                continue
        obs = scene.observe(env.data, rest)
        labels = classify(obs, np.full(len(scene.names), R), cfg)
        if labels[k] != STAGES.index(stage):
            logs.append({**info, "why": f"classified as {STAGES[labels[k]] if labels[k] < len(STAGES) else 'fallen'}"})
            continue
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
        mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
        return dict(seed=int(seed), stage=stage, target=int(k), arm=arm if stage != "reach" else -1,
                    state=state, rest_height=rest, labels=labels, info=info), logs + [{**info, "ok": True}]
    return None, logs
