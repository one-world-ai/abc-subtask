"""Reproducible sweep (chain H) starts; over-bin starts are chained from a physical carry.

Grasps are centred on the fingertip *pinch zone*, not the kit's grasp site: on the YAM gripper the finger pads close
4.4 cm from ``<arm>_grasp_site`` along link_6 -y (measured, ``subtask_rewards/sweep/probes/sw_hold.py``). Bottles are
long enough not to notice; a 3.5 cm paper ball held at the site is missed by both fingers.
"""
from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.bottles_starts import idle_action, move_link
from abc_sim.task_eval.stagekit import (ARMS, GRIP_MAX, STATE_SPEC, T, arm_ids, axis_angle_quat, grasp_from_state,
                                       hold_object, inv, place_gripper, set_object_pose)
from abc_sim.task_eval.sweep_stages import PINCH as _PINCH, R, STAGES, StageConfig, classify

SEED_BASE = 970_000_000
R_DOWN = np.diag([1.0, -1.0, -1.0])        # link_6 z (towards the fingers) -> world -z
PINCH = np.array(_PINCH)                   # pinch-zone centre in the link_6 frame (pads y -0.05..-0.038)
# On the table the fingertips (link_6 z 0.147) must stay above the tabletop: centre the scrap 1 cm from the tips.
TABLE_PINCH = np.array([0.0, -0.044, 0.137])
ARM_BASES = np.array([[0.252, 0.31, 0.0], [0.252, -0.31, 0.0]])


def scrap_radius(model, scene, k):
    """Half the largest extent of the scrap's collision hull (m)."""
    from abc_sim.task_eval.stagekit.scene import geoms_of, subtree
    pts = []
    for g in geoms_of(model, subtree(model, scene.bodies[k])):
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            mesh = int(model.geom_dataid[g])
            v = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            pts.append(v @ rot.reshape(3, 3).T + model.geom_pos[g])
    pts = np.concatenate(pts)
    return float(0.5 * np.max(pts.max(0) - pts.min(0)))


def pinch_grasp(data, scene, k, yaw, arm, *, pinch=PINCH, tilt=None):
    """link_6 pointing down (optionally leaned by ``tilt`` = (axis, rad)), the scrap's CoM at the pinch zone."""
    c = np.asarray(data.subtree_com[scene.bodies[k]], dtype=np.float64)
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    rot = rz @ R_DOWN
    if tilt is not None:
        flat = np.empty(9)
        mujoco.mju_quat2Mat(flat, axis_angle_quat(np.asarray(tilt[0], float), float(tilt[1])))
        rot = flat.reshape(3, 3) @ rot
    t6 = np.eye(4)
    t6[:3, :3] = rot
    t6[:3, 3] = c - rot @ pinch
    return {"arm": int(arm), "T_obj_link6": inv(scene.object_T(data, k)) @ t6,
            "fingers": np.array([GRIP_MAX, GRIP_MAX]), "yaw": float(yaw)}


def yaw_candidates(data, scene, k, arm):
    """Top-down wrist yaws in order of preference. Reachable for the reset scraps on the same-side arm: everything
    except roughly -2.0..-0.4 rad (16-yaw scan, seeds 970M+0..5)."""
    return [0.0, 0.39, -0.39, 0.79, 2.75, -2.75, np.pi, 1.18, 1.57, 1.96, 2.36, -2.36]


# Lean angles for the carry, in order. Negative leans plan from the held starts; 0 and positive mostly do not
# (IK path unreachable; seeds 970M+0..3, dbg_carry).
CARRY_TILTS = (-30, -45, -40, -20, -60, 30, 45, 40)


SETTLE_ACTIONS = 128  # reset scraps still roll 1-4 cm for ~100 actions (seeds 970M+0..5); measure after this


def settled_reset(env, seed):
    """reset(seed) as a fresh episode, then hold the arms still until the scraps stop rolling."""
    env.forget_arm_state()
    env.reset(seed=seed, randomize=True)
    env.evaluate_task()
    action = idle_action(env)
    for _ in range(SETTLE_ACTIONS):
        env.step(action, render_obs=False)


def carry_to_opening(env, scene, k, arm, *, tilt_deg=40, height_m=0.05):
    """Lift, lean away from the arm base, then traverse above the rim keeping the grasp rigid in the plan."""
    data, model = env.data, env.model
    grasp = grasp_from_state(model, data, scene, k, arm)
    com = data.subtree_com[scene.bodies[k]].copy()
    obj = scene.object_T(data, k)
    bin_pos = data.qpos[scene.bin_adr:scene.bin_adr + 3]
    rotation = data.xmat[scene.bin_body].reshape(3, 3)
    rim = bin_pos + rotation @ [0, scene.geometry._bin_top_y, 0]
    towards = rim - ARM_BASES[arm]
    axis = np.array([-towards[1], towards[0], 0.0])
    axis /= np.linalg.norm(axis)
    flat = np.empty(9)
    mujoco.mju_quat2Mat(flat, axis_angle_quat(axis, np.deg2rad(tilt_deg)))
    rotated = flat.reshape(3, 3) @ obj[:3, :3]
    offset = obj[:3, :3].T @ (obj[:3, 3] - com)
    high = max(com[2], rim[2] + scrap_radius(model, scene, k) + height_m)
    for location, orientation in ((np.array([com[0], com[1], high]), rotated),
                                  (np.array([rim[0], rim[1], high]), rotated)):
        desired = np.eye(4)
        desired[:3, :3] = orientation
        desired[:3, 3] = location + orientation @ offset
        ok, why = move_link(env, arm, desired @ grasp["T_obj_link6"], actions=96)
        if not ok:
            return False, why
        if not scene.contact_obs(data)["grasped"][k]:
            return False, "scrap slipped during carry"
    action = idle_action(env)
    for _ in range(32):
        env.step(action, render_obs=False)
    obs = scene.observe(data, env._task_evaluator.tracker.rest_height)
    return bool(obs["over"][k] and obs["grasped"][k]), "carry end is not held over opening"


def sampled_drop_hold(env, scene, k, *, tilt_deg, rng):
    """Fallback grasp source 3: a leaning pinch grasp over the opening, verified under gravity."""
    data = env.data
    bin_pos = data.qpos[scene.bin_adr:scene.bin_adr + 3].copy()
    arm = 0 if bin_pos[1] > 0 else 1
    rim = bin_pos + data.xmat[scene.bin_body].reshape(3, 3) @ [0, scene.geometry._bin_top_y, 0]
    adr = scene.qpos_adr[k]
    com = data.subtree_com[scene.bodies[k]].copy()
    want = np.array([rim[0], rim[1], rim[2] + scrap_radius(env.model, scene, k) + rng.uniform(0.03, 0.07)])
    set_object_pose(env, scene, k, data.qpos[adr:adr + 3] - com + want, data.qpos[adr + 3:adr + 7].copy())
    towards = rim - ARM_BASES[arm]
    axis = np.array([-towards[1], towards[0], 0.0])
    axis /= np.linalg.norm(axis)
    last = {}
    for yaw in yaw_candidates(data, scene, k, arm):
        grasp = pinch_grasp(data, scene, k, yaw, arm, tilt=(axis, np.deg2rad(tilt_deg)))
        ok, detail = hold_object(env, scene, k, grasp, rng=rng)
        if ok:
            detail["source"] = "sampled over-bin grasp (fallback after carry chain)"
            return True, detail, arm
        last = detail
    return False, last, arm


def open_around(env, scene, k, arm, rng):
    """Pick start: open gripper teleported around the resting scrap; the open fingers must not touch it."""
    last = {}
    for yaw in yaw_candidates(env.data, scene, k, arm):
        grasp = pinch_grasp(env.data, scene, k, yaw + rng.uniform(-0.1, 0.1), arm, pinch=TABLE_PINCH)
        ok, detail = place_gripper(env, scene, k, grasp, rng=rng)
        if not ok:
            last = detail
            continue
        action = idle_action(env)
        action[arm * 7 + 6] = 1
        for _ in range(10):
            env.step(action, render_obs=False)
        c = scene.contact_obs(env.data)
        if c["touched"][k]:
            last = {**detail, "why": "open fingers touch the scrap"}
            continue
        return True, {**detail, "yaw": float(grasp["yaw"])}
    return False, last


def held_above(env, scene, k, arm, rng, lift=(0.06, 0.12)):
    """Carry start: the scrap lifted above its rest spot and held in a pinch grasp, checked under gravity."""
    adr = scene.qpos_adr[k]
    pos = env.data.qpos[adr:adr + 3].copy()
    pos[2] += rng.uniform(*lift)
    set_object_pose(env, scene, k, pos, env.data.qpos[adr + 3:adr + 7].copy())
    last = {}
    for yaw in yaw_candidates(env.data, scene, k, arm):
        grasp = pinch_grasp(env.data, scene, k, yaw, arm)
        ok, detail = hold_object(env, scene, k, grasp, rng=rng)
        if ok:
            return True, detail
        last = detail
    return False, last


def build_start(env, seed, stage, *, rng=None, tries=8):
    rng = rng or np.random.default_rng(seed)
    logs = []
    cfg = StageConfig()
    for attempt in range(1 if stage == "reach" else tries):
        settled_reset(env, seed)
        scene = env._task_evaluator.scene
        rest = scene.heights(env.data)
        obs = scene.observe(env.data, rest)
        k = env._task_evaluator.pick_target(obs)
        arm = 0 if obs["xy"][k, 1] > 0 else 1
        info = dict(attempt=attempt, target=k, arm=arm)
        if stage != "reach":
            if stage == "pick":
                ok, detail = open_around(env, scene, k, arm, rng)
            else:
                ok, detail = held_above(env, scene, k, arm, rng)
                if ok and stage == "drop":
                    # The previous stage's physical solver is the source, not an over-bin teleport.
                    ok, why = carry_to_opening(env, scene, k, arm,
                                               tilt_deg=CARRY_TILTS[attempt % len(CARRY_TILTS)])
                    if not ok:
                        settled_reset(env, seed)
                        scene = env._task_evaluator.scene
                        ok, detail, arm = sampled_drop_hold(env, scene, k,
                                                            tilt_deg=(-40, 40, -50, 50)[attempt % 4], rng=rng)
                        detail["chain_failure"] = why
                    else:
                        detail["source"] = "carry chain"
            info.update({key: value for key, value in detail.items() if key != "q"})
            info["arm"] = arm
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
