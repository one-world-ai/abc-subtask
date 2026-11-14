"""Reproducible dish-rack starts (grasp source 3 and chaining) and the scripted motions they chain through.

carry    an upright plate pinched at its rim by one gripper, lowest point 3-6 cm above the table at the plate's own
         spot, random yaw; closed on the weightless plate, then held under gravity (stagekit ``hold_object``)
insert   chained: the end state of the scripted carry (``carry_to_slot``) from a carry start
release  chained: the end state of the scripted insert (``lower_into_slot``) from an insert start
Fallbacks (used only when the chain fails; recorded in the start's info): the plate set at the slot entry (insert) or
seated in the slot (release), weightless close, gravity hold.

Grasps: ``dishrack_geom.edge_grasp`` at rim angle theta in the plate plane (0 = top). Measured reach
(``pilot_reach2.py``): collision-free IK at the slot entry and seated poses mostly needs theta 30-75 deg off the top.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from abc_sim.task_eval.bottles_starts import idle_action
from abc_sim.task_eval.dishrack_geom import GRIP_HALF_MAX, TABLE_Z, edge_grasp, quat_of_R, upright_rotation
from abc_sim.task_eval.dishrack_stages import LABELS, P, STAGES, classify
from abc_sim.task_eval.stagekit import (ARMS, STATE_SPEC, T, actuator_limits, arm_ids, grasp_from_state, hold_action,
                                       ik_link6, set_object_pose, touches_other)

SEED_BASE = 1_300_000_000
THETAS_DEG = (-30, 30, -45, 45, -60, 60, -75, 75)
BELOW_RIM_M = 0.015
# Repair round 2 (G3 carry, 2026-11-14): pinch at the plate's top (theta 0) and tilt the approach by beta in the
# plate plane, instead of pinching the rim at theta off the top with a radial approach. The off-top pinch creeps
# (gravity torque about the jaw axis, 1-2 deg per chunk) and leaked |idle return| >= 0.05 on 16/88 starts.
GRASP_THETA = 0.0
# lowest point of a carry start above the table: 3-6 cm starts slipped onto the table within 10 idle chunks in 5 of
# 6 seeds (rim pinch creep, pilot_carry.py 2026-11-14)
START_LOW_M = (0.06, 0.10)   # round 3 tried 3-6 cm again: 29/40 built vs 35/40, so kept at 6-10 cm
PRESETTLE_ACTIONS = 0    # 64 (repair round 1) made G3 worse: 23/40 valid
BUDGET_S = 8.0   # CPU s for the carry-start grasp search (G3: <= 10 s per start)
# chain only if the carry start was found by then (carry ~2.5 s, insert ~1.5 s, fallback ~1-2 s CPU)
CHAIN_START_BY_S = dict(insert=2.5, release=2.5)
CHAIN_CARRY_CANDIDATES = 4   # round 3: a failing chained carry hands its time to the slot fallback sooner


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def robot_geoms(model):
    out = set()
    for b in range(model.nbody):
        if model.body(b).name.startswith(("left_", "right_")):
            out |= {g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                    and (model.geom_contype[g] or model.geom_conaffinity[g])}
    return out


def collision_free(env, scene, k, arm, q, T_plate, grip_half, *, robot=None, ignore_rack=False) -> bool:
    """Robot at q (``arm``) with jaws open ``grip_half`` and plate k at ``T_plate``: no robot geom penetrates
    the rack / table / another plate by > 1 mm (the target plate is allowed)."""
    model, data = env.model, env.data
    robot = robot if robot is not None else robot_geoms(model)
    d2 = mujoco.MjData(model)
    d2.qpos[:] = data.qpos
    ids = arm_ids(model, ARMS[arm])
    d2.qpos[ids.qadr] = q
    d2.qpos[ids.finger_qadr[0]], d2.qpos[ids.finger_qadr[1]] = grip_half, -grip_half
    a = int(scene.qpos_adr[k])
    d2.qpos[a:a + 3] = T_plate[:3, 3]
    d2.qpos[a + 3:a + 7] = quat_of_R(T_plate[:3, :3])
    mujoco.mj_forward(model, d2)
    own = scene._allgeoms_of(k)
    for con in d2.contact[:d2.ncon]:
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in robot) == (g2 in robot) or con.dist >= -0.001:
            continue
        other = g2 if g1 in robot else g1
        if other in own or (ignore_rack and other in scene.rack_geoms):
            continue
        return False
    return True


def solve_pose(env, arm, T6, *, data=None, seeds=6):
    model = env.model
    q, pe, re = ik_link6(model, data if data is not None else env.data, ARMS[arm], T6, seeds=seeds,
                         limits=actuator_limits(model, ARMS[arm]))
    if q is None or pe > 3e-3 or re > 0.05:
        return None
    return q


def plan_and_move(env, arm, targets, *, actions=(96,), grip=0.0, waypoints=8, tol=(0.005, 0.05)):
    """Cartesian legs through link_6 poses ``targets``; every waypoint of every leg is IK-solved (continuing from
    the previous solution) before anything moves. Returns (ok, why)."""
    model, data = env.model, env.data
    ids = arm_ids(model, ARMS[arm])
    planning = mujoco.MjData(model)
    planning.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, planning)
    limits = actuator_limits(model, ARMS[arm])
    legs = []
    for li, target in enumerate(targets):
        start = T(planning.xpos[ids.link6].copy(), planning.xquat[ids.link6].copy())
        qa, qb = np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(qa, start[:3, :3].ravel())
        mujoco.mju_mat2Quat(qb, target[:3, :3].ravel())
        if qa @ qb < 0:
            qb = -qb
        qs = []
        for f in np.linspace(0, 1, waypoints + 1)[1:]:
            quat = (1 - f) * qa + f * qb
            quat /= np.linalg.norm(quat)
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, quat)
            pose = np.eye(4)
            pose[:3, :3] = mat.reshape(3, 3)
            pose[:3, 3] = start[:3, 3] * (1 - f) + target[:3, 3] * f
            # continuity: one IK seed, from the previous waypoint (no random restarts = no joint-space jumps)
            q, pe, re = ik_link6(model, planning, ARMS[arm], pose, seeds=1, limits=limits)
            if q is None or pe > tol[0] or re > tol[1]:
                return False, f"IK unreachable on leg {li} (position {pe:.4f}, rotation {re:.3f})"
            qs.append(q)
            planning.qpos[ids.qadr] = q
            mujoco.mj_kinematics(model, planning)
        legs.append(qs)
    # execute: joint targets interpolated every action between consecutive waypoints (no target jumps)
    q_prev = idle_action(env)[arm * 7:arm * 7 + 6].astype(np.float64)
    for li, qs in enumerate(legs):
        n = actions[min(li, len(actions) - 1)]
        per = max(1, n // waypoints)
        for q in qs:
            for j in range(1, per + 1):
                act = idle_action(env)
                act[arm * 7:arm * 7 + 6] = q_prev + (q - q_prev) * (j / per)
                act[arm * 7 + 6] = grip
                env.step(act, render_obs=False)
            q_prev = np.asarray(q, np.float64)
    return True, ""


def settle(env, arm, steps, grip=0.0):
    act = idle_action(env)
    act[arm * 7 + 6] = grip
    for _ in range(steps):
        env.step(act, render_obs=False)


def yaw_aligned_pose(scene, data, k, slot, low, sign):
    """Plate k's current pose turned about world z so its flat normal points along ``sign`` x the slot axis, moved so
    its bbox centre is over the slot centre and its lowest point is ``low`` above the rack base. Keeping the wrist's
    tilt (only yaw + translation) keeps the in-hand grasp valid. -> (4x4 pose, |yaw change| rad)."""
    rp, Rr = data.xpos[scene.rack_body].copy(), data.xmat[scene.rack_body].reshape(3, 3).copy()
    R = data.xmat[scene.bodies[k]].reshape(3, 3).copy()
    n = R[:, 2].copy()
    n[2] = 0
    goal = sign * Rr[:, slot["axis"]].copy()
    goal[2] = 0
    dyaw = float(np.arctan2(n[0] * goal[1] - n[1] * goal[0], n @ goal))
    Rt = Rz(dyaw) @ R
    pc = scene.plate_pc[k]
    rel = (scene.plate_pts[k] - pc) @ Rt.T
    c = np.zeros(3)
    c[slot["axis"]], c[1 - slot["axis"]] = slot["t"], slot["u"]
    c[2] = low - float((rel @ Rr[:, 2]).min())
    Tp = np.eye(4)
    Tp[:3, :3] = Rt
    Tp[:3, 3] = rp + Rr @ c - Rt @ pc
    return Tp, abs(dyaw)


def _rack_top_under(scene, data, k, xy, margin=0.02) -> float:
    """Highest rack point (rack frame z) under a disc of the plate's half-width + ``margin`` at world ``xy``."""
    rp, Rr = data.xpos[scene.rack_body], data.xmat[scene.rack_body].reshape(3, 3)
    c = (np.r_[xy, rp[2]] - rp) @ Rr
    r = float(np.abs(scene.plate_pts[k] - scene.plate_pc[k]).max()) + margin
    near = scene.rack_pts[np.linalg.norm(scene.rack_pts[:, :2] - c[:2], axis=1) <= r]
    return float(near[:, 2].max()) if len(near) else -np.inf


def _leg_clear(scene, data, k, xy0, low0, xy1, low1, *, clearance=0.015, n=12) -> bool:
    return all(low0 + f * (low1 - low0) >= _rack_top_under(scene, data, k, xy0 + f * (xy1 - xy0)) + clearance
               for f in np.linspace(0, 1, n))


def _shift(Tp, scene, k, xy, low, data):
    """``Tp`` moved so the plate's bbox centre is at world ``xy`` and its lowest point ``low`` above the rack base."""
    Rr = data.xmat[scene.rack_body].reshape(3, 3)
    out = Tp.copy()
    centre = (Tp[:3, :3] @ scene.plate_pc[k] + Tp[:3, 3])
    rel = (scene.plate_pts[k] - scene.plate_pc[k]) @ Tp[:3, :3].T
    low_now = float((centre - data.xpos[scene.rack_body]) @ Rr[:, 2] + (rel @ Rr[:, 2]).min())
    out[:2, 3] += xy - centre[:2]
    out[:3, 3] += Rr[:, 2] * (low - low_now)
    return out


def other_plates_top(scene, data, k) -> float:
    """Highest collision point of the other plates, rack frame z (-inf if none)."""
    rp, Rr = data.xpos[scene.rack_body], data.xmat[scene.rack_body].reshape(3, 3)
    tops = [-np.inf]
    for j in range(len(scene.names)):
        if j != k:
            R = data.xmat[scene.bodies[j]].reshape(3, 3)
            pts = scene.plate_pts[j] @ R.T + data.xpos[scene.bodies[j]]
            tops.append(float(((pts - rp) @ Rr[:, 2]).max()))
    return max(tops)


def carry_paths(scene, data, k, Tp, entry_low, hover_extra_m=0.03):
    """Candidate plate-pose sequences from the current pose to ``Tp`` (the entry pose), cheapest first: a direct
    diagonal to the hover pose above the slot; a low approach to the footprint's edge, a rise there, the traverse;
    a vertical lift at the plate's spot, the traverse. Legs that would sweep the plate through the rack are dropped."""
    cur = scene.object_T(data, k)
    xy0 = scene.observe_centre(data, k)
    low0 = float(scene.observe_low(data, k))
    xy1 = (Tp[:3, :3] @ scene.plate_pc[k] + Tp[:3, 3])[:2]
    hover_low = max(_rack_top_under(scene, data, k, xy1) + hover_extra_m, entry_low, low0)
    peak = max(hover_low, max(_rack_top_under(scene, data, k, xy0 + f * (xy1 - xy0)) for f in np.linspace(0, 1, 12))
               + hover_extra_m)
    hover = _shift(Tp, scene, k, xy1, peak, data)
    paths = []
    if _leg_clear(scene, data, k, xy0, low0, xy1, peak):
        paths.append(("diagonal", [hover, Tp]))
    # low approach: at least 3 cm above the other plates' tops, up to where the plate's disc would meet the rack
    travel = max(low0, other_plates_top(scene, data, k) + 0.03)
    f_edge = 0.0
    for f in np.linspace(0, 1, 41):
        if _rack_top_under(scene, data, k, xy0 + f * (xy1 - xy0), 0.03) >= travel - 0.015:
            break
        f_edge = f
    if f_edge > 0.05:
        xye = xy0 + f_edge * (xy1 - xy0)
        start_up = _shift(cur, scene, k, xy0, travel, data)
        edge_low = _shift(Tp, scene, k, xye, travel, data)
        edge_high = _shift(Tp, scene, k, xye, peak, data)
        paths.append(("edge", ([start_up] if travel > low0 + 0.005 else []) + [edge_low, edge_high, hover, Tp]))
    lift = _shift(cur, scene, k, xy0, peak, data)
    paths.append(("lift", [lift, hover, Tp]))
    return paths


def _entry_candidates(scene, data, k, *, with_slot=False):
    """Entry poses (plate 1.5 cm above a slot's entry, turned about the vertical only), centre line first, then by
    yaw change in 20 deg bins, then spread along the rack (alternating ends) so a short list covers it."""
    lo, hi = scene.fp_lo, scene.fp_hi
    cands = []
    for slot in scene.slots:
        mid = (lo[1 - slot["axis"]] + hi[1 - slot["axis"]]) / 2
        for sign in (1, -1):
            Tp, dyaw = yaw_aligned_pose(scene, data, k, slot, slot["entry"] + 0.015, sign)
            if dyaw > np.radians(100):
                continue
            centre = abs(slot["u"] - mid) < 1e-6
            span = (lo[slot["axis"]], hi[slot["axis"]])
            spread = abs((slot["t"] - span[0]) / max(span[1] - span[0], 1e-6) - 0.5)
            cands.append((not centre, round(dyaw / 0.35), -spread, Tp, slot))
    cands.sort(key=lambda c: c[:3])
    return [(c[3], c[4]) if with_slot else c[3] for c in cands]


def carry_to_slot(env, scene, k, arm, *, max_candidates=10, rng=None):
    """Scripted carry: move the plate (turning about the vertical only, so the in-hand grasp stays valid) along one of
    ``carry_paths`` to 1.5 cm above a slot's entry. Candidates (slot, normal sign) are ordered by yaw change, centre
    line first; only collision-free, IK-feasible entry poses are tried; every leg is planned before anything moves.
    Returns (ok, why)."""
    model, data = env.model, env.data
    g = grasp_from_state(model, data, scene, k, arm)
    grasp, grip_half = g["T_obj_link6"], float(np.mean(np.abs(g["fingers"])))
    robot = robot_geoms(model)
    tried, notes = 0, []
    for Tp, slot in _entry_candidates(scene, data, k, with_slot=True):
        q = solve_pose(env, arm, Tp @ grasp, seeds=6)
        if q is None or not collision_free(env, scene, k, arm, q, Tp, grip_half, robot=robot):
            continue
        tried += 1
        for name, poses in carry_paths(scene, data, k, Tp, slot["entry"] + 0.015):
            n = len(poses)
            ok, why = plan_and_move(env, arm, [p @ grasp for p in poses], actions=(128,) * (n - 1) + (96,),
                                    waypoints=12)
            if ok:
                settle(env, arm, 32)
                if not scene.contact_obs(data)["grasped"][k]:
                    return False, f"plate slipped during carry ({name})"
                return True, name
            notes.append(f"{name}: {why}")
        if tried >= max_candidates:
            break
    return False, f"no plannable carry ({tried} candidates)" + (f": {notes[-1]}" if notes else "")


def best_slot(scene, obs, k):
    terms = scene._slot_terms(obs, k)
    return scene.slots[int(np.argmin(np.hypot(terms[:, 0], terms[:, 1]) + 0.5 * terms[:, 2]))]


def lower_into_slot(env, scene, k, arm, *, press_m=0.004):
    """Scripted insert: turn/shift onto the nearest slot's centre at 1 cm above its entry (or where the plate is, if
    lower), then lower straight down to ``press_m`` below the measured seat and hold 2 chunks. Returns (ok, why)."""
    model, data = env.model, env.data
    obs = scene.observe(data, env._task_evaluator.tracker.rest_height)
    slot = best_slot(scene, obs, k)
    n = data.xmat[scene.bodies[k]].reshape(3, 3)[:, 2]
    sign = 1 if n @ data.xmat[scene.rack_body].reshape(3, 3)[:, slot["axis"]] >= 0 else -1
    grasp = grasp_from_state(model, data, scene, k, arm)["T_obj_link6"]
    above, _ = yaw_aligned_pose(scene, data, k, slot, min(float(obs["low"][k]), slot["entry"] + 0.01), sign)
    seat, _ = yaw_aligned_pose(scene, data, k, slot, slot["seat"] - press_m, sign)
    ok, why = plan_and_move(env, arm, [above @ grasp, seat @ grasp], actions=(64, 192), waypoints=12)
    if not ok:
        return False, why
    settle(env, arm, 64)
    return True, ""


def release_and_retreat(env, arm, *, back_m=0.10, chunks_after=3):
    """Scripted release: open in place and wait 2 chunks, back the open jaws off 1.5 cm slowly along the approach, then
    the rest of ``back_m`` (else straight up), then wait. Repair round 3 (G4 release): the one-step 10 cm retreat after a
    1-chunk open dragged the plate out of its slot in 7/16 planned runs."""
    model, data = env.model, env.data
    ids = arm_ids(model, ARMS[arm])
    settle(env, arm, 64, grip=1.0)
    T6 = T(data.xpos[ids.link6].copy(), data.xquat[ids.link6].copy())
    for direction in (-T6[:3, 2], np.array([0.0, 0.0, 1.0])):
        first, rest = T6.copy(), T6.copy()
        first[:3, 3] += direction * 0.015
        rest[:3, 3] += direction * back_m
        ok, why = plan_and_move(env, arm, [first, rest], actions=(96, 96), grip=1.0)
        if ok:
            break
    settle(env, arm, 32 * chunks_after, grip=1.0)
    return ok, why


def _grip_half(info):
    return float(min(GRIP_HALF_MAX, info["thickness"] / 2 + 0.004))


def hold_plate(env, scene, k, T_plate, grasp, *, allow_rack=False, close_steps=10, hold_steps=30,
               max_drift_m=0.03, max_speed=0.05):
    """Set plate k at ``T_plate``, put grasp['arm'] on it (jaws 1 cm wider), close on the weightless plate, hold
    under gravity. ``allow_rack``: the plate may rest on the rack (seated fallback). -> (ok, info)."""
    model, data = env.model, env.data
    arm = int(grasp["arm"])
    set_object_pose(env, scene, k, T_plate[:3, 3], quat_of_R(T_plate[:3, :3]))
    q = solve_pose(env, arm, T_plate @ grasp["T_obj_link6"])
    if q is None:
        return False, {"why": "no IK for the grasp"}
    ids = arm_ids(model, ARMS[arm])
    data.qpos[ids.qadr], data.qvel[ids.dadr] = q, 0.0
    f = min(GRIP_HALF_MAX, grasp["fingers"][0] + 0.01)
    data.qpos[ids.finger_qadr[0]], data.qpos[ids.finger_qadr[1]] = f, -f
    data.qvel[list(ids.finger_dadr)] = 0.0
    mujoco.mj_forward(model, data)
    if not collision_free(env, scene, k, arm, q, T_plate, f, ignore_rack=False):
        return False, {"why": "gripper pose collides with the rack, table or another plate"}
    if not allow_rack and touches_other(model, data, scene, k, arm):
        return False, {"why": "plate pose collides with something"}
    act = hold_action(env, arm, q, 0.0)
    a, dof = int(scene.qpos_adr[k]), int(scene.dof_adr[k])
    obj_q = data.qpos[a:a + 7].copy()
    sub = [b for b in range(model.nbody) if b == scene.bodies[k] or model.body_parentid[b] == scene.bodies[k]]
    gc = model.body_gravcomp[sub].copy()
    model.body_gravcomp[sub] = 1.0
    try:
        for _ in range(close_steps):
            env.step(act, render_obs=False)
    finally:
        model.body_gravcomp[sub] = gc
    for _ in range(hold_steps):
        env.step(act, render_obs=False)
    obs = scene.contact_obs(data)
    drift = float(np.linalg.norm(data.qpos[a:a + 3] - obj_q[:3]))
    speed = float(np.linalg.norm(data.qvel[dof:dof + 3]))
    info = dict(q=q, drift=round(drift, 4), speed=round(speed, 4), held=bool(obs["grasp_arm"][k] in (arm, 2)))
    if not info["held"]:
        return False, {**info, "why": "plate not held by both jaws"}
    if drift > max_drift_m or speed > max_speed:
        return False, {**info, "why": f"plate slipped (drift {drift:.3f} m, speed {speed:.3f} m/s)"}
    if not allow_rack and touches_other(model, data, scene, k, arm):
        return False, {**info, "why": "held plate touches something else"}
    return True, info


def reaches_a_slot(env, scene, k, grasp, T_plate, *, robot=None, limit=8) -> bool:
    """The grasp, with the plate turned about the vertical only, reaches the entry pose of at least one slot
    (IK-feasible and collision-free): a reachability condition on the start, not a plan."""
    data = env.data
    arm = int(grasp["arm"])
    saved = data.qpos.copy()
    a = int(scene.qpos_adr[k])
    data.qpos[a:a + 3] = T_plate[:3, 3]
    data.qpos[a + 3:a + 7] = quat_of_R(T_plate[:3, :3])
    mujoco.mj_kinematics(env.model, data)
    try:
        for Tp in _entry_candidates(scene, data, k)[:limit]:
            q = solve_pose(env, arm, Tp @ grasp["T_obj_link6"], seeds=2)
            if q is not None and collision_free(env, scene, k, arm, q, Tp, grasp["fingers"][0], robot=robot):
                return True
        return False
    finally:
        data.qpos[:] = saved
        mujoco.mj_forward(env.model, data)


def _start_spots(scene, data, k):
    """Candidate xy for the carry start: the plate's own spot, then 1/3 and 2/3 of the way to the rack centre, kept
    only while the plate's disc stays 3 cm outside the rack footprint."""
    x0 = scene.object_T(data, k)
    own = (x0[:3, :3] @ scene.plate_pc[k] + x0[:3, 3])[:2]
    rack = data.xpos[scene.rack_body][:2]
    spots = [own]
    for f in (1 / 3, 2 / 3):
        xy = own + f * (rack - own)
        if _rack_top_under(scene, data, k, xy, 0.03) == -np.inf:
            spots.append(xy)
    return spots


def carry_start(env, scene, k, rng, *, below_rim_m=BELOW_RIM_M, deadline=None):
    """Source 3: upright plate, random yaw, lowest point 6-10 cm above the table, rim pinch, at the first of
    ``_start_spots`` where a grasp has IK. A grasp is used only if it can reach some slot's entry pose
    (``reaches_a_slot``); rim angles nearest the top are tried first (less in-hand creep: torque ~ sin theta)."""
    model, data = env.model, env.data
    yaw = rng.uniform(-np.pi, np.pi)
    R = Rz(yaw) @ upright_rotation(np.eye(3), 0)   # local x up: the top-of-rim pinch hangs in equilibrium
    pc = scene.plate_pc[k]
    rel = (scene.plate_pts[k] - pc) @ R.T
    low = TABLE_Z + rng.uniform(START_LOW_M[0], START_LOW_M[1])
    base = {0: np.array([0.252, 0.31]), 1: np.array([0.252, -0.31])}
    tries = []
    robot = robot_geoms(model)
    reach_cache = {}
    deadline = deadline if deadline is not None else time.process_time() + BUDGET_S
    for spot_i, xy in enumerate(_start_spots(scene, data, k)):
        Tp = np.eye(4)
        Tp[:3, :3] = R
        Tp[:3, 3] = np.r_[xy, low - rel[:, 2].min()] - R @ pc
        for arm in sorted((0, 1), key=lambda a: np.linalg.norm(xy - base[a])):
            for th in np.radians(THETAS_DEG):
                for side in (1, -1):
                    T6l, _, info = edge_grasp(scene.plate_pts[k], GRASP_THETA, beta=th, side=side,
                                              below_rim_m=below_rim_m)
                    if not info["fits"]:
                        return None, {"why": f"plate too thick for the jaws ({info['thickness']:.3f} m)"}
                    grasp = dict(arm=arm, T_obj_link6=T6l, fingers=[_grip_half(info)] * 2)
                    if time.process_time() > deadline:
                        return None, {"why": "over the start time budget"}
                    if solve_pose(env, arm, Tp @ T6l, seeds=4) is None:
                        tries.append("no IK at the start pose")
                        continue
                    key = (arm, round(float(th), 3), side)   # reach depends on the grasp, not on the start spot
                    if key not in reach_cache:
                        reach_cache[key] = reaches_a_slot(env, scene, k, grasp, Tp, robot=robot)
                    if not reach_cache[key]:
                        tries.append("grasp reaches no slot entry")
                        continue
                    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
                    mujoco.mj_getState(model, data, state, STATE_SPEC)
                    ok, det = hold_plate(env, scene, k, Tp, grasp)
                    if ok:
                        # pre-settle 2 chunks: the in-hand creep is fastest right after the close (G3 idle leak)
                        if PRESETTLE_ACTIONS:
                            settle(env, arm, PRESETTLE_ACTIONS)
                        if PRESETTLE_ACTIONS and not scene.contact_obs(data)["grasped"][k] or touches_other(model, data, scene, k, arm):
                            ok, det = False, {**det, "why": "plate slipped or touched something while settling"}
                    if ok:
                        return grasp, {**{a: b for a, b in det.items() if a != "q"}, "arm": arm, "spot": spot_i,
                                       "beta_deg": round(float(np.degrees(th))), "side": side}
                    if time.process_time() > deadline:
                        return None, {"why": "over the start time budget"}
                    tries.append(det.get("why"))
                    mujoco.mj_setState(model, data, state, STATE_SPEC)
                    mujoco.mj_forward(model, data)
    return None, {"why": f"no holding grasp ({len(tries)} tried; last: {tries[-1] if tries else '-'})"}


def slot_pose(scene, data, k, slot, low, flip=False):
    """Upright plate (local x up) in ``slot``'s plane, bbox centre over the slot, lowest point ``low`` (rack frame)."""
    rp, Rr = data.xpos[scene.rack_body].copy(), data.xmat[scene.rack_body].reshape(3, 3).copy()
    R = upright_rotation(Rr, slot["axis"])
    if flip:
        R = Rz(np.pi) @ R
    pc = scene.plate_pc[k]
    rel = (scene.plate_pts[k] - pc) @ R.T
    c = np.zeros(3)
    c[slot["axis"]], c[1 - slot["axis"]] = slot["t"], slot["u"]
    c[2] = low - float((rel @ Rr[:, 2]).min())
    Tp = np.eye(4)
    Tp[:3, :3] = R
    Tp[:3, 3] = rp + Rr @ c - R @ pc
    return Tp


def slot_fallback(env, scene, k, stage, rng, *, max_holds=6, max_ik=60, deadline=None):
    """Source 3 at the rack: the plate set upright at a slot entry (insert: lowest point 1.5 cm above the entry) or
    seated in it (release: lowest point 2 mm above the measured seat), top-of-rim pinch with a tilted approach,
    weightless close, gravity hold. The (slot, flip, arm, tilt, side) combinations are shuffled (nearer arm first)
    so the IK budget covers the whole rack (repair round 1: the first version only ever tried the first slot).
    -> (grasp | None, info)."""
    data = env.data
    robot = robot_geoms(env.model)
    base = {0: np.array([0.252, 0.31]), 1: np.array([0.252, -0.31])}
    combos = []
    for si, slot in enumerate(scene.slots):
        for flip in (False, True):
            Tp = slot_pose(scene, data, k, slot,
                           slot["entry"] + 0.015 if stage == "insert" else slot["seat_mid"] + 0.002, flip)
            near_arm = min((0, 1), key=lambda a: np.linalg.norm(Tp[:2, 3] - base[a]))
            for arm in (0, 1):
                for th in THETAS_DEG:
                    for side in (1, -1):
                        combos.append((arm != near_arm, rng.uniform(), Tp, arm, np.radians(th), side))
    combos.sort(key=lambda c: c[:2])
    holds = 0
    last = "no slot pose with a reachable, collision-free grasp"
    for i, (_, _, Tp, arm, th, side) in enumerate(combos[:max_ik]):
        if deadline is not None and time.process_time() > deadline:
            return None, {"why": "over the start time budget"}
        T6l, _, info = edge_grasp(scene.plate_pts[k], GRASP_THETA, beta=th, side=side, below_rim_m=BELOW_RIM_M)
        if not info["fits"]:
            return None, {"why": f"plate too thick for the jaws ({info['thickness']:.3f} m)"}
        q = solve_pose(env, arm, Tp @ T6l, seeds=2)
        if q is None or not collision_free(env, scene, k, arm, q, Tp, _grip_half(info), robot=robot):
            continue
        grasp = dict(arm=arm, T_obj_link6=T6l, fingers=[_grip_half(info)] * 2)
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
        mujoco.mj_getState(env.model, data, state, STATE_SPEC)
        ok, det = hold_plate(env, scene, k, Tp, grasp, allow_rack=stage == "release")
        holds += 1
        if ok:
            return grasp, {**{a: b for a, b in det.items() if a != "q"}, "arm": arm,
                           "beta_deg": round(float(np.degrees(th))), "side": side}
        last = det.get("why", last)
        mujoco.mj_setState(env.model, data, state, STATE_SPEC)
        mujoco.mj_forward(env.model, data)
        if holds >= max_holds:
            break
    return None, {"why": last}


def holds_when_opened(env, scene, k, arm, rest, *, actions=32) -> bool:
    """Release-start check (repair round 2, G4 release): on a scratch copy of the state, open ``arm``'s jaws in place
    for one chunk; True if the plate is still in the stock in-rack geometry (upright, over the footprint, touching
    the rack). The kinematic slot map does not test that a slot holds a released plate (``dr_slots.py`` did)."""
    model = env.model
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, env.data, state, STATE_SPEC)
    d3 = mujoco.MjData(model)
    mujoco.mj_setState(model, d3, state, STATE_SPEC)
    mujoco.mj_forward(model, d3)
    d3.ctrl[env._ctrl_indices[arm * 7 + 6]] = GRIP_HALF_MAX
    for _ in range(actions * env._control_decimation):
        mujoco.mj_step(model, d3)
    obs = scene.observe(d3, rest)
    return bool(obs["in_rack_geom"][k])


def _fresh(env, seed):
    env.forget_arm_state()
    env.reset(seed=seed, randomize=True)
    env.evaluate_task()
    return env._task_evaluator


def build_start(env, seed, stage, *, rng=None, tries=None):
    """Start of ``stage`` for ``seed``: carry = source 3 at the plate's spot; insert / release = chained through the
    scripted carry (and insert), else the slot fallback (source 3 at the rack). -> (start | None, log)."""
    rng = rng or np.random.default_rng(seed)
    logs = []
    want = STAGES.index(stage)
    t_begin = time.process_time()
    tries = tries or (4 if stage == "release" else 2)   # release: retry other slots (holds_when_opened)
    for attempt in range(tries):
        if time.process_time() - t_begin > BUDGET_S:
            logs.append(dict(attempt=attempt, why="over the start time budget"))
            break
        ev = _fresh(env, seed)
        scene = ev.scene
        rest = scene.heights(env.data)
        obs = scene.observe(env.data, rest)
        k = ev.pick_target(obs)
        scene.set_target(env.data, k)
        info = dict(attempt=attempt, target=k, slots=len(scene.slots))
        if not scene.slots:
            logs.append({**info, "why": "no slot fits this plate in this rack"})
            break
        # insert / release: the carry start is only the chain's first step; cap its search so the slot fallback keeps
        # its share of the 10 s budget (repair round 2)
        grasp, det = carry_start(env, scene, k, rng, deadline=t_begin + (BUDGET_S if stage == "carry"
                                                                          else CHAIN_START_BY_S[stage]))
        info.update(det)
        chain_why = None
        if grasp is None:
            chain_why = f"carry start: {det.get('why')}"
        elif stage in ("insert", "release") and time.process_time() - t_begin > CHAIN_START_BY_S[stage]:
            chain_why = "chain skipped: it would not finish within the start time budget"
        elif stage in ("insert", "release"):
            arm = int(grasp["arm"])
            ok, why = carry_to_slot(env, scene, k, arm, rng=rng, max_candidates=CHAIN_CARRY_CANDIDATES)
            info["source"] = "chained from the scripted carry"
            if not ok:
                chain_why = f"carry chain: {why}"
            elif stage == "release":
                ev.begin(env.model, env.data, k, rest_height=rest)
                ok, why = lower_into_slot(env, scene, k, arm)
                info["source"] = "chained from the scripted carry + insert"
                if not ok:
                    chain_why = f"insert chain: {why}"
            if chain_why is None:
                o = scene.observe(env.data, rest)
                labels = np.full(len(scene.names), P)
                for _ in range(3):
                    labels = classify(o, labels, ev.cfg)
                if labels[k] != want:
                    chain_why = f"chain ended as {LABELS[labels[k]]}"
        if chain_why is not None:
            if stage == "carry":
                logs.append({**info, "why": chain_why})
                continue
            ev = _fresh(env, seed)
            scene = ev.scene
            scene.set_target(env.data, k)
            grasp, det = slot_fallback(env, scene, k, stage, rng, deadline=t_begin + BUDGET_S + 0.8)
            info = {**dict(attempt=attempt, target=k, slots=len(scene.slots)), **det,
                    "source": "slot fallback (source 3 at the rack)", "chain_failure": chain_why}
            if grasp is None:
                logs.append(info)
                continue
            if stage == "release":   # held just above the seat: press it down onto the rack (the insert motion)
                ev.begin(env.model, env.data, k, rest_height=rest)
                ok, why = lower_into_slot(env, scene, k, int(grasp["arm"]))
                if not ok:
                    logs.append({**info, "why": f"fallback settle: {why}"})
                    continue
        arm = int(grasp["arm"])
        mujoco.mj_forward(env.model, env.data)
        if stage == "release" and not holds_when_opened(env, scene, k, arm, rest):
            logs.append({**info, "why": "the slot does not hold the plate when the jaws open"})
            continue
        obs = scene.observe(env.data, rest)
        labels = np.full(len(scene.names), P)
        for _ in range(3):
            labels = classify(obs, labels, ev.cfg)
        if labels[k] != want:
            logs.append({**info, "why": f"classified as {LABELS[labels[k]]}"})
            continue
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
        mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
        return dict(seed=int(seed), stage=stage, target=int(k), arm=arm, state=state, rest_height=rest,
                    labels=labels, info=info), logs + [{**info, "ok": True}]
    return None, logs
