"""LEGO cross-side handoff: grasp plans, the scripted handoff solver and the cross-side carry/drop start setup
(stage table amendment 3; stage tests are ``handoff_stage`` via ``LegoScene.observe``).

Geometry (measured, STATUS.md "Handoff"): blocks are 4.1-5.5 x 2.1-2.8 x 1.2-1.7 cm boxes, finger pads 3 cm wide.
Two top-down grasps side by side do not fit on one block, so the giver holds it top-down across its *width* at the
middle (pads cover the middle 3 cm of the long faces) and the receiver closes along its *length* on the two free end
faces, approaching from its own base side (horizontal or tilted). Pinch centres are this scene's measured closed-pad
midpoints ``LegoScene.pad_link6`` (link_6 (0, -0.044, 0.130)), not ``<arm>_grasp_site``.

Motion helpers (arm-only, task-independent) come from ``inhand_starts``; the object-specific ones are re-written here
for block k (inhand's assume object 0).
"""
from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.inhand_starts import (action_with, approach_pose, current_action, link6_T, move_joints,
                                             move_link, restore, retreat, run, snapshot, teleport_arm)
from abc_sim.task_eval.stagekit import (ARMS, GRIP_MAX, T, actuator_limits, arm_ids, axis_angle_quat, hold_object,
                                        ik_link6, inv, qmul, quat_of, set_object_pose)

BASES = (np.array([.252, .31, 0.]), np.array([.252, -.31, 0.]))
# Handoff points (block CoM) inside the LEGO region HANDOFF (x .40-.60, |y| <= .10, z .80-1.10) and the block's
# long-axis yaw there (0 = along x, so the receiver approaches across the table from its own side).
MIDS = ((.50, 0., .90), (.47, 0., .87), (.53, 0., .87), (.50, 0., .95), (.45, 0., .92))
AXIS_YAWS = np.deg2rad([0, 20, -20, 40, -40])
GIVER_TILTS = np.deg2rad([0, 20, -20, 35, -35, 50, -50])
RECV_ELEV = np.deg2rad([25, 35, 45, 15, 55, 0, -15, -25])  # receiver approach elevation (+ = from above); G5 round 1 added 15/35/55/-15
IK_TOL = (2.5e-3, .04)


def block_frame(data, scene, k):
    """(centre, long axis u, width axis w, up axis, half sizes sorted long..short) of block k's collision box."""
    g = scene.block_geoms[k]
    Rg = data.geom_xmat[g].reshape(3, 3)
    size = scene.model.geom_size[g]
    order = np.argsort(-size)
    u, w, h = (Rg[:, i].copy() for i in order)
    if h[2] < 0:
        h = -h
    return data.geom_xpos[g].copy(), u, w, h, size[order]


def _pose(R, centre, pad):
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = centre - R @ pad
    return out


def _rot(axis, angle):
    return T(np.zeros(3), axis_angle_quat(np.asarray(axis, float), float(angle)))[:3, :3]


def giver_T6s(data, scene, k, arm):
    """Top-down grasps across the block's width at its middle, pads on the centre, tilted about the closing axis."""
    c, u, w, h, _ = block_frame(data, scene, k)
    out = []
    for tilt in GIVER_TILTS:
        for flip in (1., -1.):
            z = -h
            x = flip * w
            R = np.column_stack([x, np.cross(z, x), z])
            out.append((f'giver_tilt{np.rad2deg(tilt):+.0f}_{int(flip)}', _pose(_rot(x, tilt) @ R, c, scene.pad_link6[arm])))
    return out


def receiver_T6s(data, scene, k, arm):
    """Grasps closing along the block's length (pads on the end faces), approaching from ``arm``'s base side."""
    c, u, w, h, _ = block_frame(data, scene, k)
    n = np.cross(h, u)
    n /= np.linalg.norm(n)
    if n @ (c - BASES[arm]) < 0:
        n = -n
    out = []
    for elev in RECV_ELEV:
        z = np.cos(elev) * n - np.sin(elev) * h
        for flip in (1., -1.):
            x = flip * u
            x = x - (x @ z) * z
            x /= np.linalg.norm(x)
            R = np.column_stack([x, np.cross(z, x), z])
            out.append((f'recv_elev{np.rad2deg(elev):+.0f}_{int(flip)}', _pose(R, c, scene.pad_link6[arm])))
    return out


def arm_contacts(data, scene, arm):
    """Names of bodies (not of ``arm`` itself) touching ``arm``'s subtree."""
    m = scene.model
    out = []
    for con in data.contact[:data.ncon]:
        a1, a2 = scene.arm_of[con.geom1], scene.arm_of[con.geom2]
        for x, y, gy in ((a1, a2, con.geom2), (a2, a1, con.geom1)):
            if x == arm and y != arm:
                out.append(m.body(int(m.geom_bodyid[gy])).name or 'world')
    return out


def solve(env, arm, T6):
    q, pe, re = ik_link6(env.model, env.data, ARMS[arm], T6, limits=actuator_limits(env.model, ARMS[arm]), seeds=3)
    return q if q is not None and pe < IK_TOL[0] and re < IK_TOL[1] else None


def open_receiver_fits(env, scene, k, arm, limit=2):
    """Up to ``limit`` (name, T6, q) receiver grasps whose open gripper, teleported there, touches nothing."""
    out = []
    for name, T6 in receiver_T6s(env.data, scene, k, arm):
        q = solve(env, arm, T6)
        if q is None:
            continue
        snap = snapshot(env)
        teleport_arm(env, arm, q)
        hit = arm_contacts(env.data, scene, arm)
        restore(env, snap)
        if not hit:
            out.append((name, T6, q))
            if len(out) >= limit:
                break
    return out


def _yawed(R, dyaw):
    return _rot([0., 0., 1.], dyaw) @ R


def plan_region_pose(env, scene, k, giver):
    """Where the giver should bring the held block: the first (mid, long-axis yaw) whose giver pose is reachable from
    the current grasp and for which the receiver has a collision-free open grasp there (checked by teleporting block
    and giver, then restored). -> dict(giver_T6, mid, yaw) or None."""
    data = env.data
    B = 1 - giver
    T_blk = scene.object_T(data, k)
    T_l6 = link6_T(env, giver)
    rel = inv(T_blk) @ T_l6
    com = data.subtree_com[scene.bodies[k]].copy()
    off = T_blk[:3, 3] - com
    _, u, _, _, _ = block_frame(data, scene, k)
    yaw_now = np.arctan2(u[1], u[0])
    ids = arm_ids(env.model, ARMS[giver])
    a = int(scene.qpos_adr[k])
    for mid in MIDS:
        for yaw in AXIS_YAWS:
            d = (yaw - yaw_now + np.pi / 2) % np.pi - np.pi / 2  # the long axis is a line: turn by at most 90 deg
            Rz = _rot([0., 0., 1.], d)
            target = np.eye(4)
            target[:3, :3] = Rz @ T_blk[:3, :3]
            target[:3, 3] = np.asarray(mid) + Rz @ off
            T6 = target @ rel
            q = solve(env, giver, T6)
            if q is None:
                continue
            snap = snapshot(env)
            set_object_pose(env, scene, k, target[:3, 3], quat_of(target[:3, :3]))
            data.qpos[ids.qadr] = q
            mujoco.mj_forward(env.model, data)
            fits = open_receiver_fits(env, scene, k, B, limit=1)
            restore(env, snap)
            if fits:
                return dict(giver_T6=T6, mid=mid, yaw=float(np.rad2deg(yaw)), turn=float(np.rad2deg(d)))
    return None


def monitored_release(env, scene, k, giver, sag_m=.02, speed_mps=.03):
    """The giver opens in steps while the receiver holds; if the block sags > ``sag_m`` or the receiver loses a
    finger, or it still moves at the end, the giver closes again (False)."""
    B = 1 - giver
    b, d = scene.bodies[k], int(scene.dof_adr[k])
    z0 = float(env.data.subtree_com[b][2])
    for grip, n in ((.35, 4), (.7, 4), (1., 4), (1., 4), (1., 6), (1., 6)):
        run(env, action_with(env, None, {giver: grip}), n)
        f = scene.contact_obs(env.data)['hits'][k].reshape(2, 2).sum(1)
        if z0 - float(env.data.subtree_com[b][2]) > sag_m or f[B] < 2:
            run(env, action_with(env, None, {giver: 0.}), 16)
            return False
    if np.linalg.norm(env.data.qvel[d:d + 3]) > speed_mps:
        run(env, action_with(env, None, {giver: 0.}), 16)
        return False
    return True


def bring_to_region(env, scene, k, giver):
    """Giver carries its held block up and into the handoff region (pose from ``plan_region_pose``)."""
    plan = plan_region_pose(env, scene, k, giver)
    if plan is None:
        return False, 'no region pose with a free receiver grasp', None
    up = link6_T(env, giver)
    up[2, 3] = max(up[2, 3], plan['giver_T6'][2, 3])
    for target in (up, plan['giver_T6']):
        ok, why = move_link(env, giver, target, actions=96, grip=0., tol=(.01, .3))
        if not ok:
            return False, why, plan
    run(env, current_action(env), 16)
    if scene.contact_obs(env.data)['grasp_arm'][k] not in (giver, 2):
        return False, 'block slipped on the way to the region', plan
    return True, '', plan


def receive(env, scene, k, giver, *, tries=3, teleport=False):
    """Receiver approaches (joint space to a pre-grasp, then straight in), closes; once both its fingers are on the
    block the giver opens (monitored), lifts and backs off. ``teleport`` (start-building fallback only) puts the
    receiver at the grasp instead of approaching."""
    B = 1 - giver
    whys = []
    for name, T6, q in open_receiver_fits(env, scene, k, B, limit=tries):
        if teleport:
            teleport_arm(env, B, q)
            run(env, action_with(env, {B: q}, {B: 1.}), 4)
            qpre = None
        else:
            qpre = approach_pose(env, B, T6, q)
            if qpre is None:
                whys.append(f'{name}: pre-grasp unreachable')
                continue
            move_joints(env, B, qpre, actions=96, grip=1.)
            ok, why = move_link(env, B, T6, actions=48, grip=1., n_way=4)
            if not ok:
                whys.append(f'{name}: {why}')
                move_joints(env, B, qpre, actions=32, grip=1.)
                continue
        run(env, action_with(env, None, {B: 0.}), 24)
        f = scene.contact_obs(env.data)['hits'][k].reshape(2, 2).sum(1)
        if f[B] == 2 and f[giver] == 2 and monitored_release(env, scene, k, giver):
            retreat(env, giver)
            run(env, current_action(env), 16)
            return True, name
        whys.append(f'{name}: receiver closed with {f[B]} fingers' if f[B] < 2 else f'{name}: receiver cannot hold it alone')
        run(env, action_with(env, None, {B: 1.}), 12)
        if qpre is not None:
            move_joints(env, B, qpre, actions=32, grip=1.)
    return False, '; '.join(whys) or 'no free receiver grasp'


def handoff_solver(env, scene, k, giver):
    ok, why, plan = bring_to_region(env, scene, k, giver)
    if not ok:
        return False, why
    return receive(env, scene, k, giver)


def receiver_hold_in_region(env, scene, k, receiver, rng):
    """Grasp source 3: block moved weightless into the region, the receiver holds it alone (end-face grasp);
    the giver stays where it is (home after a reset)."""
    a = int(scene.qpos_adr[k])
    com = env.data.subtree_com[scene.bodies[k]].copy()
    whys = []
    base = snapshot(env)
    for mid in MIDS:
        for yaw in AXIS_YAWS:
            restore(env, base)
            T_blk = scene.object_T(env.data, k)
            _, u, _, _, _ = block_frame(env.data, scene, k)
            d = (yaw - np.arctan2(u[1], u[0]) + np.pi / 2) % np.pi - np.pi / 2
            Rz = _rot([0., 0., 1.], d)
            set_object_pose(env, scene, k, np.asarray(mid) + Rz @ (T_blk[:3, 3] - com),
                            qmul(quat_of(Rz), env.data.qpos[a + 3:a + 7].copy()))
            for name, T6, q in open_receiver_fits(env, scene, k, receiver, limit=2):
                teleport_arm(env, receiver, q)
                grasp = dict(arm=int(receiver), T_obj_link6=inv(scene.object_T(env.data, k)) @ T6,
                             fingers=np.array([GRIP_MAX - .01, GRIP_MAX - .01]))
                ok, info = hold_object(env, scene, k, grasp, rng=rng)
                if ok:
                    return True, dict(grasp=name, mid=list(mid), yaw=float(np.rad2deg(yaw)), drift=info.get('drift'))
                whys.append(f'{name}: {info.get("why")}')
    return False, dict(why='; '.join(whys[-3:]) or 'no receiver grasp in the region')


def creep_check(env, scene, k, arm, limit_m):
    """64 idle actions: still held by ``arm`` and sunk at most ``limit_m`` (the LEGO build-time slip check)."""
    z0 = float(env.data.subtree_com[scene.bodies[k], 2])
    run(env, current_action(env), 64)
    creep = z0 - float(env.data.subtree_com[scene.bodies[k], 2])
    return bool(scene.contact_obs(env.data)['grasp_arm'][k] == arm and creep <= limit_m), creep


def build_cross_start(env, seed, stage, *, rng, tries, began):
    """Carry / drop start of a cross-side block: the receiver holds it alone after a handoff.

    Source 1 (chain): the handoff start (giver's sampled lifted hold) + the scripted handoff solver. Source 3
    (fallback): the block moved weightless into the region and the receiver holds it alone. Drop then chains the
    carry solver (``carry_to_opening`` with the receiver, tilts DROP_TILTS + 30 deg) from that held snapshot."""
    from abc_sim.task_eval import handoff_stage as hs
    from abc_sim.task_eval.lego_starts import (BUDGET_S, CREEP_M, DROP_CREEP_M, DROP_TILTS, _clock, block_yaw,
                                               carry_to_opening, native_grasp)
    from abc_sim.task_eval.lego_stages import R, STAGES, StageConfig, classify
    from abc_sim.task_eval.stagekit import STATE_SPEC
    cfg, logs = StageConfig(), []
    limit = DROP_CREEP_M if stage == 'drop' else CREEP_M
    for attempt in range(tries):
        if attempt and _clock() - began > BUDGET_S:
            break
        env.forget_arm_state(); env.reset(seed=seed, randomize=True); env.evaluate_task()
        scene = env._task_evaluator.scene
        rest = scene.heights(env.data)
        obs = scene.observe(env.data, rest)
        k = env._task_evaluator.pick_target(obs)
        giver = int(obs['giver'][k]); B = 1 - giver
        info = dict(attempt=attempt, target=int(k), target_name=scene.names[k], arm=B, giver=giver, receiver=B,
                    cross_side=True, start_side='left' if obs['xy'][k, 1] > 0 else 'right',
                    bin_side='left' if B == 0 else 'right')
        ok = False
        # Carry: source 1 on even attempts, source 3 on odd ones. Drop: the other way round -- drop chains the carry
        # solver from either, and source 3 leaves the CPU budget for it.
        if attempt % 2 == (0 if stage == 'carry' else 1):
            a = scene.qpos_adr[k]
            pos = env.data.qpos[a:a+3].copy(); pos[2] = rest[k, 2] + rng.uniform(.06, .12)
            set_object_pose(env, scene, k, pos, env.data.qpos[a+3:a+7].copy())
            yaw = block_yaw(env.model, env.data, scene, k) + (0., np.pi)[(attempt // 2) % 2]
            ok, d = hold_object(env, scene, k, native_grasp(env.data, scene, k, yaw, giver), rng=rng)
            info['source'] = 'chained handoff solver'
            why = d.get('why', '')
            if ok:
                ok, why = handoff_solver(env, scene, k, giver)
            if ok:
                o = scene.observe(env.data, rest)
                ok = bool(hs.received_in_air(o, cfg.handoff)[k])
                why = '' if ok else 'handoff solver ended without the receiver holding it alone in the air'
        else:
            ok, d = receiver_hold_in_region(env, scene, k, B, rng)
            info.update(source='sampled receiver hold in the region (grasp source 3)', **{x: y for x, y in d.items() if x != 'why'})
            why = d.get('why', '')
        if ok and _clock() - began > BUDGET_S + 4:
            ok, why = False, 'CPU budget'
        if ok:
            # Drop uses its measured 1.5 mm limit here too (G3 drop round 3; round 4 rejected 6/52 cross-side seeds
            # at 1.1-1.2 mm in the region before the carry was tried).
            ok, creep = creep_check(env, scene, k, B, limit)
            info['creep_m'] = creep
            why = '' if ok else f'held block creeps or slips ({creep:.4f} m per 2 chunks)'
        if ok and stage == 'drop':
            held = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
            mujoco.mj_getState(env.model, env.data, held, STATE_SPEC)
            tried = []
            for tilt in (0, -30, 30):
                mujoco.mj_setState(env.model, env.data, held, STATE_SPEC); mujoco.mj_forward(env.model, env.data)
                ok, why = carry_to_opening(env, scene, k, B, tilt_deg=tilt, follow=True)
                if ok:
                    ok, creep = creep_check(env, scene, k, B, limit)
                    why = '' if ok else f'held block creeps or slips ({creep:.4f} m per 2 chunks)'
                tried.append(dict(tilt=tilt, why=why))
                if ok:
                    info['tilt'] = tilt
                    break
                if _clock() - began > BUDGET_S + 4:
                    break
            info['chain_tilts'] = tried
        if not ok:
            logs.append({**info, 'why': why}); continue
        o = scene.observe(env.data, rest)
        labels = classify(o, np.full(len(scene.names), R), cfg)
        labels = classify(o, labels, cfg); labels = classify(o, labels, cfg)
        if labels[k] != STAGES.index(stage):
            logs.append({**info, 'why': f'classified as {int(labels[k])}'}); continue
        state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC)); mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
        return dict(seed=int(seed), stage=stage, target=int(k), arm=B, state=state, rest_height=rest, labels=labels,
                    info=info), logs + [{**info, 'ok': True}]
    return None, logs
