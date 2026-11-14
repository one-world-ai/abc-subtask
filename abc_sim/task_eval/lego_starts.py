"""Reproducible LEGO starts with physical chaining and verified sampled-grasp fallbacks."""
from __future__ import annotations

from contextlib import contextmanager
import time

import mujoco
import numpy as np

from abc_sim.task_eval.bottles_starts import bottle_yaw, idle_action, move_link
from abc_sim.task_eval.lego_stages import COLORS, R, STAGES, StageConfig, classify
from abc_sim.task_eval.stagekit import (ARMS, STATE_SPEC, T, arm_ids, axis_angle_quat, grasp_from_state,
                                       hold_object, place_gripper, qmul, set_object_pose, inv, GRIP_MAX)

SEED_BASE = 930_000_000
block_yaw = bottle_yaw  # the verified inertia-axis jaw alignment applies to the rectangular LEGO collision box
# Start-building retry budget in *process CPU seconds* (G3 drop round 3): a wall-clock budget made whether a
# marginal seed builds depend on how loaded the shared node was.
BUDGET_S = 8.5
_clock = time.process_time
CREEP_M = .001  # held-block sink allowed over 64 idle actions at build time (G3 carry round 1)
# G3 drop round 3: 17/17 same-side seeds rejected in round 2 sank 1.00-1.14 mm in that window (grip settling)
# and then stayed in stage for 30 idle chunks with return 0, so 1 mm contradicts the measured physics for drop.
DROP_CREEP_M = .0015
DROP_TILTS = (-30, 0)  # round 2 chain successes: -30 25, 0 7, every other tilt 0 (of 142 tries)


def native_grasp(data, scene, k, yaw, arm, *, pad_z=None):
    """Anchor the measured *finger pads*, rather than the offset grasp site, at the block centre."""
    rotation = np.array([[np.cos(yaw), np.sin(yaw), 0.],
                         [np.sin(yaw), -np.cos(yaw), 0.], [0., 0., -1.]])
    pad = scene.pad_link6[arm].copy()
    if pad_z is not None:
        pad[2] = pad_z
    pose = np.eye(4); pose[:3, :3] = rotation
    pose[:3, 3] = data.subtree_com[scene.bodies[k]] - rotation @ pad
    return dict(arm=int(arm), T_obj_link6=inv(scene.object_T(data,k)) @ pose,
                fingers=np.array([.02,.02]), yaw=float(yaw))


class NumericalInvalid(RuntimeError):
    pass


def warning_counts(data):
    return np.array([int(warning.number) for warning in data.warning])


@contextmanager
def reject_numerics(env):
    step = env.step
    def guarded(action, **kwargs):
        data = env.data
        before, time_before = warning_counts(data), float(data.time)
        if before.any():
            raise NumericalInvalid('simulator state already has warnings')
        try:
            result = step(action, **kwargs)
        finally:
            if (env.data is not data or np.any(warning_counts(data) > before)
                    or data.time <= time_before or not np.all(np.isfinite(data.qpos))
                    or not np.all(np.isfinite(data.qvel))):
                raise NumericalInvalid('numerical warning, reset, or nonfinite simulator state')
        return result
    env.step = guarded
    try:
        yield
    finally:
        env.step = step


def pick_from_open(env, scene, k, arm):
    ids = arm_ids(env.model, ARMS[arm])
    goal = T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])
    pad_world = goal[:3, 3] + goal[:3, :3] @ scene.pad_link6[arm]
    goal[:3, 3] += env.data.subtree_com[scene.bodies[k]]-pad_world
    ok, why = move_link(env, arm, goal, actions=32, grip=1.)
    if not ok:
        return False, why
    action = idle_action(env); action[arm*7+6] = 0
    for _ in range(32): env.step(action, render_obs=False)
    goal = T(env.data.xpos[ids.link6], env.data.xquat[ids.link6]); goal[2, 3] += .08
    ok, why = move_link(env, arm, goal, actions=64)
    if ok:
        action = idle_action(env)
        for _ in range(32): env.step(action, render_obs=False)
    obs = scene.contact_obs(env.data)
    return bool(ok and obs['grasped'][k]), why or 'pick end is not grasped'


def centered_hold(env, scene, k, arm):
    ids=arm_ids(env.model,ARMS[arm]);rotation=env.data.xmat[ids.link6].reshape(3,3)
    pad=env.data.xpos[ids.link6]+rotation@scene.pad_link6[arm]
    along=float((env.data.subtree_com[scene.bodies[k]]-pad)@rotation[:,2])
    return abs(along)<=.006,along


def carry_to_opening(env, scene, k, arm, *, tilt_deg=0, destination=None, follow=False):
    """Carry the held block over an opening keeping the grasp. ``follow`` (cross-side receiver carries, amendment
    3) also turns the block about the vertical by the change in bearing from the arm's base, so a side grasp keeps
    pointing away from the base (without it the receiver's wrist needs a 90 deg turn over its bin: IK fails)."""
    model, data = env.model, env.data
    grasp = grasp_from_state(model, data, scene, k, arm)
    com = data.subtree_com[scene.bodies[k]].copy()
    obj = scene.object_T(data, k)
    destination = int(scene.colors[k]) if destination is None else int(destination)
    a = scene.bin_adrs[destination]
    rim = data.qpos[a:a+3] + data.xmat[scene.bin_bodies[destination]].reshape(3,3) @ [0.,0.,scene.rim_local[destination]]
    base = np.array([.252, .31 if arm == 0 else -.31, 0.])
    towards = rim-base
    axis = np.array([-towards[1], towards[0], 0.]); axis /= np.linalg.norm(axis)
    qr = axis_angle_quat(axis, np.deg2rad(tilt_deg))
    if follow:
        turn = np.arctan2(towards[1], towards[0]) - np.arctan2(com[1]-base[1], com[0]-base[0])
        qr = qmul(qr, axis_angle_quat(np.array([0., 0., 1.]), (turn + np.pi) % (2*np.pi) - np.pi))
    flat = np.empty(9); mujoco.mju_quat2Mat(flat, qr)
    orientation = flat.reshape(3, 3) @ obj[:3, :3]
    offset = obj[:3, :3].T @ (obj[:3, 3]-com)
    high = max(com[2], rim[2]+scene.reaches[k]+.02)
    for location in (np.array([com[0], com[1], high]), np.array([rim[0], rim[1], high])):
        target = np.eye(4); target[:3, :3] = orientation
        target[:3, 3] = location + orientation @ offset
        ok, why = move_link(env, arm, target @ grasp['T_obj_link6'], actions=96)
        if not ok: return False, why
        if not scene.contact_obs(data)['grasped'][k]: return False, 'block slipped during carry'
    action = idle_action(env)
    for _ in range(32): env.step(action, render_obs=False)
    obs = scene.observe(data, env._task_evaluator.tracker.rest_height)
    return bool(obs['grasped'][k] and (destination != scene.colors[k] or obs['over'][k])), 'carry end is not held over the matching opening'


def sampled_drop_hold(env, scene, k, *, tilt_deg, rng, lowered=False, desired_yaw=0.):
    data, model = env.data, env.model
    c = scene.colors[k]; bin_y = data.qpos[scene.bin_adrs[c]+1]
    arm = 0 if bin_y > 0 else 1
    yaw = block_yaw(model, data, scene, k)
    grasp = native_grasp(data, scene, k, yaw, arm)
    com = data.subtree_com[scene.bodies[k]].copy()
    obj = scene.object_T(data, k)
    rim = scene.opening_world(data, k)
    base = np.array([.252, .31 if arm == 0 else -.31])
    towards = rim[:2]-base
    axis = np.array([-towards[1], towards[0], 0.]); axis /= np.linalg.norm(axis)
    qr = qmul(axis_angle_quat(axis, np.deg2rad(tilt_deg)),
              axis_angle_quat(np.array([0.,0.,1.]), desired_yaw-yaw))
    flat=np.empty(9); mujoco.mju_quat2Mat(flat, qr); rotation=flat.reshape(3, 3)
    desired = rim + [0, 0, rng.uniform(.035, .06)]
    if lowered:
        desired = data.qpos[scene.bin_adrs[c]:scene.bin_adrs[c]+3] + data.xmat[scene.bin_bodies[c]].reshape(3,3) @ [0.,0.,scene.geometry._goal_regions[COLORS[c]][0][2]+.01]
    origin = desired + rotation @ (obj[:3, 3]-com)
    set_object_pose(env, scene, k, origin, qmul(qr, data.qpos[scene.qpos_adr[k]+3:scene.qpos_adr[k]+7].copy()))
    ok, detail = hold_object(env, scene, k, grasp, rng=rng)
    detail.update(source='sampled safely lowered hold' if lowered else 'sampled over-bin grasp (fallback after carry chain)', arm=arm)
    return ok, detail, arm


def _build_start(env, seed, stage, *, rng, tries):
    cfg, logs = StageConfig(), []
    began=_clock()
    for attempt in range(1 if stage == 'reach' else tries):
        if attempt and _clock()-began>BUDGET_S:
            break
        env.forget_arm_state(); env.reset(seed=seed, randomize=True); env.evaluate_task()
        scene = env._task_evaluator.scene
        rest = scene.heights(env.data)
        obs = scene.observe(env.data, rest)
        k = env._task_evaluator.pick_target(obs)
        arm = int(obs['near_arm'][k])
        info = dict(attempt=attempt, target=int(k), target_name=scene.names[k], arm=arm,
                    start_side='left' if obs['xy'][k, 1] > 0 else 'right',
                    bin_side='left' if env.data.qpos[scene.bin_adrs[scene.colors[k]]+1] > 0 else 'right',
                    cross_side=bool(obs['cross'][k]))
        # Amendment 3: cross-side blocks go through the handoff. Handoff starts exist only for them (the giver holds
        # it lifted, as pick's goal); their carry/drop starts are held by the receiver after a handoff.
        if stage == 'handoff':
            if not obs['cross'][k]:
                return None, [{**info, 'excluded': 'same-side block: no handoff', 'why': 'same-side block: no handoff'}]
            arm = info['arm'] = int(obs['giver'][k])
        elif stage in ('carry', 'drop') and obs['cross'][k]:
            from abc_sim.task_eval.lego_handoff import build_cross_start
            return build_cross_start(env, seed, stage, rng=rng, tries=tries, began=began)
        if stage != 'reach':
            yaw = block_yaw(env.model, env.data, scene, k) + rng.uniform(-.08, .08) + (0,np.pi,np.pi/2,3*np.pi/2,np.pi/4,5*np.pi/4,3*np.pi/4,7*np.pi/4)[attempt%8]
            grasp = native_grasp(env.data, scene, k, yaw, arm, pad_z=(.142,.145,.139)[attempt%3])
            if stage in ('drop', 'handoff'):
                # G3 drop round 3: the physical pick never gave a pad-centred hold (0/204 carry and 0/100 drop
                # attempts, block 1.0-1.2 cm off the pad axis), so drop goes straight to the sampled hold.
                ok, detail = False, dict(why='drop skips the physical pick')
            else:
                ok, detail = place_gripper(env, scene, k, grasp, rng=rng)
            if ok:
                action=idle_action(env); action[arm*7+6]=1
                for _ in range(10): env.step(action, render_obs=False)
                now=scene.observe(env.data, rest)
                ok = not now['touched'][k] and np.linalg.norm(now['xy'][k]-obs['xy'][k]) < .01
                if not ok: detail['why']='open gripper touches or pushes the block'
            if stage in ('handoff', 'carry', 'drop'):
                if ok:
                    ok, why = pick_from_open(env, scene, k, arm)
                    detail['source']='physical pick solver end state'
                    if ok:
                        ok,along=centered_hold(env,scene,k,arm)
                        detail['pad_axis_offset_m']=along
                        if not ok:why=f'pick holds the block at pad edge ({along:.4f} m)'
                    if not ok: detail['why']=why
                if not ok:
                    a=scene.qpos_adr[k]
                    pos=env.data.qpos[a:a+3].copy(); pos[2]=rest[k, 2]+rng.uniform(.06, .12)
                    set_object_pose(env,scene,k,pos,env.data.qpos[a+3:a+7].copy())
                    grasp=native_grasp(env.data,scene,k,yaw,arm)
                    ok,detail=hold_object(env,scene,k,grasp,rng=rng)
                    detail['source']='sampled table-side hold (fallback after physical pick)'
                if ok and stage == 'drop':
                    # Chain the carry solver: every approach tilt from one verified held snapshot
                    # (G3 round 2; round 1 spent a whole reset + regrasp per tilt and ran out of time).
                    held=np.empty(mujoco.mj_stateSize(env.model,STATE_SPEC))
                    mujoco.mj_getState(env.model,env.data,held,STATE_SPEC)
                    tried=[]
                    # Round 3: only the tilts that ever succeeded; the saved budget buys another grasp yaw.
                    for tilt in DROP_TILTS:
                        if _clock()-began>BUDGET_S:
                            break
                        mujoco.mj_setState(env.model,env.data,held,STATE_SPEC);mujoco.mj_forward(env.model,env.data)
                        ok,why=carry_to_opening(env,scene,k,arm,tilt_deg=tilt)
                        if ok:
                            ok,along=centered_hold(env,scene,k,arm)
                            if not ok:why=f'held block at pad edge ({along:.4f} m)'
                        if ok:
                            z0=float(env.data.subtree_com[scene.bodies[k],2]);action=idle_action(env)
                            for _ in range(64):env.step(action,render_obs=False)
                            creep=z0-float(env.data.subtree_com[scene.bodies[k],2])
                            ok=bool(scene.contact_obs(env.data)['grasped'][k] and creep<=DROP_CREEP_M)
                            if not ok:why=f'held block creeps or slips ({creep:.4f} m per 2 chunks)'
                        tried.append(dict(tilt=tilt,why='' if ok else why))
                        if ok:
                            detail['source']='carry solver chain from a held carry start';detail['tilt']=tilt
                            break
                    detail['chain_tilts']=tried
                    if not ok and attempt == 0:
                        # Round 3: a second table-side grasp yaw first; the over-bin fallback (3/18 in round 2)
                        # only from attempt 1 on.
                        detail['chain_failure']=why
                    elif not ok:
                        chain_why=why
                        env.forget_arm_state();env.reset(seed=seed,randomize=True);env.evaluate_task()
                        scene=env._task_evaluator.scene
                        snapshot=np.empty(mujoco.mj_stateSize(env.model,STATE_SPEC))
                        mujoco.mj_getState(env.model,env.data,snapshot,STATE_SPEC)
                        sampled=[]
                        for tilt,yaw_goal,lowered in ((-30,np.pi,False),(30,0.,False),(-50,np.pi,False),
                                                      (50,0.,False),(0.,np.pi,True),(0.,0.,True)):
                            if _clock()-began>BUDGET_S:
                                break
                            mujoco.mj_setState(env.model,env.data,snapshot,STATE_SPEC);mujoco.mj_forward(env.model,env.data)
                            ok,detail,arm=sampled_drop_hold(env,scene,k,tilt_deg=tilt,rng=rng,
                                                           lowered=lowered,desired_yaw=yaw_goal)
                            sampled.append(detail.copy())
                            if ok:break
                        detail['sample_attempts']=[{key:value for key,value in sample.items() if key!='q'} for sample in sampled]
                        detail['chain_failure']=chain_why
                if ok:
                    ok,along=centered_hold(env,scene,k,arm)
                    detail['pad_axis_offset_m']=along
                    if ok:
                        z0=float(env.data.subtree_com[scene.bodies[k],2]);action=idle_action(env)
                        for _ in range(64):env.step(action,render_obs=False)
                        creep=z0-float(env.data.subtree_com[scene.bodies[k],2])
                        ok=bool(scene.contact_obs(env.data)['grasped'][k] and creep<=(DROP_CREEP_M if stage=="drop" else CREEP_M))
                        detail['creep_m']=creep
                        if not ok:detail['why']=f'held block creeps or slips ({creep:.4f} m per 2 chunks)'
                    else:detail['why']=f'held block at pad edge ({along:.4f} m)'
            info.update({key:value for key,value in detail.items() if key != 'q'})
            if not ok:
                logs.append(info); continue
        obs=scene.observe(env.data,rest)
        labels=classify(obs,np.full(len(scene.names),R),cfg)
        if labels[k] != STAGES.index(stage):
            logs.append({**info,'why':f'classified as {int(labels[k])}'}); continue
        state=np.empty(mujoco.mj_stateSize(env.model,STATE_SPEC)); mujoco.mj_getState(env.model,env.data,state,STATE_SPEC)
        return dict(seed=int(seed),stage=stage,target=int(k),arm=arm if stage != 'reach' else -1,
                    state=state,rest_height=rest,labels=labels,info=info),logs+[{**info,'ok':True}]
    return None,logs


def build_start(env, seed, stage, *, rng=None, tries=8):
    rng = rng or np.random.default_rng(seed)
    try:
        with reject_numerics(env):
            return _build_start(env,seed,stage,rng=rng,tries=tries)
    except NumericalInvalid as error:
        return None,[dict(seed=int(seed),numerical_invalid=True,why=str(error))]
