"""Clutter start construction and task-local collision-checked Cartesian plans.

No target substitution: index zero always denotes the prompted object. Failed
IK/contact paths are reported as explicit carry-coverage exclusions.
"""
from __future__ import annotations
import mujoco
import numpy as np
from abc_sim.task_eval.clutter_stages import R,STAGES,StageConfig,classify
from abc_sim.task_eval.put_relative_starts import topdown_grasp
from abc_sim.task_eval.stagekit import ARMS,STATE_SPEC,T,actuator_limits,arm_ids,grasp_from_state,hold_object,ik_link6,place_gripper,set_object_pose,inv,axis_angle_quat,GRIP_MAX,subtree
SEED_BASE=940_000_000

def idle_action(env):
    a=np.zeros(14,np.float32)
    for i,arm in enumerate(ARMS):
        a[i*7:i*7+6]=env.data.qpos[arm_ids(env.model,arm).qadr]
        a[i*7+6]=env.data.ctrl[env._ctrl_indices[i*7+6]]/.0475
    return a

def native_grasp(data,scene,k,yaw,arm,*,pad_z=.1297):
    # Actual pad centre in link6 coordinates; the named TCP has a 4.4 cm lateral offset.
    r=np.array([[np.cos(yaw),np.sin(yaw),0.],[np.sin(yaw),-np.cos(yaw),0.],[0.,0.,-1.]])
    pose=np.eye(4);pose[:3,:3]=r
    pose[:3,3]=data.subtree_com[scene.bodies[k]]-r@np.array([0.,-.044,pad_z])
    return dict(arm=int(arm),T_obj_link6=inv(scene.object_T(data,k))@pose,fingers=np.array([GRIP_MAX,GRIP_MAX]),yaw=float(yaw))

def tilted_grasp(env,scene,grasp,tilt):
    pose=scene.object_T(env.data,0)@grasp['T_obj_link6'];com=env.data.subtree_com[scene.bodies[0]].copy()
    base=env.data.xpos[env.model.jnt_bodyid[env.model.joint(ARMS[grasp['arm']]+'_joint1').id]]
    direction=com-base;axis=np.array([-direction[1],direction[0],0.]);axis/=np.linalg.norm(axis)
    anchor=pose[:3,:3].T@(com-pose[:3,3]);pose[:3,:3]=_rotation(axis_angle_quat(axis,np.deg2rad(tilt)))@pose[:3,:3]
    pose[:3,3]=com-pose[:3,:3]@anchor;grasp['T_obj_link6']=inv(scene.object_T(env.data,0))@pose
    return grasp

def _rotation(quat):
    flat=np.empty(9);mujoco.mju_quat2Mat(flat,quat);return flat.reshape(3,3)

def plan_link(env,arm,goal,*,held=False,waypoints=12,allow_target_collision=False,collision_check=True):
    """IK plus collision checks along the interpolated link/object path; does not execute."""
    m,d,scene=env.model,env.data,env._task_evaluator.scene;ids=arm_ids(m,ARMS[arm])
    start=T(d.xpos[ids.link6],d.xquat[ids.link6]);qa=np.empty(4);qb=np.empty(4)
    mujoco.mju_mat2Quat(qa,start[:3,:3].ravel());mujoco.mju_mat2Quat(qb,goal[:3,:3].ravel())
    if qa@qb<0:qb=-qb
    scratch=mujoco.MjData(m);scratch.qpos[:]=d.qpos
    active_bodies=subtree(m,int(m.jnt_bodyid[m.joint(ARMS[arm]+'_joint1').id]))
    active_geoms={g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in active_bodies}
    baseline={tuple(sorted((int(c.geom1),int(c.geom2)))) for c in d.contact}
    relative=grasp_from_state(m,d,scene,0,arm)['T_obj_link6'] if held else None
    paths=[]
    for fraction in np.linspace(0,1,waypoints+1)[1:]:
        qquat=qa*(1-fraction)+qb*fraction;qquat/=np.linalg.norm(qquat)
        pose=np.eye(4);pose[:3,:3]=_rotation(qquat);pose[:3,3]=start[:3,3]*(1-fraction)+goal[:3,3]*fraction
        q,pe,re=ik_link6(m,scratch,ARMS[arm],pose,limits=actuator_limits(m,ARMS[arm]))
        if q is None or pe>.008 or re>.2:return None,f'IK unreachable: position {pe:.4f} rotation {re:.3f}'
        scratch.qpos[ids.qadr]=q;mujoco.mj_forward(m,scratch)
        if held:
            ot=T(scratch.xpos[ids.link6],scratch.xquat[ids.link6])@inv(relative)
            a=scene.qpos_adr[0];quat=np.empty(4);mujoco.mju_mat2Quat(quat,ot[:3,:3].ravel())
            scratch.qpos[a:a+3]=ot[:3,3];scratch.qpos[a+3:a+7]=quat;mujoco.mj_forward(m,scratch)
        for c in scratch.contact:
            x,y=int(c.geom1),int(c.geom2);ox,oy=scene.owner[x],scene.owner[y]
            target_finger=(allow_target_collision and ((ox[0]==0 and ox[1]==0) or (oy[0]==0 and oy[1]==0))) or (ox[0]==0 and ox[1]==0 and oy[0]==1 and oy[1]//2==arm) or (oy[0]==0 and oy[1]==0 and ox[0]==1 and ox[1]//2==arm)
            relevant=(x in active_geoms or y in active_geoms or held and ((ox[0]==0 and ox[1]==0) or (oy[0]==0 and oy[1]==0)))
            unchanged_self=(x in active_geoms and y in active_geoms and tuple(sorted((x,y))) in baseline)
            if collision_check and relevant and not target_finger and not unchanged_self and c.dist<-.001:
                return None,f'collision on direct path: {m.geom(x).name}/{m.geom(y).name}'
        paths.append(q)
    return paths,''

def move_link(env,arm,goal,*,held=False,grip=0.,actions=96,allow_target_collision=False,collision_check=True):
    path,why=plan_link(env,arm,goal,held=held,allow_target_collision=allow_target_collision,collision_check=collision_check)
    if path is None:return False,why
    hold=idle_action(env)
    for q in path:
        a=hold.copy();a[arm*7:arm*7+6]=q;a[arm*7+6]=grip
        for _ in range(max(1,actions//len(path))):env.step(a,render_obs=False)
    return True,''

def opening_pose(env,scene,*,extra=.015,tilt=0.,arm=0):
    """Keep target orientation and align its origin to the live goal centre above the actual rim."""
    data=env.data;obj=scene.object_T(data,0);rb=data.xmat[scene.box_body].reshape(3,3)
    if tilt:
        ids=arm_ids(env.model,ARMS[arm]);base=env.data.xpos[env.model.jnt_bodyid[env.model.joint(ARMS[arm]+'_joint1').id]]
        direction=data.qpos[scene.box_adr:scene.box_adr+3]-base;axis=np.array([-direction[1],direction[0],0.]);axis/=np.linalg.norm(axis)
        obj[:3,:3]=_rotation(axis_angle_quat(axis,np.deg2rad(tilt)))@obj[:3,:3]
    r=rb.T@obj[:3,:3];lowest=float(np.min(scene.geometry._target_collision_points@r[2]))
    local=scene.geometry._goal_local_center.copy();local[2]=scene.geometry._box_rim_z+scene.cfg.clearance_m+extra-lowest
    desired=obj.copy();desired[:3,3]=data.qpos[scene.box_adr:scene.box_adr+3]+rb@local
    return desired

CARRY_YAWS=(90.,-90.,180.,45.,-45.)  # handoff amendment: tried only after every original (yaw 0) plan failed

def carry_plan(env,arm,*,execute=False):
    scene=env._task_evaluator.scene;d,m=env.data,env.model;g=grasp_from_state(m,d,scene,0,arm)
    notes=[];com=d.subtree_com[scene.bodies[0]].copy()
    for yaw,tilt in [(0.,t) for t in (0.,-35.,35.,-50.,50.)]+[(y,t) for y in CARRY_YAWS for t in (0.,-35.,35.)]:
        if yaw:
            # Hand + target turned about the vertical through the CoM before aligning with the opening (a receiver
            # that took the target side-on from its side points away from the box).
            Rz=_rotation(axis_angle_quat(np.array([0.,0.,1.]),np.deg2rad(yaw)));obj=scene.object_T(d,0)
            turned=obj.copy();turned[:3,:3]=Rz@obj[:3,:3];turned[:3,3]=com+Rz@(obj[:3,3]-com)
            a=scene.qpos_adr[0];saved=d.qpos[a:a+7].copy()
            set_object_pose(env,scene,0,turned[:3,3],quat_from_matrix(turned[:3,:3]))
            desired=opening_pose(env,scene,tilt=tilt,arm=arm)
            d.qpos[a:a+7]=saved;mujoco.mj_forward(m,d)
        else:desired=opening_pose(env,scene,tilt=tilt,arm=arm)
        goal=desired@g['T_obj_link6']
        if yaw and plan_link(env,arm,goal,held=True,waypoints=1)[0] is None:notes.append('end pose unreachable');continue
        path,why=plan_link(env,arm,goal,held=True)
        if path is not None:
            if not execute:return path,''
            return move_link(env,arm,goal,held=True,actions=128)
        notes.append(why)
    return (False if execute else None),'; '.join(notes)

def place_candidates(env,arm,*,releases=(.03,),yaws=(0.,90.,-90.),shifts=(0.,.02,-.02,.04,-.04),margin=.005):
    """Link-6 goals that lower the held target straight into the live box: release clearance above the physical
    floor (< drop_clearance), a yaw of hand+target about the target CoM, and an XY shift that keeps the target origin
    inside the scorer's goal AABB (minus ``margin``). Ordered smallest change first."""
    d=env.data;ev=env._task_evaluator;s=ev.scene;ids=arm_ids(env.model,ARMS[arm])
    clearance=float(s.observe(d,ev.tracker.rest_height)['clearance'][0])
    rb=d.xmat[s.box_body].reshape(3,3);box=d.qpos[s.box_adr:s.box_adr+3]
    origin=d.qpos[s.qpos_adr[0]:s.qpos_adr[0]+3].copy();com=d.subtree_com[s.bodies[0]].copy()
    cen,half=s.geometry._goal_local_center,s.geometry._goal_half_size
    link=T(d.xpos[ids.link6],d.xquat[ids.link6]);out=[]
    for release in releases:
        for yaw in yaws:
            r=_rotation(axis_angle_quat(np.array([0.,0.,1.]),np.deg2rad(yaw)))
            for dx in shifts:
                for dy in shifts:
                    shift=rb@np.array([dx,dy,0.]);shift[2]-=max(clearance-release,0.)
                    final=com+r@(origin-com)+shift
                    if np.any(np.abs((final-box)@rb-cen)[:2]>half[:2]-margin):continue
                    goal=link.copy();goal[:3,:3]=r@link[:3,:3];goal[:3,3]=com+r@(link[:3,3]-com)+shift
                    out.append((dict(release_m=release,yaw_deg=yaw,shift_m=(dx,dy)),goal))
    return out

def place_plan(env,arm,*,execute=False,max_candidates=None):
    """First collision-free straight lowering (held target, fingers and arm vs box walls/clutter) into the box.
    Returns (path or executed flag, choice or why)."""
    why='';cands=place_candidates(env,arm)
    for choice,goal in cands[:max_candidates]:
        # Cheap pre-filter: the deepest pose alone (one IK + collision check) before the full 16-waypoint path.
        path,why=plan_link(env,arm,goal,held=True,waypoints=1)
        if path is None:continue
        path,why=plan_link(env,arm,goal,held=True,waypoints=16)
        if path is None:continue
        if not execute:return path,choice
        ok,why=move_link(env,arm,goal,held=True,actions=128)
        return ok,(choice if ok else why)
    return (False if execute else None),f'no collision-free lowering in {len(cands)} candidates; last: {why}'

def build_start(env,seed,stage,*,rng=None,tries=16):
    if stage in ('handoff','carry'):
        # Handoff scenes (picking arm != box arm): handoff starts, and carry starts after a handoff. Direct-reach
        # scenes keep the original carry builder below; they have no handoff start.
        row,log=build_handoff_start(env,seed,stage,rng=rng,tries=tries)
        if stage=='handoff' or not (log and log[-1].get('skipped')):return row,log
    rng=rng or np.random.default_rng(seed);logs=[];cfg=StageConfig()
    for attempt in range(1 if stage=='reach' else tries):
        env.forget_arm_state()
        try:env.reset(seed=seed,randomize=True);env.evaluate_task()
        except (RuntimeError,ValueError) as ex:return None,[dict(why=f'reset invalid: {ex}',reset_invalid=True)]
        m,d,ev=env.model,env.data,env._task_evaluator;scene=ev.scene;rest=scene.heights(d)
        original=scene.object_T(d,0);obs=scene.observe(d,rest)
        near=int(obs['near_arm'][0]);arm=near if stage in ('pick','carry') else (near if attempt<8 else 1-near)
        yaw0=float(np.arctan2(d.xmat[scene.bodies[0]].reshape(3,3)[1,0],d.xmat[scene.bodies[0]].reshape(3,3)[0,0]))
        yaw=float(yaw0+(0.,np.pi,np.pi/2,3*np.pi/2)[attempt%4]+rng.uniform(-.06,.06))
        use_native=attempt%8<4 or stage in ('pick','carry') and attempt>=8
        info=dict(attempt=attempt,target=0,arm=arm,target_label=scene.label,category=getattr(env._last_randomization,'metadata',{}).get('target_category'),source='reset')
        if stage!='reach':
            grasp=native_grasp(d,scene,0,yaw,arm,pad_z=.145 if stage=='pick' else .1297) if use_native else topdown_grasp(d,scene,0,yaw,arm,below_com_m=0. if stage=='pick' else .01)
            info['grasp_mode']='native_pad' if use_native else 'site'
            if stage in ('pick','carry') and attempt>=8:
                grasp=tilted_grasp(env,scene,grasp,-30.)
                info['approach_tilt_deg']=-30.
            if stage=='pick' and attempt>=12:
                pose=scene.object_T(d,0)@grasp['T_obj_link6'];base=d.xpos[m.jnt_bodyid[m.joint(ARMS[arm]+'_joint1').id]]
                direction=base-d.subtree_com[scene.bodies[0]];direction[2]=0.;direction/=np.linalg.norm(direction)
                pose[:3,3]+=direction*.02;grasp['T_obj_link6']=inv(scene.object_T(d,0))@pose
                info['approach_offset_m']=.02
            if stage=='pick':
                ok,detail=place_gripper(env,scene,0,grasp,rng=rng)
                if ok:
                    a=idle_action(env);a[arm*7+6]=1
                    for _ in range(10):env.step(a,render_obs=False)
                    obs=scene.observe(d,rest)
                    ok=not obs['touched'][0] and obs['moved_xy'][0]<.01
                    if not ok:detail['why']='pregrasp touches or moves target'
                info['source']='sampled open target grasp'
            else:
                target=original.copy();target[2,3]+=float(rng.uniform(.07,.12))
                if stage=='place':
                    arm=0 if d.qpos[scene.box_adr+1]>0 else 1
                    # Measured on 24 seeds: +35 deg lean never had IK; level grasps and a -35 deg lean do.
                    tilt=0. if attempt<4 else -35.
                    target=opening_pose(env,scene,extra=float(rng.uniform(.01,.04)),tilt=tilt,arm=arm)
                    # Jaw axis aligned with a box side so the fingers can enter the 14 x 13 cm interior.
                    rb=d.xmat[scene.box_body].reshape(3,3);yaw=float(np.arctan2(rb[1,0],rb[0,0])+np.pi/2*(attempt%4)+rng.uniform(-.06,.06))
                    raise_m=.015 if attempt>=12 else 0.;native=attempt<4 or attempt>=8
                    grasp=native_grasp(d,scene,0,yaw,arm,pad_z=.1297+raise_m) if native else topdown_grasp(d,scene,0,yaw,arm,below_com_m=.01)
                    info['arm']=arm;info['tilt_deg']=tilt;info['grasp_mode']='native_pad' if native else 'site';info['pad_raise_m']=raise_m
                set_object_pose(env,scene,0,target[:3,3],quat_from_matrix(target[:3,:3]))
                if stage!='place':
                    grasp=native_grasp(d,scene,0,yaw,arm) if use_native else topdown_grasp(d,scene,0,yaw,arm,below_com_m=.01)
                    if attempt>=8:grasp=tilted_grasp(env,scene,grasp,-30.)
                ok,detail=hold_object(env,scene,0,grasp,rng=rng)
                info['source']='sampled held target grasp' if stage=='carry' else 'sampled over-box grasp'
                if ok and stage=='carry':
                    path,why=carry_plan(env,arm)
                    info['direct_path_verified']=path is not None
                    if path is None:
                        ok=False;detail['why']=why;info['coverage_exclusion']=True
                if ok and stage=='place':
                    # Stage table: a place start needs a grasp that can lower without fingers hitting the rim.
                    path,choice=place_plan(env,arm)
                    info['lowering_verified']=path is not None
                    if path is None:ok=False;detail['why']=choice;info['lowering_exclusion']=True
                if ok:
                    # Reject obvious creeping grasps before the full G3 ten-chunk check.
                    z0=float(d.qpos[scene.qpos_adr[0]+2]);a=idle_action(env)
                    for _ in range(32):env.step(a,render_obs=False)
                    creep=z0-float(d.qpos[scene.qpos_adr[0]+2]);detail['creep_mm']=float(creep*1000)
                    if creep>.0015:ok=False;detail['why']='held target creeps >1.5 mm/chunk'
            info.update({k:v for k,v in detail.items() if k!='q'})
            if not ok:logs.append(info);continue
        obs=scene.observe(d,rest);labels=classify(obs,np.full(len(scene.names),R),cfg)
        if labels[0]!=STAGES.index(stage):logs.append({**info,'why':f'wrong start label {labels[0]}'});continue
        # Setup must also preserve all non-target objects and fixture.
        if np.any(obs['off_table']) or obs['box_moved']>cfg.box_moved_m or obs['box_tilt']>cfg.box_tilt_rad:
            logs.append({**info,'why':'setup moved box or lost an object'});continue
        state=np.empty(mujoco.mj_stateSize(m,STATE_SPEC));mujoco.mj_getState(m,d,state,STATE_SPEC)
        row=dict(seed=int(seed),stage=stage,target=0,arm=arm if stage!='reach' else -1,state=state,rest_height=rest,labels=labels,info=info)
        return row,logs+[{**info,'ok':True}]
    if stage=='place':
        # Grasp source 1 (handoff amendment, place coverage): a carry start of this seed (direct or after a
        # handoff) + the scripted carry, then the same lowering verification.
        row,clog=place_from_carry(env,seed,rng=rng)
        return row,logs+clog
    return None,logs

def place_from_carry(env,seed,*,rng=None):
    cfg=StageConfig();row,log=build_start(env,seed,'carry',rng=rng)
    if row is None:return None,[dict(source='chained carry solver',why='no carry start: '+str(log[-1].get('why','') if log else ''))]
    arm=int(row['arm']);rest=row['rest_height'];info=dict(source='chained carry solver',arm=arm,carry_source=row['info'].get('source'))
    ok,why=carry_plan(env,arm,execute=True)
    if not ok:return None,[{**info,'why':'carry: '+str(why)}]
    a=idle_action(env)
    for _ in range(32):env.step(a,render_obs=False)
    m,d,scene=env.model,env.data,env._task_evaluator.scene
    path,choice=place_plan(env,arm);info['lowering_verified']=path is not None
    if path is None:return None,[{**info,'why':str(choice),'lowering_exclusion':True}]
    z0=float(d.qpos[scene.qpos_adr[0]+2])
    for _ in range(32):env.step(a,render_obs=False)
    creep=z0-float(d.qpos[scene.qpos_adr[0]+2]);info['creep_mm']=float(creep*1000)
    if creep>.0015:return None,[{**info,'why':'held target creeps >1.5 mm/chunk'}]
    obs=scene.observe(d,rest);labels=np.where(np.arange(len(scene.names))==0,STAGES.index('carry'),R)
    for _ in range(3):labels=classify(obs,labels,cfg)
    if labels[0]!=STAGES.index('place'):return None,[{**info,'why':f'wrong start label {labels[0]}'}]
    if np.any(obs['off_table']) or obs['box_moved']>cfg.box_moved_m or obs['box_tilt']>cfg.box_tilt_rad:
        return None,[{**info,'why':'setup moved box or lost an object'}]
    state=np.empty(mujoco.mj_stateSize(m,STATE_SPEC));mujoco.mj_getState(m,d,state,STATE_SPEC)
    out=dict(seed=int(seed),stage='place',target=0,arm=arm,state=state,rest_height=rest,labels=labels,info=info)
    if 'receiver' in row['info']:info['receiver']=row['info']['receiver']
    return out,[{**info,'ok':True}]

def quat_from_matrix(r):
    q=np.empty(4);mujoco.mju_mat2Quat(q,np.asarray(r).ravel());return q


# ------------------------------------------------------------------------------------------------ handoff (2026-11-14)
# Scenes whose picking arm (the arm nearest the target at reset, as for pick starts) is not the box arm need a
# handoff: the box is at y = +-0.515 and only the box-side arm reaches it (53/53 such direct carries were
# "IK unreachable" in the direct-reach pilot). Giver = picking arm, receiver = box arm. The handoff itself reuses
# the inhand pieces (pinch-centre grasps, monitored release, retreat) through a target-only scene view.

import time

from abc_sim.task_eval import inhand_starts as IS
from abc_sim.task_eval.clutter_stages import H

HANDOFF_POINTS = ((0.56, 0.03, 0.95), (0.52, 0.0, 0.92), (0.60, 0.05, 0.98), (0.55, 0.0, 1.03))  # CoM, y towards giver
HANDOFF_BUDGET_S = 9.0  # CPU s per handoff start
CARRY_CHAIN_S, CARRY_BUDGET_S = 8.0, 16.0  # CPU s: chained handoff, then the source-3 fallback


class TargetView:
    """The prompted target as a one-object scene for the inhand helpers (they index object 0 and reshape per-object
    contact arrays to one object)."""

    def __init__(self, scene):
        self.s = scene
        self.model = scene.model
        self.names = scene.names[:1]
        self.bodies = scene.bodies[:1]
        self.qpos_adr, self.dof_adr = scene.qpos_adr[:1], scene.dof_adr[:1]
        self.owner, self.arm_of = scene.owner, scene.arm_of

    def object_T(self, data, k=0):
        return self.s.object_T(data, 0)

    def world_points(self, data):
        return self.s.world_points(data)

    def contact_obs(self, data):
        return {k: v[:1] for k, v in self.s.contact_obs(data).items()}


def box_arm(env) -> int:
    s = env._task_evaluator.scene
    return 0 if env.data.qpos[s.box_adr + 1] > 0 else 1


def needs_handoff(env) -> bool:
    """At a reset: the picking (nearest) arm is not the box arm."""
    s = env._task_evaluator.scene
    o = s.observe(env.data, s.heights(env.data))
    return int(o['near_arm'][0]) != box_arm(env)


def lift_hold(env, scene, arm, attempt, rng):
    """The old carry-start hold: target lifted 7-12 cm above its reset pose, ``arm`` holding it (native-pad or site
    top-down grasp, -30 deg approach tilt from attempt 8), plus a one-chunk creep check. -> (ok, info)."""
    d = env.data
    original = scene.object_T(d, 0)
    yaw0 = float(np.arctan2(d.xmat[scene.bodies[0]].reshape(3, 3)[1, 0], d.xmat[scene.bodies[0]].reshape(3, 3)[0, 0]))
    yaw = float(yaw0 + (0., np.pi, np.pi / 2, 3 * np.pi / 2)[attempt % 4] + rng.uniform(-.06, .06))
    native = attempt % 8 < 4 or attempt >= 8
    target = original.copy()
    target[2, 3] += float(rng.uniform(.07, .12))
    set_object_pose(env, scene, 0, target[:3, 3], quat_from_matrix(target[:3, :3]))
    grasp = native_grasp(d, scene, 0, yaw, arm) if native else topdown_grasp(d, scene, 0, yaw, arm, below_com_m=.01)
    if attempt >= 8:
        grasp = tilted_grasp(env, scene, grasp, -30.)
    info = dict(grasp_mode='native_pad' if native else 'site', approach_tilt_deg=-30. if attempt >= 8 else 0.)
    ok, detail = hold_object(env, scene, 0, grasp, rng=rng)
    info.update({k: v for k, v in detail.items() if k != 'q'})
    if ok:
        z0 = float(d.qpos[scene.qpos_adr[0] + 2])
        a = idle_action(env)
        for _ in range(32):
            env.step(a, render_obs=False)
        creep = z0 - float(d.qpos[scene.qpos_adr[0] + 2])
        info['creep_mm'] = float(creep * 1000)
        if creep > .0015:
            ok = False
            info['why'] = 'held target creeps >1.5 mm/chunk'
    return ok, info


TRANSPORT_TURNS = [(0., 0.)] + [(t, y) for t in (-20., 20., -40., 40.) for y in (0., -30., 30.)]


def aligned_turns(env, giver, tilts=(0., -20., 20., -40., 40.)):
    """(tilt, yaw) turns that bring the giver's closing axis (link_6 x) onto world y: the receiver then comes in from
    its side and closes along x on the faces the giver leaves free."""
    ids = arm_ids(env.model, ARMS[giver])
    x = env.data.xmat[ids.link6].reshape(3, 3)[:, 0]
    phi = float(np.rad2deg(np.pi / 2 - np.arctan2(x[1], x[0])))
    yaws = sorted({((phi + k * 180. + 180.) % 360.) - 180. for k in (0, 1)}, key=abs)
    return [(t, y) for t in tilts for y in yaws]


def transport(env, giver, points=HANDOFF_POINTS, turns=None):
    """Giver moves the held target's CoM into the handoff region, hand + target rotated about the CoM by a tilt
    (about the horizontal axis across the arm's reach direction) and a yaw: first the turns that align the giver's
    closing axis with y, then generic ones (the wrist cannot keep a top-down pose that high). The first fully planned
    straight move (collision-checked with the held target) is executed; if none, the target goes 10 cm straight up
    first and the search repeats from there. -> (ok, why)."""
    s = env._task_evaluator.scene
    d, m = env.data, env.model
    ids = arm_ids(m, ARMS[giver])
    sg = 1.0 if giver == 0 else -1.0
    base = d.xpos[m.jnt_bodyid[m.joint(ARMS[giver] + '_joint1').id]]
    turns = aligned_turns(env, giver) + TRANSPORT_TURNS if turns is None else turns
    whys = []
    for phase in (0, 1):
        if phase:
            up = T(d.xpos[ids.link6], d.xquat[ids.link6])
            up[2, 3] += .10
            ok, why = move_link(env, giver, up, held=True, actions=64)
            if not ok:
                return False, '; '.join(whys[-2:] + ['up: ' + why])
        com = d.subtree_com[s.bodies[0]].copy()
        link = T(d.xpos[ids.link6], d.xquat[ids.link6])
        for tilt, yaw in turns:
            for p in points:
                p = np.array([p[0], sg * p[1], p[2]])
                direction = p - base
                axis = np.array([-direction[1], direction[0], 0.])
                axis /= np.linalg.norm(axis)
                Rr = _rotation(axis_angle_quat(np.array([0., 0., 1.]), np.deg2rad(yaw))) @ _rotation(
                    axis_angle_quat(axis, np.deg2rad(tilt)))
                goal = np.eye(4)
                goal[:3, :3] = Rr @ link[:3, :3]
                goal[:3, 3] = p + Rr @ (link[:3, 3] - com)
                if plan_link(env, giver, goal, held=True, waypoints=1)[0] is None:  # cheap end-pose check first
                    continue
                path, why = plan_link(env, giver, goal, held=True)
                if path is None:
                    whys.append(why)
                    continue
                return move_link(env, giver, goal, held=True, actions=128)
    return False, '; '.join(whys[-3:]) or 'no reachable pose in the handoff region'


def receiver_candidates(env, view, giver):
    """Receiver (box arm) pinch-centre grasps of the target where the giver holds it, most promising first:
    side-on from the receiver's side closing along x (the faces a giver closing along y leaves free), turned /
    pitched a little, then closing vertically, then on the lower part, then the inhand handoff families. World T6s."""
    d = env.data
    sA = 1.0 if giver == 0 else -1.0
    g = IS.item_geometry(env, view, giver)
    com = np.asarray(d.subtree_com[view.bodies[0]])
    height = g['hi'][2] - g['lo'][2]
    out = []
    for rolls, dzs in (((0.0, np.pi), (0.0,)), ((np.pi / 2, -np.pi / 2), (0.0,)), ((0.0, np.pi), (-0.25, 0.25))):
        for dz in dzs:
            pt = com + np.array([0., 0., dz * height])
            for turn, pitch in ((0., 0.), (.3, 0.), (-.3, 0.), (0., .3), (0., -.3), (.6, 0.), (-.6, 0.)):
                app = IS.Rz(turn) @ np.array([0.0, sA * np.cos(pitch), np.sin(pitch)])
                for roll in rolls:
                    out.append(IS.side_T6(pt, app, roll))
    for mode in IS.MODES:
        for fa in (0, 1):
            for fb in (0, 1):
                P = IS.handoff_plan(env, view, mode, giver, fa, fb)
                if P is not None:
                    out.append(P[1])
    return out


def exchange(env, giver, *, teleport=False, tries=3, deadline=None, max_scan=None, hint=None):
    """The receiver takes the target from the giver in the air (inhand ``handoff_solver`` steps: pre-grasp in joint
    space, straight in, close; only with both receiver fingers on it the giver opens (monitored), then retreats).
    ``teleport`` (start fallback only) puts the receiver at its grasp. -> (ok, why)."""
    s = env._task_evaluator.scene
    view = TargetView(s)
    B = 1 - giver
    whys, done = [], 0
    seen = []
    first = [] if hint is None else [s.object_T(env.data, 0) @ hint]
    for T6 in first + receiver_candidates(env, view, giver)[:max_scan]:
        if deadline and time.process_time() > deadline:
            whys.append('time budget')
            break
        got = IS.check_receiver(env, view, B, T6)
        if got is None or any(np.allclose(got[0], o, atol=2e-3) for o in seen):
            continue
        seen.append(got[0])
        TB, qB = got
        if teleport:
            IS.teleport_arm(env, B, qB)
            IS.run(env, IS.action_with(env, {B: qB}, {B: 1.0}), 4)
            qpre = None
        else:
            qpre = IS.approach_pose(env, B, TB, qB)
            if qpre is None:
                whys.append('pre-grasp unreachable')
                continue
            IS.move_joints(env, B, qpre, actions=96, grip=1.0)
            ok, why = IS.move_link(env, B, TB, actions=48, grip=1.0, n_way=4)
            if not ok:
                whys.append('grasp: ' + why)
                IS.move_joints(env, B, qpre, actions=32, grip=1.0)
                continue
        IS.run(env, IS.action_with(env, None, {B: 0.0}), 24)
        f = view.contact_obs(env.data)['hits'].reshape(2, 2).sum(1)
        if f[B] == 2 and f[giver] == 2 and IS.monitored_release(env, view, giver):
            retreat(env, giver)
            IS.run(env, IS.current_action(env), 16)
            return True, ''
        whys.append(f'receiver closed with {f[B]} fingers' if f[B] < 2 else 'receiver could not hold it alone')
        IS.run(env, IS.action_with(env, None, {B: 1.0}), 12)
        if qpre is not None:
            IS.move_joints(env, B, qpre, actions=32, grip=1.0)
        done += 1
        if done >= tries:
            break
    return False, '; '.join(whys[-3:]) or 'no reachable receiver grasp'


def _restore(env, snap):
    IS.restore(env, snap)
    env._task_evaluator._last_sim_time = None  # start building rewinds physics time on purpose


def _teleport_held(env, giver, q, rel):
    """Giver at joints q, the target moved rigidly with its link_6 (grasp ``rel`` = T_obj_link6)."""
    s = env._task_evaluator.scene
    ids = arm_ids(env.model, ARMS[giver])
    env.data.qpos[ids.qadr] = q
    env.data.qvel[ids.dadr] = 0.
    mujoco.mj_forward(env.model, env.data)
    ot = T(env.data.xpos[ids.link6], env.data.xquat[ids.link6]) @ inv(rel)
    set_object_pose(env, s, 0, ot[:3, 3], quat_from_matrix(ot[:3, :3]))


def plan_handoff(env, giver, *, max_poses=4, max_scan=30, deadline=None):
    """Choose where the giver presents the target: region poses (closing axis along y, then along x, tilts/yaws), each
    with a fully planned collision-checked straight path; the first whose presented target has a receiver grasp that
    is reachable, touches nothing when open and has a reachable pre-grasp (checked by teleporting giver + target in a
    snapshot). -> (goal T6 of the giver, receiver T_obj grasp) or (None, why)."""
    s = env._task_evaluator.scene
    d, m = env.data, env.model
    ids = arm_ids(m, ARMS[giver])
    sg = 1.0 if giver == 0 else -1.0
    B = 1 - giver
    base = d.xpos[m.jnt_bodyid[m.joint(ARMS[giver] + '_joint1').id]]
    view = TargetView(s)
    rel = grasp_from_state(m, d, s, 0, giver)['T_obj_link6']
    al = aligned_turns(env, giver, tilts=(0., -20., 20.))
    turns = al + [(t, y + 90.) for t, y in al] + [(t, y - 90.) for t, y in al] + TRANSPORT_TURNS
    com = d.subtree_com[s.bodies[0]].copy()
    link = T(d.xpos[ids.link6], d.xquat[ids.link6])
    snap = IS.snapshot(env)
    tried, whys = 0, []
    for tilt, yaw in turns:
        for p in HANDOFF_POINTS[:2]:
            if tried >= max_poses or deadline and time.process_time() > deadline:
                return None, '; '.join(whys[-3:]) or 'no region pose'
            p = np.array([p[0], sg * p[1], p[2]])
            axis = np.array([-(p - base)[1], (p - base)[0], 0.])
            axis /= np.linalg.norm(axis)
            Rr = _rotation(axis_angle_quat(np.array([0., 0., 1.]), np.deg2rad(yaw))) @ _rotation(
                axis_angle_quat(axis, np.deg2rad(tilt)))
            goal = np.eye(4)
            goal[:3, :3] = Rr @ link[:3, :3]
            goal[:3, 3] = p + Rr @ (link[:3, 3] - com)
            if plan_link(env, giver, goal, held=True, waypoints=1)[0] is None:
                continue
            path, why = plan_link(env, giver, goal, held=True)
            if path is None:
                whys.append(why)
                continue
            tried += 1
            _teleport_held(env, giver, path[-1], rel)
            found = None
            for T6 in receiver_candidates(env, view, giver)[:max_scan]:
                got = IS.check_receiver(env, view, B, T6)
                if got is not None and IS.approach_pose(env, B, *got) is not None:
                    found = inv(s.object_T(d, 0)) @ got[0]
                    break
            _restore(env, snap)
            if found is not None:
                return (goal, found), ''
            whys.append(f'tilt {tilt:+.0f} yaw {yaw:+.0f}: no receiver grasp')
    return None, '; '.join(whys[-3:]) or 'no region pose'


def retreat(env, arm):
    """Open giver backs its fingers straight out (8 / 5 cm along -link_6 z), then the inhand retreat (up and away)."""
    for back in (.08, .05):
        goal = IS.link6_T(env, arm)
        goal[:3, 3] -= goal[:3, 2] * back
        ok, _ = IS.move_link(env, arm, goal, actions=48, grip=1.0, n_way=6, tol=(0.01, 0.3))
        if ok:
            break
    IS.retreat(env, arm)


def handoff_solver(env, giver, deadline=None):
    """Scripted handoff: plan a presentation pose with a receiver grasp (``plan_handoff``), giver moves there, then the
    receiver takes it (``exchange``, the planned grasp first). -> (ok, why)."""
    plan, why = plan_handoff(env, giver, deadline=deadline)
    if plan is None:
        return False, 'plan: ' + why
    goal, hint = plan
    ok, why = move_link(env, giver, goal, held=True, actions=128)
    if not ok:
        return False, 'transport: ' + why
    return exchange(env, giver, deadline=deadline, tries=2, hint=hint, max_scan=36)


def _labels(scene, obs, prev, cfg, n=3):
    labels = np.full(len(scene.names), R) if prev is None else prev
    for _ in range(n):
        labels = classify(obs, labels, cfg)
    return labels


def build_handoff_start(env, seed, stage, *, rng=None, tries=16, budget_s=HANDOFF_BUDGET_S):
    """``handoff``: giver holds the target lifted 7-12 cm above its reset spot (the pick goal). ``carry`` in a
    handoff scene: source 1 = that start + the scripted handoff solver; fallback (source 3) = the receiver holds it
    alone in the handoff region (weightless teleport). Both carry sources must also have a verified direct carry
    path from the receiver to the box (as direct-reach carry starts)."""
    rng = rng or np.random.default_rng(seed)
    cfg = StageConfig()
    t0 = time.process_time()
    if stage == 'carry':
        budget_s = CARRY_BUDGET_S
    deadline = t0 + budget_s
    logs = []
    env.forget_arm_state()
    try:
        env.reset(seed=seed, randomize=True)
        env.evaluate_task()
    except (RuntimeError, ValueError) as ex:
        return None, [dict(why=f'reset invalid: {ex}', reset_invalid=True)]
    m, d, ev = env.model, env.data, env._task_evaluator
    scene = ev.scene
    rest = scene.heights(d)
    B = box_arm(env)
    obs = scene.observe(d, rest)
    giver = int(obs['near_arm'][0])
    base = dict(seed=int(seed), target=0, giver=giver, receiver=B, target_label=scene.label,
                category=getattr(env._last_randomization, 'metadata', {}).get('target_category'))
    if giver == B:
        return None, [dict(base, why='direct-reach scene: no handoff', skipped=True)]
    snap = IS.snapshot(env)
    for attempt in range(tries + 1):  # the last one: chained from a pick start
        if time.process_time() > deadline:
            logs.append(dict(base, attempt=attempt, why='time budget'))
            break
        _restore(env, snap)
        if attempt < tries:
            info = dict(base, attempt=attempt, source='sampled giver grasp, lifted')
            ok, gi = lift_hold(env, scene, giver, attempt, rng)
        else:
            info = dict(base, attempt=attempt, source='chained pick start + scripted lift')
            ok, gi = pick_chain(env, seed, giver, rng)
            m, d, ev = env.model, env.data, env._task_evaluator  # reset() may rebuild the model
            scene = ev.scene
        info.update(gi)
        if not ok:
            logs.append(info)
            continue
        arm = giver
        if stage == 'carry':
            arm = B
            chained = time.process_time() < t0 + CARRY_CHAIN_S
            ok, why = handoff_solver(env, giver, deadline=t0 + CARRY_CHAIN_S) if chained else (False, 'chain budget used')
            info['source'] = 'chained handoff solver'
            if ok:
                o = scene.observe(d, rest)
                ok = bool(o['f_recv'][0] == 2 and o['f_give'][0] == 0 and not o['env_touch'][0]
                          and o['lift'][0] >= cfg.lift_m)
                why = '' if ok else 'handoff ended without the receiver holding it alone in the air'
            if not ok:
                info['chain_failure'] = why
                ok, why = _receiver_fallback(env, scene, B, rest, rng)
                info['source'] = 'sampled receiver grasp in the handoff region (grasp source 3)'
            if ok:
                path, why = carry_plan(env, B)
                info['direct_path_verified'] = path is not None
                if path is None:
                    ok = False
                    info['coverage_exclusion'] = True
            if ok:
                z0 = float(d.qpos[scene.qpos_adr[0] + 2])
                a = idle_action(env)
                for _ in range(32):
                    env.step(a, render_obs=False)
                creep = z0 - float(d.qpos[scene.qpos_adr[0] + 2])
                info['creep_mm'] = float(creep * 1000)
                if creep > .0015:
                    ok, why = False, 'held target creeps >1.5 mm/chunk'
            if not ok:
                info['why'] = why
                logs.append(info)
                continue
        obs = scene.observe(d, rest)
        prev = None if stage == 'handoff' else np.where(np.arange(len(scene.names)) == 0, H, R)
        labels = _labels(scene, obs, prev, cfg)
        if labels[0] != STAGES.index(stage):
            logs.append({**info, 'why': f'wrong start label {STAGES[labels[0]] if labels[0] < len(STAGES) else labels[0]}'})
            continue
        if np.any(obs['off_table']) or obs['box_moved'] > cfg.box_moved_m or obs['box_tilt'] > cfg.box_tilt_rad:
            logs.append({**info, 'why': 'setup moved box or lost an object'})
            continue
        state = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
        mujoco.mj_getState(m, d, state, STATE_SPEC)
        info['build_cpu_s'] = round(time.process_time() - t0, 2)
        row = dict(seed=int(seed), stage=stage, target=0, arm=int(arm), state=state, rest_height=rest, labels=labels,
                   info=info)
        return row, logs + [{**info, 'ok': True}]
    return None, logs


def pick_chain(env, seed, giver, rng):
    """Grasp source 1 for a handoff start: a pick start of this seed (the picking arm open around the target) plus the
    pick solver's motion: close, lift 8 cm (collision-checked with the held target). -> (ok, info)."""
    row, log = build_start(env, seed, 'pick', rng=rng)
    env._task_evaluator._last_sim_time = None
    if row is None:
        return False, dict(why='no pick start: ' + str(log[-1].get('why', '') if log else ''))
    if int(row['arm']) != giver:
        return False, dict(why='pick start uses the box arm')
    a = idle_action(env)
    a[giver * 7 + 6] = 0.
    for _ in range(32):
        env.step(a, render_obs=False)
    ids = arm_ids(env.model, ARMS[giver])
    for up in (.08, .06):
        goal = T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])
        goal[2, 3] += up
        ok, why = move_link(env, giver, goal, held=True, grip=0., actions=96)
        if ok:
            break
    if not ok:
        return False, dict(why='lift: ' + why)
    o = env._task_evaluator.scene.contact_obs(env.data)
    return bool(o['grasp_arm'][0] == giver), dict(lift_m=up, why='' if o['grasp_arm'][0] == giver else 'not held after lift')


def _receiver_fallback(env, scene, B, rest, rng):
    """Receiver alone holds the target in the handoff region (giver sent home), grasp source 3."""
    view = TargetView(scene)
    d = env.data
    sg = -1.0 if B == 0 else 1.0  # giver side
    com = np.array([0.55, sg * 0.03, 0.95])
    plans = lambda: IS.single_plans(env, view, B, side_from=-sg)  # noqa: E731
    IS.teleport_arm(env, 1 - B, env.get_init_q()[(1 - B) * 7:(1 - B) * 7 + 6])
    ev = env._task_evaluator
    ev.allow_rewind = True  # hold_at restores snapshots between grasp tries
    try:
        ok, info = IS.hold_at(env, view, B, com, plans, rng, deadline=time.monotonic() + 3.0)
    finally:
        ev.allow_rewind = False
        ev._last_sim_time = None
    return ok, info.get('why', '')
