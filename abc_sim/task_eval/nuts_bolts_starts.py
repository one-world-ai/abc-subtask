"""Fixed-seed native-pad starts. Invalid physics never becomes a successful start.

Handoff amendment (2026-11-14): ``handoff`` starts exist only for cross-side scenes (the target's rest y on the other
side from its bin): the giver (the arm that is not on the bin's side) holds the part lifted 6-12 cm where it lay
(= the cross-side pick goal), high on the shank (``HIGH_PAD_Z``). Cross-side ``carry`` starts: the post-handoff state
synthesized (``receiver_hold``: the receiver's side grasp in the middle, giver at home); cross-side ``drop`` starts:
the cross-side carry solver's end state synthesized (``receiver_hold_over_bin``). Chaining the scripted ``handoff_solver`` costs
16-19 CPU-s per start here, over the 10 s budget. Same-side starts are built exactly as before.
"""
from __future__ import annotations
from contextlib import contextmanager
import time
import mujoco
import numpy as np
from abc_sim.task_eval.nuts_bolts_stages import R,STAGES,StageConfig,classify,PAD
from abc_sim.task_eval.stagekit import (ARMS,GRIP_MAX,STATE_SPEC,T,actuator_limits,arm_ids,hold_object,ik_link6,inv,place_gripper,set_object_pose,grasp_from_state)
SEED_BASE=950_000_000

class NumericalFault(RuntimeError):pass

def warnings(data):return np.array([w.number for w in data.warning],int)

def numerical_check(data,before_time,before_warnings):
    if not np.isfinite(data.time) or data.time<before_time-1e-10 or np.any(warnings(data)>before_warnings) or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise NumericalFault('MuJoCo numerical warning/time reset/nonfinite state')

@contextmanager
def checked_physics(env):
    original=env.step
    def checked(action,**kwargs):
        t=float(env.data.time);w=warnings(env.data)
        if w.any():raise NumericalFault('MuJoCo state already has warnings')
        result=original(action,**kwargs)
        numerical_check(env.data,t,w)
        return result
    env.step=checked
    try:yield
    finally:env.step=original

def idle_action(env):
    a=np.zeros(14,np.float32)
    for i,arm in enumerate(ARMS):
        a[i*7:i*7+6]=env.data.qpos[arm_ids(env.model,arm).qadr]
        a[i*7+6]=env.data.ctrl[env._ctrl_indices[i*7+6]]/.0475
    return a

def native_grasp(data,scene,k,yaw,arm,*,pad_z=.1297):
    # +x in the grasp-site frame maps to -y in link6. Closed pad boxes are centred at link6 y=-.044.
    r=np.array([[np.cos(yaw),np.sin(yaw),0],[np.sin(yaw),-np.cos(yaw),0],[0,0,-1.]])
    pose=np.eye(4);pose[:3,:3]=r
    pose[:3,3]=data.subtree_com[scene.bodies[k]]-r@np.array([0,-.044,pad_z])
    return dict(arm=int(arm),T_obj_link6=inv(scene.object_T(data,k))@pose,fingers=np.array([GRIP_MAX,GRIP_MAX]),yaw=float(yaw))

def part_yaw(env,scene,k):
    if scene.assigned[k]==0:return 0.
    axis=env.data.xmat[scene.bodies[k]].reshape(3,3)[:,2]
    return float(np.arctan2(axis[1],axis[0])+np.pi/2)

def move_link(env,arm,target,*,actions=64,grip=0.):
    ids=arm_ids(env.model,ARMS[arm]);m,d=env.model,env.data
    first=T(d.xpos[ids.link6],d.xquat[ids.link6]);qa=np.empty(4);qb=np.empty(4)
    mujoco.mju_mat2Quat(qa,first[:3,:3].ravel());mujoco.mju_mat2Quat(qb,target[:3,:3].ravel())
    if qa@qb<0:qb=-qb
    planning=mujoco.MjData(m);planning.qpos[:]=d.qpos;qs=[]
    for f in np.linspace(0,1,9)[1:]:
        q=(1-f)*qa+f*qb;q/=np.linalg.norm(q);r=np.empty(9);mujoco.mju_quat2Mat(r,q)
        pose=np.eye(4);pose[:3,:3]=r.reshape(3,3);pose[:3,3]=(1-f)*first[:3,3]+f*target[:3,3]
        sol,pe,re=ik_link6(m,planning,ARMS[arm],pose,limits=actuator_limits(m,ARMS[arm]),seeds=3)
        if sol is None or pe>.008 or re>.2:return False,f'IK path unreachable (position {pe:.4f}, rotation {re:.3f})'
        qs.append(sol);planning.qpos[ids.qadr]=sol
    base=idle_action(env)
    for q in qs:
        action=base.copy();action[arm*7:arm*7+6]=q;action[arm*7+6]=grip
        for _ in range(max(1,actions//8)):env.step(action,render_obs=False)
    return True,''

def carry_to_opening(env,scene,k,arm):
    m,d=env.model,env.data;b=int(scene.assigned[k]);bid=scene.bin_bodies[b];a=scene.bin_adrs[b]
    grasp=grasp_from_state(m,d,scene,k,arm);obj=scene.object_T(d,k);com=d.subtree_com[scene.bodies[k]].copy()
    r=d.xmat[bid].reshape(3,3);rim=d.qpos[a:a+3]+r@np.array([0,0,scene.rim[b]])
    # Raise above wall by measured full target extent, then carry; retain cross-table candidates.
    extent=max(m.geom_rbound[g] for g in scene.collision_geoms[k])
    high=max(com[2],rim[2]+extent+.02)
    for center in (np.array([com[0],com[1],high]),np.array([rim[0],rim[1],high])):
        desired=obj.copy();desired[:3,3]+=center-com
        ok,why=move_link(env,arm,desired@grasp['T_obj_link6'],actions=96)
        if not ok:return False,why
        if not scene.contact_obs(d)['grasped'][k]:return False,'part slipped during carry'
    action=idle_action(env)
    for _ in range(32):env.step(action,render_obs=False)
    obs=scene.observe(d,env._task_evaluator.tracker.rest_height)
    return bool(obs['over'][k] and obs['grasped'][k]),'carry end not held over opening'

# ------------------------------------------------------------------------------------------------- handoff (cross-side)
# Measured (probes in ../simulation/outputs/subtask_rewards/nuts_bolts/handoff/): bolts stand on their heads, so a
# top-down grasp holds one upright between finger plates 6.4 cm wide (link_6 y -.050...014) and 8 cm long (z .067-.147).
# Nothing sticks out of a grasp centred on the CoM. A grasp high on the shank (CoM at link_6 z >= .150, i.e. at or
# below the fingertips at z .147) leaves the head and lower shank (2-3 cm) below the fingertips; the receiver takes that from
# the side (link_6 z horizontal, its plates pointing down, pad strip 3 mm below the giver's fingertips), the bolt hangs
# along its axis and stays held after the giver opens (sag ~1 mm, no MuJoCo warning). A horizontal presentation (the
# head stub held top-down) pivots out of the receiver's fingers within 8 actions; flipping the giver's wrist is
# unreachable. For a low release the receiver then rolls 180 deg about its approach axis (``roll_receiver``): the bolt
# hangs head-up and its shank sticks 0.6-2 cm below the receiver's fingers.
MIDS=((.40,.05,.90),(.45,0.,.90),(.42,.03,.88),(.40,.08,.92))  # y mirrored for a right-arm giver
YAW_OFFSETS=np.deg2rad([0,180,-90,90,45,-45,135,-135,30,-30])
GAP=.003  # receiver pad strip's top edge this far below the giver's fingertips
MIN_STICK_OUT=.016  # part below the giver's fingertips needed for the receiver's strip (1.2 cm) + GAP
HIGH_PAD_Z=(.152,.160)  # handoff starts: the giver's grasp height (CoM in link_6 z), a top-down pick high on the bolt

def Rz(a):return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])

def reach_along(data,scene,k,u):
    """How far part k's collision primitives extend from its CoM along the unit vector u (m)."""
    com=data.subtree_com[scene.bodies[k]];best=-np.inf
    for g in scene.collision_geoms[k]:
        r=data.geom_xmat[g].reshape(3,3);ul=r.T@u;c=(data.geom_xpos[g]-com)@u;sz=scene.model.geom_size[g];typ=int(scene.model.geom_type[g])
        if typ==int(mujoco.mjtGeom.mjGEOM_BOX):e=np.abs(ul)@sz
        elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):e=sz[0]*np.sqrt(max(1-ul[2]**2,0))+sz[1]*abs(ul[2])
        else:e=scene.model.geom_rbound[g]
        best=max(best,c+e)
    return best

def fingertip(env,arm):
    l=arm_ids(env.model,ARMS[arm]).link6;return env.data.xpos[l]+env.data.xmat[l].reshape(3,3)@np.array([0,-.044,.147])

def stick_out(env,scene,k,giver):
    """How far part k reaches below the giver's fingertips (m)."""
    d=env.data;low=d.subtree_com[scene.bodies[k]][2]-reach_along(d,scene,k,np.array([0,0,-1.]))
    return float(fingertip(env,giver)[2]-low)

def present_plans(env,scene,k,giver,mids=MIDS):
    """Giver link_6 goals that put the part's CoM at a middle point, turned about the vertical (an upright bolt is
    symmetric about it) -> [(mid, T6)] the giver's IK reaches."""
    d,m=env.data,env.model;l=arm_ids(m,ARMS[giver]).link6;R6=d.xmat[l].reshape(3,3);p6=d.xpos[l]
    c6=R6.T@(d.subtree_com[scene.bodies[k]]-p6);out=[];lim=actuator_limits(m,ARMS[giver])
    for mid in mids:
        mid=np.array([mid[0],mid[1] if giver==0 else -mid[1],mid[2]])
        for a in YAW_OFFSETS:
            Rn=Rz(a)@R6;goal=np.eye(4);goal[:3,:3]=Rn;goal[:3,3]=mid-Rn@c6
            q,pe,re=ik_link6(m,d,ARMS[giver],goal,limits=lim,seeds=2)
            if q is not None and pe<3e-3 and re<.05:out.append((mid,goal));break
    return out

def receiver_plans(env,scene,k,giver,limit=2):
    """Receiver side grasps under the giver's fingertips: link_6 z horizontal (12 approach directions), plates
    pointing down, pad strip centre on the part's axis 3 mm + 6 mm below the fingertips -> [(T6, q, q_pregrasp)]
    reachable with an open gripper touching nothing, also along the approach (``approach_clear``)."""
    from abc_sim.task_eval.inhand_starts import check_receiver
    d=env.data;com=d.subtree_com[scene.bodies[k]];axis=d.xmat[scene.bodies[k]].reshape(3,3)[:,2]
    if abs(axis[2])<.5:return []
    z=fingertip(env,giver)[2]-GAP-.006;P=com+axis*(z-com[2])/axis[2];out=[]
    side=np.array([0,1. if giver==0 else -1.,0])  # from the receiver's side towards the giver's
    turns=sorted(np.deg2rad(np.arange(0,360,30)),key=lambda t:-np.array([np.cos(t),np.sin(t),0])@side)
    for turn in turns:
        zr=np.array([np.cos(turn),np.sin(turn),0.]);yr=np.array([0,0,-1.]);Rr=np.stack([np.cross(yr,zr),yr,zr],1)
        for depth in (.128,.135):
            T6=np.eye(4);T6[:3,:3]=Rr;T6[:3,3]=P-Rr@np.array([0,-.044,depth])
            got=check_receiver(env,scene,1-giver,T6)
            qpre=None if got is None else approach_clear(env,scene,1-giver,*got)
            if qpre is not None:
                out.append((*got,qpre))
                if len(out)>=limit:return out
                break
    return out

def approach_clear(env,scene,arm,T6,q,backs=(.03,.055,.08,.12,.16)):
    """The straight approach to T6 along its link_6 z is collision-free: 3 to 16 cm back (IK seeded from q, walking
    back) the open gripper touches nothing but parts (kinematic check on a scratch copy of the state) -> the joints
    16 cm back (the staging pose the solver moves to in joint space before the straight approach), or None."""
    from abc_sim.task_eval.inhand_starts import arm_touches,teleport_arm,snapshot,restore
    m=env.model;ids=arm_ids(m,ARMS[arm]);planning=mujoco.MjData(m);planning.qpos[:]=env.data.qpos;planning.qpos[ids.qadr]=q
    lim=actuator_limits(m,ARMS[arm]);snap=snapshot(env);first=None
    try:
        for back in backs:
            pre=T6.copy();pre[:3,3]-=T6[:3,2]*back
            qq,pe,re=ik_link6(m,planning,ARMS[arm],pre,limits=lim,seeds=1)
            if qq is None or pe>.005 or re>.1:return None
            planning.qpos[ids.qadr]=qq;teleport_arm(env,arm,qq)
            if arm_touches(env,scene,arm,ignore_item=True):return None
            first=qq
        return first
    finally:restore(env,snap)

class _Single:
    """Part k seen as the single item inhand_starts.monitored_release expects (bodies[0], dof_adr[0], contact_obs)."""
    def __init__(self,scene,k):self.scene,self.k=scene,k;self.bodies=[scene.bodies[k]];self.dof_adr=[scene.dof_adr[k]]
    def contact_obs(self,data):
        o=self.scene.contact_obs(data);return {key:v[self.k:self.k+1] for key,v in o.items()}

def present(env,scene,k,giver,goal):
    """Giver lifts the part to the goal height, then moves it to the goal pose (physical, other arm keeps its target)."""
    from abc_sim.task_eval.inhand_starts import move_link as move
    d=env.data;l=arm_ids(env.model,ARMS[giver]).link6
    up=T(d.xpos[l],d.xquat[l]);up[2,3]=max(up[2,3],goal[2,3])
    for T6,n in ((up,32),(goal,96)):
        ok,why=move(env,giver,T6,actions=n,grip=0.)
        if not ok:return False,'present: '+why
    return True,''

def handoff_solver(env,scene,k,giver,*,release=True,teleport=False,max_plans=3):
    """Scripted handoff. The giver presents the upright part in the middle; the receiver approaches horizontally to a
    side grasp under the giver's fingertips and closes; only with both receiver fingers on it the giver opens
    (monitored), lifts and backs off. ``release=False`` stops once both hold it (both-hold hack); ``teleport`` (start
    fallback only) puts the receiver at its grasp. -> (ok, why)."""
    from abc_sim.task_eval.inhand_starts import (action_with,approach_pose,current_action,monitored_release,move_joints,
                                                 move_link as move,retreat,run,teleport_arm)
    B=1-giver;whys=[]
    if stick_out(env,scene,k,giver)<MIN_STICK_OUT:
        return False,f'part sticks out {100*stick_out(env,scene,k,giver):.1f} cm below the giver fingertips (< 1.6 cm: grasp too low for a handoff)'
    plans=present_plans(env,scene,k,giver)
    if not plans:return False,'giver cannot reach a middle presentation'
    for mid,goal in plans[:max_plans]:
        ok,why=present(env,scene,k,giver,goal)
        if not ok:whys.append(why);continue
        run(env,current_action(env),8)
        for TB,qB,qpre in receiver_plans(env,scene,k,giver):
            if teleport:
                teleport_arm(env,B,qB);run(env,action_with(env,{B:qB},{B:1.}),4);qpre=None
            else:
                move_joints(env,B,qpre,actions=64,grip=1.)
                ok,why=move(env,B,TB,actions=128,grip=1.,n_way=8)
                if not ok:whys.append('receiver: '+why);move_joints(env,B,qpre,actions=32,grip=1.);continue
            run(env,action_with(env,None,{B:0.}),24)
            f=scene.contact_obs(env.data)['hits'][k].reshape(2,2).sum(1)
            if f[B]==2 and f[giver]==2:
                if not release:return True,'both hold'
                if monitored_release(env,_Single(scene,k),giver):
                    retreat(env,giver);run(env,current_action(env),16);return True,''
                whys.append('receiver could not hold it alone')
            else:whys.append(f'receiver closed with {f[B]} fingers, giver {f[giver]}')
            run(env,action_with(env,None,{B:1.}),12)
            if qpre is not None:move_joints(env,B,qpre,actions=32,grip=1.)
        else:
            if not whys or not whys[-1].startswith('receiver'):whys.append('no receiver grasp under the giver')
    return False,'; '.join(whys[-3:]) or 'no handoff plan'

def receiver_hold(env,scene,k,recv,rng,*,mids=MIDS):
    """Grasp source 3 for cross-side carry: the state the handoff solver leaves, synthesized -- the upright part in the
    middle, held alone by the receiver's side grasp (plates down, pad strip centre 1.2-2.0 cm above the part's lowest
    point, as after a handoff under the giver's fingertips), the giver at home. -> (ok, info)."""
    from abc_sim.task_eval.inhand_starts import reachable
    d=env.data;adr=scene.qpos_adr[k];whys=[]
    for mid in rng.permutation(np.array(mids))[:2]:
        mid=np.array([mid[0],mid[1] if recv==1 else -mid[1],mid[2]+rng.uniform(-.02,.04)])
        off=d.qpos[adr:adr+3]-d.subtree_com[scene.bodies[k]];set_object_pose(env,scene,k,mid+off,d.qpos[adr+3:adr+7].copy())
        com=d.subtree_com[scene.bodies[k]].copy();axis=d.xmat[scene.bodies[k]].reshape(3,3)[:,2]
        if abs(axis[2])<.5:return False,dict(why='part not upright')
        low=com[2]-reach_along(d,scene,k,np.array([0,0,-1.]));z=low+rng.uniform(.012,.020);P=com+axis*(z-com[2])/axis[2]
        side=np.array([0,-1. if recv==0 else 1.,0])
        for turn in sorted(np.deg2rad(np.arange(0,360,30)),key=lambda t:-np.array([np.cos(t),np.sin(t),0])@side):
            zr=np.array([np.cos(turn),np.sin(turn),0.]);yr=np.array([0,0,-1.]);Rr=np.stack([np.cross(yr,zr),yr,zr],1)
            T6=np.eye(4);T6[:3,:3]=Rr;T6[:3,3]=P-Rr@np.array([0,-.044,.13])
            Tt,q=reachable(env,recv,T6)
            if Tt is None:continue
            grasp=dict(arm=recv,T_obj_link6=inv(scene.object_T(d,k))@Tt,fingers=np.array([GRIP_MAX,GRIP_MAX]))
            ok,info=hold_object(env,scene,k,grasp,rng=rng)
            if ok:return True,dict(info,turn=round(float(np.rad2deg(turn))),mid=mid.round(3).tolist())
            whys.append(info.get('why'))
            set_object_pose(env,scene,k,mid+off,d.qpos[adr+3:adr+7].copy())
    return False,dict(why='; '.join(str(w) for w in whys[-3:]) or 'no receiver side grasp')

def carry_turned(env,scene,k,arm):
    """Cross-side carry (receiver's side grasp): like ``carry_to_opening`` but the hand may turn about the vertical
    through the part (an upright bolt is symmetric about it) -- the side grasp cannot keep its heading all the way to
    the bin (carry G4 round 0: 0/20 planned). Raise in place, then the first turn (YAW_OFFSETS) whose goal over the
    opening the IK reaches. -> (ok, why)."""
    m,d=env.model,env.data;b=int(scene.assigned[k]);bid=scene.bin_bodies[b];a=scene.bin_adrs[b]
    l=arm_ids(m,ARMS[arm]).link6;R6=d.xmat[l].reshape(3,3).copy();p6=d.xpos[l].copy()
    com=d.subtree_com[scene.bodies[k]].copy();c6=R6.T@(com-p6)
    r=d.xmat[bid].reshape(3,3);rim=d.qpos[a:a+3]+r@np.array([0,0,scene.rim[b]])
    extent=max(m.geom_rbound[g] for g in scene.collision_geoms[k]);high=max(com[2],rim[2]+extent+.02)
    up=np.eye(4);up[:3,:3]=R6;up[:3,3]=p6+np.array([0,0,high-com[2]])
    ok,why=move_link(env,arm,up,actions=64)
    if not ok:return False,'raise: '+why
    lim=actuator_limits(m,ARMS[arm]);target=np.array([rim[0],rim[1],high]);why='no turn reaches the opening'
    for alpha in YAW_OFFSETS:
        Rn=Rz(alpha)@R6;goal=np.eye(4);goal[:3,:3]=Rn;goal[:3,3]=target-Rn@c6
        q,pe,re=ik_link6(m,d,ARMS[arm],goal,limits=lim,seeds=2)
        if q is None or pe>.005 or re>.1:continue
        ok,why=move_link(env,arm,goal,actions=128)
        if ok:break
    if not ok:return False,why
    if not scene.contact_obs(d)['grasped'][k]:return False,'part slipped during carry'
    action=idle_action(env)
    for _ in range(32):env.step(action,render_obs=False)
    obs=scene.observe(d,env._task_evaluator.tracker.rest_height)
    return bool(obs['over'][k] and obs['grasped'][k]),'carry end not held over opening'

def receiver_hold_over_bin(env,scene,k,recv,rng):
    """Cross-side drop source (G3 repair 3): the state the cross-side carry solver ends in, synthesized -- the bolt
    head-up (the receiver rolled after the handoff) over the opening, its lowest point 1-4 cm above the rim, held by the
    receiver's side grasp with plates up on the head end (pad strip centre 1.2-2.0 cm below the top). -> (ok, info)."""
    from abc_sim.task_eval.inhand_starts import reachable
    from abc_sim.task_eval.stagekit import axis_angle_quat,qmul
    d=env.data;adr=scene.qpos_adr[k];b=int(scene.assigned[k]);a=scene.bin_adrs[b]
    rim=d.qpos[a:a+3]+d.xmat[scene.bin_bodies[b]].reshape(3,3)@np.array([0,0,scene.rim[b]])
    quat=qmul(axis_angle_quat(np.array([1.,0,0]),np.pi),d.qpos[adr+3:adr+7].copy())
    set_object_pose(env,scene,k,d.qpos[adr:adr+3].copy(),quat)
    com=d.subtree_com[scene.bodies[k]].copy();below=reach_along(d,scene,k,np.array([0,0,-1.]))
    off=d.qpos[adr:adr+3]-com;target=np.array([rim[0],rim[1],rim[2]+below+rng.uniform(.01,.04)])
    set_object_pose(env,scene,k,target+off,quat)
    com=d.subtree_com[scene.bodies[k]].copy();axis=d.xmat[scene.bodies[k]].reshape(3,3)[:,2]
    top=com[2]+reach_along(d,scene,k,np.array([0,0,1.]));z=top-rng.uniform(.012,.020);P=com+axis*(z-com[2])/axis[2]
    side=np.array([0,-1. if recv==0 else 1.,0]);whys=[]
    for turn in sorted(np.deg2rad(np.arange(0,360,30)),key=lambda t:-np.array([np.cos(t),np.sin(t),0])@side):
        zr=np.array([np.cos(turn),np.sin(turn),0.]);yr=np.array([0,0,1.]);Rr=np.stack([np.cross(yr,zr),yr,zr],1)
        T6=np.eye(4);T6[:3,:3]=Rr;T6[:3,3]=P-Rr@np.array([0,-.044,.13])
        Tt,q=reachable(env,recv,T6)
        if Tt is None:continue
        grasp=dict(arm=recv,T_obj_link6=inv(scene.object_T(d,k))@Tt,fingers=np.array([GRIP_MAX,GRIP_MAX]))
        ok,info=hold_object(env,scene,k,grasp,rng=rng)
        if ok:return True,dict(info,turn=round(float(np.rad2deg(turn))))
        whys.append(info.get('why'));set_object_pose(env,scene,k,target+off,quat)
        if len(whys)>=3:break
    return False,dict(why='; '.join(str(w) for w in whys[-3:]) or 'no receiver side grasp over the opening')

def plates_down(env,arm):
    """True if ``arm``'s link_6 y points down (the side grasp the handoff leaves): its plates hang below the part."""
    return float(env.data.xmat[arm_ids(env.model,ARMS[arm]).link6].reshape(3,3)[2,1])<-.5

def roll_receiver(env,scene,k,arm,*,depth=.13,actions=128):
    """Roll ``arm``'s link_6 180 deg about its own z around the pad strip (joint space): the held bolt turns head-up
    and its shank hangs below the fingers, so it can be lowered onto a floor. -> (ok, why)."""
    from abc_sim.task_eval.inhand_starts import current_action,move_joints,run
    d,m=env.data,env.model;l=arm_ids(m,ARMS[arm]).link6;R=d.xmat[l].reshape(3,3).copy();p=d.xpos[l].copy()
    c=np.array([0,-.044,depth]);Rp=R@np.diag([-1.,-1.,1.]);goal=np.eye(4);goal[:3,:3]=Rp;goal[:3,3]=p+R@c-Rp@c
    q,pe,re=ik_link6(m,d,ARMS[arm],goal,limits=actuator_limits(m,ARMS[arm]),seeds=6)
    if q is None or pe>.005 or re>.1:return False,'roll unreachable'
    move_joints(env,arm,q,actions=actions,grip=0.);run(env,current_action(env),16)
    if not scene.contact_obs(d)['grasped'][k]:return False,'part slipped during the roll'
    return True,''

def cross_side(env,scene,k,rest=None):
    rest=scene.heights(env.data) if rest is None else rest
    cross,recv=scene.handoff_roles(rest);return bool(cross[k]),int(recv[k])

def build_start(env,seed,stage,*,rng=None,tries=8):
    rng=rng or np.random.default_rng(seed);logs=[];cfg=StageConfig();began=time.process_time()  # CPU time: wall time on a shared node measures contention, not start cost
    for attempt in range(1 if stage=='reach' else tries):
        if attempt and time.process_time()-began>9.5:break
        try:
            env.forget_arm_state();env.reset(seed=seed,randomize=True);env.evaluate_task()
            ev=env._task_evaluator;scene=ev.scene;rest=scene.heights(env.data);obs=scene.observe(env.data,rest)
            k=ev.pick_target(obs);arm=int(obs['near_arm'][k]);original_arm=arm
            b=int(scene.assigned[k]);bin_y=env.data.qpos[scene.bin_adrs[b]+1]
            cross,recv=cross_side(env,scene,k,rest)
            info=dict(attempt=attempt,target=k,part=scene.names[k],arm=arm,original_arm=arm,
                      cross_table=bool(np.sign(obs['xy'][k,1])!=np.sign(bin_y)))
            if stage=='handoff' and not cross:
                return None,logs+[{**info,'why':'same-side scene: no handoff','skipped':True}]
            handoff_chain=cross and stage in ('handoff','carry','drop')
            if handoff_chain:arm=1-recv  # the giver holds it first
            if warnings(env.data).any():raise NumericalFault('warnings at reset')
            with checked_physics(env):
                if stage!='reach':
                    yaw=part_yaw(env,scene,k)+(0,np.pi,np.pi/4,3*np.pi/4,np.pi/2,3*np.pi/2,np.pi/8,9*np.pi/8)[attempt%8]
                    if stage=='pick':
                        grasp=native_grasp(env.data,scene,k,yaw,arm,pad_z=(.145,.148,.142)[attempt%3])
                        ok,detail=place_gripper(env,scene,k,grasp,rng=rng)
                        if ok:
                            act=idle_action(env);act[arm*7+6]=1.
                            for _ in range(10):env.step(act,render_obs=False)
                            o=scene.observe(env.data,rest)
                            ok=not o['touched'][k] and np.linalg.norm(o['xy'][k]-obs['xy'][k])<.01
                            if not ok:detail['why']='open fingers touch or push part'
                    elif handoff_chain and stage!='handoff':
                        # Cross-side carry/drop: chaining the scripted handoff (~460 actions) took 16-19 CPU-s per start on
                        # this node, over the 10 s start budget, so both use grasp source 3 (STATUS.md, G3).
                        ok,detail=False,dict(why='handoff chain over the 10 s start budget')
                    else:
                        adr=scene.qpos_adr[k];pos=env.data.qpos[adr:adr+3].copy();pos[2]+=rng.uniform(.06,.12)
                        set_object_pose(env,scene,k,pos,env.data.qpos[adr+3:adr+7].copy())
                        # Cross-side: the giver holds it high on the shank (a top-down pick of an upright bolt), the only
                        # grasp that leaves something below its fingertips for the receiver (see HIGH_PAD_Z).
                        grasp=native_grasp(env.data,scene,k,yaw,arm,**(dict(pad_z=rng.uniform(*HIGH_PAD_Z)) if handoff_chain else {}))
                        ok,detail=hold_object(env,scene,k,grasp,rng=rng)
                        if ok and stage=='drop':
                            ok,why=carry_to_opening(env,scene,k,arm)
                            detail['source']='physical carry chain'
                            if not ok:detail['why']=why
                    if stage in ('carry','drop'):
                        if stage=='carry' and handoff_chain:
                            # Cross-side carry: grasp source 3, the post-handoff state synthesized (receiver_hold). The
                            # chain (handoff start + handoff_solver, ~460 actions) took 16-19 CPU-s per start on this
                            # node, over the 10 s start budget, so it is not tried here (STATUS.md, G3).
                            env.forget_arm_state();env.reset(seed=seed,randomize=True);env.evaluate_task()
                            scene=env._task_evaluator.scene;arm=recv
                            ok,detail=receiver_hold(env,scene,k,recv,rng)
                            detail.update(source='sampled post-handoff receiver hold in the middle')
                        if stage=='drop' and handoff_chain:
                            # Cross-side drop G3 repair 3: the sampled lowered top-down hold (rounds 0-2: 65, 58, 57/100
                            # valid -- the upright bolt slides out of a top-down grasp) is replaced by the post-handoff
                            # grasp the receiver really carries it with (receiver_hold_over_bin).
                            env.forget_arm_state();env.reset(seed=seed,randomize=True);env.evaluate_task()
                            scene=env._task_evaluator.scene;arm=recv
                            ok,detail=receiver_hold_over_bin(env,scene,k,recv,rng)
                            detail.update(source='sampled post-handoff receiver hold over the opening')
                        elif stage=='drop' and not ok:
                            # Independent source 3 after a failed chain; a safely lowered held pose is approved.
                            env.forget_arm_state();env.reset(seed=seed,randomize=True);env.evaluate_task()
                            scene=env._task_evaluator.scene
                            adr=scene.qpos_adr[k];a=scene.bin_adrs[b];rot=env.data.xmat[scene.bin_bodies[b]].reshape(3,3)
                            local=scene.goal_centers[b]+np.array([0,0,.015])
                            pos=env.data.qpos[a:a+3]+rot@local;arm=0 if pos[1]>0 else 1
                            set_object_pose(env,scene,k,pos,env.data.qpos[adr+3:adr+7].copy())
                            grasp=native_grasp(env.data,scene,k,yaw,arm)
                            chain_failure=detail.get('why','initial held setup failed')
                            ok,detail=hold_object(env,scene,k,grasp,rng=rng)
                            detail.update(source='sampled safely lowered hold',chain_failure=chain_failure)
                    info.update({key:value for key,value in detail.items() if key!='q'});info['arm']=arm
                    if not ok:logs.append(info);continue
                o=scene.observe(env.data,rest);labels=classify(o,np.full(len(scene.names),R),cfg)
                labels=classify(o,labels,cfg)
                if labels[k]!=STAGES.index(stage):
                    logs.append({**info,'why':f'classification {int(labels[k])}, expected {stage}'});continue
                if np.any(o['bin_moved']>cfg.bin_moved_m) or np.any(o['bin_tilt']>cfg.bin_tilt_rad):
                    logs.append({**info,'why':'bin moved during setup'});continue
                if time.process_time()-began>10:
                    return None,logs+[{**info,'why':'valid setup exceeded 10-second start budget'}]
                state=np.empty(mujoco.mj_stateSize(env.model,STATE_SPEC));mujoco.mj_getState(env.model,env.data,state,STATE_SPEC)
                return dict(seed=int(seed),stage=stage,target=int(k),arm=arm if stage!='reach' else -1,state=state,
                            rest_height=rest,labels=labels,info=info),logs+[{**info,'ok':True}]
        except (NumericalFault,RuntimeError) as error:
            logs.append(dict(attempt=attempt,why=str(error),numerical=isinstance(error,NumericalFault)))
    return None,logs
