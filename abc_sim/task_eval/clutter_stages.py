"""CPU physics stages for the prompted object in grab_clutter (approved 2026-11-13; handoff amendment 2026-11-14).

Chain: reach -> pick -> [handoff] -> carry -> place -> done. The box sits at y = +-0.515 m; only the arm on the box's
side (the "box arm", left for +y) can carry to it (STATUS.md: 53/53 opposite-side direct carries IK-unreachable).
So the other arm is the handoff **giver** and the box arm the **receiver** in every scene, fixed from the reset box
position. A target the giver holds lifted is in ``handoff``; the task-independent ``handoff_stage`` decides the
exchange (region, received, received in the air); the box arm holding it alone is ``carry``. Direct-reach scenes
(the box arm picks it) never enter ``handoff``: their labels are unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import mujoco
import numpy as np
from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.grab_clutter import GrabClutterTargetInBoxEvaluator
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching, geoms_of, subtree
from abc_sim.task_eval.stagekit.tracker import StageTracker

STAGES=('reach','pick','handoff','carry','place','done')
R,P,H,C,PL,DONE=range(6)
# Pinch centre in link_6 (finger pads span y -0.047..-0.017, z 0.07..0.15; the grasp site is 3 cm beside them;
# inhand finding). The handoff "clear" distance is measured from it, as in inhand.
PAD=np.array([0.,-.031,.125])
FALLEN=len(STAGES);LABELS=STAGES+('fallen',)
STAGE_PROMPTS={s:f'clutter subtask: {v}' for s,v in dict(reach='approach the prompted target',pick='grasp and lift the prompted target',carry='carry the prompted target over the box',place='lower the prompted target into the box and release',handoff='hand the prompted target to the box-side arm').items()}

@dataclass(frozen=True)
class StageConfig:
    reach_m:float=.05
    reach_exit_m:float=.07
    lift_m:float=.03
    push_m:float=.03
    box_moved_m:float=.03
    box_tilt_rad:float=np.deg2rad(20)
    clearance_m:float=.01
    opening_exit_m:float=.01
    drop_clearance_m:float=.04
    at_rest_speed:float=.02
    at_rest_angular_speed:float=.2
    # Reach -> pick entry needs an undisturbed target: resting targets read 0.0 mm/s at all 14 reach-solver goals,
    # a gripper slowly shoving one read 14-15 mm/s (scripted_r2 reach_push), under the 20 mm/s place at-rest bar.
    undisturbed_speed:float=.005
    hold:int=5
    # Handoff (amendment 2026-11-14): the shared handoff_stage tests with their inhand defaults (middle region
    # x 0.45-0.70, |y| <= 0.10, CoM z 0.83-1.15; clear 6 cm from the surface; air/drop 3/4 cm), unchanged.
    handoff:hs.HandoffConfig=field(default_factory=hs.HandoffConfig)
    # Handoff progress adds -min(CoM distance to the region, transport_clip): the stage starts where pick ends.
    transport_clip:float=.50
    touch_margin_m:float=.002  # handoff observations only (touch_give / touch_recv), see ClutterScene.handoff_obs
    span:dict=field(default_factory=lambda:dict(reach=.30,pick=1.,handoff=1.75,carry=.50,place=.15))
    eps:dict=field(default_factory=lambda:dict(reach=.005,pick=.05,handoff=.02,carry=.005,place=.005))

def box_surface_points(model,body,step=.006):
    """Points on the collision box geoms of ``body``'s subtree, in the body frame (clutter targets have one box
    collision geom). Face grids every ``step`` m."""
    data=mujoco.MjData(model);mujoco.mj_forward(model,data);Rb,pb=data.xmat[body].reshape(3,3),data.xpos[body];out=[]
    for g in geoms_of(model,subtree(model,body)):
        if model.geom_contype[g]==0 and model.geom_conaffinity[g]==0:continue
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:raise ValueError('clutter target collision geom is not a box')
        h=model.geom_size[g];axes=[np.linspace(-h[i],h[i],max(2,int(np.ceil(2*h[i]/step))+1)) for i in range(3)]
        for i in range(3):
            j,k=[x for x in range(3) if x!=i];A,B=np.meshgrid(axes[j],axes[k],indexing='ij')
            for sgn in (-1,1):
                p=np.zeros((A.size,3));p[:,i]=sgn*h[i];p[:,j]=A.ravel();p[:,k]=B.ravel();out.append(p)
        pts=np.concatenate(out[-6:]);w=pts@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g];out[-6:]=[(w-pb)@Rb]
    if not out:raise ValueError('target has no collision geom')
    return np.concatenate(out)

class ClutterScene(ObjectScene):
    def __init__(self,model,cfg=None):
        target=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'target_object_joint')
        super().__init__(model,[('target_object',target)]+free_joints_matching(model,r'clutter_obj_(\d+)_joint'))
        self.cfg=cfg or StageConfig();self.geometry=GrabClutterTargetInBoxEvaluator(model=model,spec=None)
        self.box_adr=self.geometry._box_qpos_addr
        self.box_body=model.body('grab_clutter_target_box').id
        self.floor_geom=model.geom('grab_clutter_target_box_floor').id
        self.label='object'
        plane=model.geom('table_plane')
        self.table_bounds=np.array([plane.pos[0]-plane.size[0],plane.pos[0]+plane.size[0],plane.pos[1]-plane.size[1],plane.pos[1]+plane.size[1]])
        self.table_z=float(plane.pos[2])
        self.points=box_surface_points(model,self.bodies[0])
        self.arm_of=np.full(model.ngeom,-1,dtype=np.int64)  # geom -> 0 left arm, 1 right arm
        for a,root in enumerate(('left_arm','right_arm')):
            self.arm_of[geoms_of(model,subtree(model,model.body(root).id))]=a
        self.link6=[model.body(f'{arm}_link_6').id for arm in ('left','right')]
        collide=lambda g:model.geom_contype[g]!=0 or model.geom_conaffinity[g]!=0  # noqa: E731
        self.target_geoms=[g for g in range(model.ngeom) if self.owner[g,0]==0 and self.owner[g,1]==0 and collide(g)]
        self.finger_geoms=[[g for g in range(model.ngeom) if self.owner[g,0]==1 and self.owner[g,1]//2==a and collide(g)]
                           for a in (0,1)]
    def world_points(self,data):
        b=self.bodies[0];return self.points@data.xmat[b].reshape(3,3).T+data.xpos[b]
    def handoff_obs(self,data,rest,pos):
        """handoff_stage observation arrays (target row only; distractors get neutral values). Receiver = the box
        arm (left if the reset box is at +y), giver = the other arm."""
        n=len(self.names);B=0 if rest[-1,1]>0 else 1;A=1-B
        touch=np.zeros((n,2),bool);env=np.zeros(n,bool)
        for c in data.contact[:data.ncon]:
            for o,other in ((int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))):
                if self.owner[o,0]==0 and self.owner[o,1]==0 and not (self.owner[other,0]==0 and self.owner[other,1]==0):
                    arm=self.arm_of[other]
                    if arm>=0:touch[0,arm]=True
                    else:env[0]=True
        # A finger within touch_margin_m also counts as on it: a receiver's closed pads lose contact for single
        # physics steps while the item settles in the pinch (measured: one contact-free step, then 2 fingers again),
        # which must not read as "dropped".
        ft=np.empty(6)
        for a in (0,1):
            if not touch[0,a]:
                touch[0,a]=any(mujoco.mj_geomDistance(self.model,data,g,f,self.cfg.touch_margin_m,ft)<self.cfg.touch_margin_m
                               for g in self.target_geoms for f in self.finger_geoms[a])
        pts=self.world_points(data)
        centres=np.stack([data.xpos[l]+data.xmat[l].reshape(3,3)@PAD for l in self.link6])
        gap=np.full((n,2),np.inf);gap[0]=np.linalg.norm(pts[None]-centres[:,None],axis=-1).min(axis=1)
        return A,B,touch,env,gap
    def heights(self,data):
        return np.vstack((np.asarray(data.subtree_com[self.bodies]).reshape(-1,3),data.qpos[self.box_adr:self.box_adr+3]))
    def observe(self,data,rest):
        rest=np.asarray(rest).reshape(-1,3);pos=np.asarray(data.subtree_com[self.bodies]).reshape(-1,3)
        n=len(self.names);rbox=data.xmat[self.box_body].reshape(3,3)
        box=data.qpos[self.box_adr:self.box_adr+3];origin=data.qpos[self.qpos_adr[0]:self.qpos_adr[0]+3]
        local=(origin-box)@rbox
        rt=rbox.T@data.xmat[self.bodies[0]].reshape(3,3)
        low=float(local[2]+np.min(self.geometry._target_collision_points@rt[2]))
        floor=self.geometry._box_floor_z;rim=self.geometry._box_rim_z
        center,half=self.geometry._goal_local_center,self.geometry._goal_half_size
        delta=np.abs(local[:2]-center[:2])-half[:2]
        xy_gap=float(np.linalg.norm(np.maximum(delta,0)))
        gap=np.full(n,np.inf);gap[0]=np.hypot(xy_gap,max(rim+self.cfg.clearance_m-low,0))
        opening=np.zeros(n,bool);opening[0]=bool(np.all(delta<=0))
        opening_loose=np.zeros(n,bool);opening_loose[0]=bool(np.all(delta<=self.cfg.opening_exit_m))
        over=np.zeros(n,bool);over[0]=opening[0] and low>=rim+self.cfg.clearance_m
        in_box=np.zeros(n,bool);in_box[0]=bool(self.geometry.evaluate_qpos_batch(data.qpos[None]).success[0])
        supported=np.zeros(n,bool);resting=np.zeros(n,bool)
        for contact in data.contact:
            for obj,other in ((int(contact.geom1),int(contact.geom2)),(int(contact.geom2),int(contact.geom1))):
                kind,k=self.owner[obj];okind,_=self.owner[other]
                if kind==0 and okind!=1:resting[k]=True
                if kind==0 and other==self.floor_geom:supported[k]=True
        speed=np.array([np.linalg.norm(data.qvel[a:a+3]) for a in self.dof_adr])
        angular=np.array([np.linalg.norm(data.qvel[a+3:a+6]) for a in self.dof_adr])
        x0,x1,y0,y1=self.table_bounds
        off=(pos[:,0]<x0)|(pos[:,0]>x1)|(pos[:,1]<y0)|(pos[:,1]>y1)|(pos[:,2]<self.table_z-.03)
        obs=dict(xy=pos[:,:2],pos=pos,lift=pos[:,2]-rest[:n,2],moved_xy=np.linalg.norm(pos[:,:2]-rest[:n,:2],axis=1),
                    gap=gap,in_opening=opening,in_opening_loose=opening_loose,over=over,in_box=in_box,floor_support=supported,
                    resting_contact=resting,speed=speed,angular_speed=angular,off_table=off,
                    clearance=np.r_[low-floor,np.zeros(n-1)],lowest=np.r_[low,np.zeros(n-1)],
                    box_moved=float(np.linalg.norm(box[:2]-rest[-1,:2])),box_tilt=float(np.arccos(np.clip(rbox[2,2],-1,1))),
                    **self.contact_obs(data))
        A,B,touch,env,gap=self.handoff_obs(data,rest,pos)
        per_arm=obs['hits'].reshape(n,2,2).sum(axis=2)
        m_gap=np.full(n,np.inf);m_gap[0]=hs.region_gap(pos[:1],self.cfg.handoff.region_lo,self.cfg.handoff.region_hi)[0]
        obs.update(com=pos,f_give=per_arm[:,A],f_recv=per_arm[:,B],touch_give=touch[:,A],touch_recv=touch[:,B],
                   gap_give=gap[:,A],gap_recv=gap[:,B],env_touch=env,m_gap=m_gap,giver=A,receiver=B)
        return obs

def classify(obs,prev,cfg):
    near=(obs['dist']<=np.where(prev==P,cfg.reach_exit_m,cfg.reach_m))|obs['touched']
    at_rest=(obs['speed']<=cfg.at_rest_speed)&(obs['angular_speed']<=cfg.at_rest_angular_speed)
    # Entering pick from reach needs an undisturbed target (at rest, moved < push_m): a gripper shoving the target
    # is not a legal reach, and the reach goal must be a valid pick precondition. Later labels keep plain `near`.
    still=(obs['speed']<=cfg.undisturbed_speed)&(obs['angular_speed']<=cfg.at_rest_angular_speed)
    near&=(prev!=R)|(still&(obs['moved_xy']<cfg.push_m))
    out=np.full(prev.shape,R,dtype=np.int64);out[near]=P
    h=cfg.handoff;lifted=obs['lift']>=cfg.lift_m
    # Handoff: the giver (non-box arm) holds it lifted -- entering also needs it touching nothing but the robot,
    # so a pick that still drags clutter stays in pick -- or the handoff_stage exchange test holds.
    give=(obs['f_give']==2)&lifted&((prev>=H)|~obs['env_touch'])
    out[give|hs.handing(obs,prev==H,h)]=H
    # Carry: the box arm holds it alone, lifted. From handoff only through handoff_stage's goal (received in the
    # air, giver clear); direct-reach scenes enter from pick as before.
    direct=obs['grasped']&lifted&(obs['f_give']==0)&~obs['touch_give']
    out[np.where(prev==H,hs.received_in_air(obs,h),direct)]=C
    place=obs['over']|((prev==PL)&obs['in_opening_loose'])
    out[place&obs['touched']&~obs['touch_give']]=PL
    free=~obs['touched']&(~at_rest|~obs['resting_contact']);out[free]=prev[free]
    out[obs['in_box']&obs['floor_support']&~obs['touched']&at_rest]=DONE
    out[obs['off_table']]=FALLEN
    return out

def progress(stage,obs,cfg=None):
    cfg=cfg or StageConfig()
    if stage==R:return -obs['dist']
    if stage==P:return .25*obs['fingers']+.5*np.clip(obs['lift']/.03,0,1)*obs['grasped']
    if stage==H:return hs.progress(obs,cfg.handoff)-np.minimum(obs['m_gap'],cfg.transport_clip)
    if stage==C:return -np.minimum(obs['gap'],1.)
    if stage==PL:return -np.clip(obs['clearance'],0,.5)+.05*~obs['touched']
    raise ValueError(stage)

class ClutterTracker(StageTracker):
    def __init__(self,scene,cfg=None):super().__init__(scene,classify,cfg or scene.cfg,start_label=R)
