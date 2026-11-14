"""Approved per-part reach/pick/(handoff)/carry/drop stages, from real CPU simulator state.

Handoff amendment (user, 2026-11-14; ``subtask_rewards/nuts_bolts/stage_table.md``): a *cross-side* part (its rest
CoM on the other side of the table, by the sign of y, from its assigned bin) must be handed from the arm that picked
it (the giver) to the arm on the bin's side (the receiver) before carry. The handoff tests are the task-independent
``handoff_stage`` (unchanged); this scene supplies its observation arrays (``f_give``/``f_recv``/``touch_give``/
``touch_recv``/``gap_give``/``env_touch``/``com``/``lift``) per part. Same-side parts are classified exactly as before.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import mujoco
import numpy as np
from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.nuts_bolts import NutsBoltsSortingEvaluator
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching, geoms_of, subtree
from abc_sim.task_eval.stagekit.tracker import StageTracker

STAGES=("reach","pick","handoff","carry","drop","done")
R,P,H,C,D,DONE=range(6)
FALLEN=len(STAGES)
LABELS=STAGES+("fallen",)
STAGE_PROMPTS={
    "reach":"nuts bolts subtask: move a gripper to the closest unsorted part",
    "pick":"nuts bolts subtask: grasp the target part and lift it",
    "handoff":"nuts bolts subtask: hand the target part to the gripper on its bin's side",
    "carry":"nuts bolts subtask: carry the target part over its assigned bin",
    "drop":"nuts bolts subtask: release the target part into its assigned bin",
}

# Pinch centre in link_6 (as inhand): the YAM <arm>_grasp_site at (0, 0, 0.1347) sits ~3 cm beside the finger pads.
PAD=np.array([0.,-.031,.125])
# Handoff region: anywhere over the table (judgement call, stage table amendment): the pick goal of a cross-side
# part (giver holds it lifted where it lay) is the handoff precondition, so ``handing`` may not require the middle.
HANDOFF=hs.HandoffConfig(region_lo=(.25,-.50,.70),region_hi=(.90,.50,1.40))
# Middle box where both arms reach a top-down grasp side by side (scripted handoffs at x .45, y 0, z .85): shaping
# only (progress pulls the part there), never a stage test.
MIDDLE_LO=np.array([.40,-.05,.82]);MIDDLE_HI=np.array([.55,.05,.95])

@dataclass(frozen=True)
class StageConfig:
    reach_m:float=.05
    reach_exit_m:float=.07
    lift_m:float=.03
    push_m:float=.03
    fallen_m:float=.05
    opening_margin_m:float=.01
    opening_exit_m:float=.01
    bin_moved_m:float=.03
    bin_tilt_rad:float=np.deg2rad(20)
    at_rest_speed:float=.02
    at_rest_angular_speed:float=.2
    hold:int=5
    # Drop amendment (user, 2026-11-14): a release counts only with the part's lowest collision extent at most this
    # far above the bin floor. Measured: releases <2 mm gave +1 14/20 (3 numerical); 2-10 mm about half numerical.
    release_max_clearance_m:float=.005
    handoff:hs.HandoffConfig=HANDOFF
    middle_m:float=.30  # handoff shaping: -min(CoM distance to the middle box, middle_m)
    eps:dict=field(default_factory=lambda:dict(reach=.005,pick=.05,handoff=.02,carry=.005,drop=.005))
    span:dict=field(default_factory=lambda:dict(reach=.30,pick=1.,handoff=1.55,carry=.40,drop=.15))

class NutsBoltsScene(ObjectScene):
    def __init__(self,model,cfg=None):
        super().__init__(model,free_joints_matching(model,r"nut_(\d+)_joint")+free_joints_matching(model,r"bolt_(\d+)_joint"))
        self.cfg=cfg or StageConfig()
        self.geometry=NutsBoltsSortingEvaluator(model=model,spec=None)
        self.bin_names=("nuts_sorting_bin","bolts_sorting_bin")
        self.bin_bodies=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in self.bin_names]
        self.bin_adrs=np.array([self.geometry._nuts_bin_qpos_addr,self.geometry._bolts_bin_qpos_addr])
        self.bin_geoms=[set(geoms_of(model,subtree(model,b))) for b in self.bin_bodies]
        self.goal_centers=np.array([self.geometry._nuts_goal_local_center,self.geometry._bolts_goal_local_center])
        self.goal_sizes=np.array([self.geometry._nuts_goal_size,self.geometry._bolts_goal_size])
        self.assigned=np.array([0 if n.startswith("nut_") else 1 for n in self.names])
        self.active=np.ones(len(self.names),bool)
        # Collision walls' inner faces and top: measured after random scaling, in each bin's local frame.
        self.inner=[];self.rim=[];self.floor=[]
        for b in range(2):
            prefix="nuts" if b==0 else "bolts"
            floor=model.geom(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,prefix+"_bin_floor"))
            back=model.geom(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,prefix+"_bin_back_wall"))
            left=model.geom(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,prefix+"_bin_left_wall"))
            self.inner.append([back.pos[0]-back.size[0],left.pos[1]-left.size[1]])
            self.rim.append(back.pos[2]+back.size[2]);self.floor.append(floor.pos[2]+floor.size[2])
        self.inner=np.array(self.inner);self.rim=np.array(self.rim);self.floor=np.array(self.floor)
        self.collision_geoms=[np.array([g for g in geoms_of(model,subtree(model,body)) if model.geom_contype[g] or model.geom_conaffinity[g]],int) for body in self.bodies]
        # Handoff: geom -> arm (0 left, 1 right, -1 not an arm); link_6 bodies for the pinch centres.
        self.arm_of=np.full(model.ngeom,-1,np.int64)
        for a,root in enumerate(("left_arm","right_arm")):
            body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,root)
            if body<0:raise ValueError(f"{root} not found")
            self.arm_of[geoms_of(model,subtree(model,body))]=a
        self.link6=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f"{arm}_link_6") for arm in ("left","right")]

    def handoff_roles(self,rest):
        """(cross, receiver arm) per part from the episode's rest positions: cross = the part's rest y and its
        assigned bin's rest y have opposite signs; receiver = the arm on the bin's side (left arm at +y)."""
        rest=np.asarray(rest).reshape(-1,3);n=len(self.names)
        bin_y=rest[n:,1][self.assigned]
        return np.sign(rest[:n,1])!=np.sign(bin_y),np.where(bin_y>0,0,1)

    def surface_gap(self,data,k,point):
        """Distance from a world point to part k's collision primitives (boxes, cylinders, else bounding sphere)."""
        best=np.inf
        for g in self.collision_geoms[k]:
            local=data.geom_xmat[g].reshape(3,3).T@(point-data.geom_xpos[g]);sz=self.model.geom_size[g];typ=int(self.model.geom_type[g])
            if typ==int(mujoco.mjtGeom.mjGEOM_BOX):d=np.linalg.norm(np.maximum(np.abs(local)-sz,0))
            elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):d=np.hypot(max(np.hypot(local[0],local[1])-sz[0],0),max(abs(local[2])-sz[1],0))
            elif typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):d=max(np.linalg.norm(local)-sz[0],0)
            elif typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):d=max(np.linalg.norm(local-np.array([0,0,np.clip(local[2],-sz[1],sz[1])]))-sz[0],0)
            else:d=max(np.linalg.norm(local)-self.model.geom_rbound[g],0)
            best=min(best,d)
        return best

    def heights(self,data):
        return np.vstack((np.asarray(data.subtree_com[self.bodies]).reshape(-1,3),[data.qpos[a:a+3] for a in self.bin_adrs]))

    def observe(self,data,rest):
        n=len(self.names);rest=np.asarray(rest).reshape(-1,3)
        pos=np.asarray(data.subtree_com[self.bodies]).reshape(-1,3)
        rotations=np.array([data.xmat[b].reshape(3,3) for b in self.bin_bodies])
        binpos=np.array([data.qpos[a:a+3] for a in self.bin_adrs])
        local=np.einsum('nbi,bij->nbj',pos[:,None,:]-binpos[None],rotations)
        membership=np.all(np.abs(local-self.goal_centers[None])<=self.goal_sizes[None],axis=2)
        ind=np.arange(n);correct=membership[ind,self.assigned]&self.active
        wrong=membership[ind,1-self.assigned]&self.active
        target_local=local[ind,self.assigned]
        inner=self.inner[self.assigned]-self.cfg.opening_margin_m
        xy_gap=np.maximum(np.abs(target_local[:,:2])-inner,0)
        # AABB of each collision primitive in the assigned bin frame, so a long tilted bolt clears the rim.
        lower=np.empty(n)
        for k,gs in enumerate(self.collision_geoms):
            rb=rotations[self.assigned[k]]
            zmin=np.inf
            for g in gs:
                delta=(data.geom_xpos[g]-binpos[self.assigned[k]])@rb
                r=rb.T@data.geom_xmat[g].reshape(3,3)
                typ=int(self.model.geom_type[g]);sz=self.model.geom_size[g]
                if typ==int(mujoco.mjtGeom.mjGEOM_BOX):extent=np.abs(r[2])@sz
                elif typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):extent=sz[0]
                elif typ in (int(mujoco.mjtGeom.mjGEOM_CYLINDER),int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                    extent=sz[0]*np.linalg.norm(r[2,:2])+sz[1]*abs(r[2,2])
                    if typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):extent+=sz[0]*abs(r[2,2])
                else:extent=self.model.geom_rbound[g]
                zmin=min(zmin,delta[2]-extent)
            lower[k]=zmin
        above=lower-self.rim[self.assigned]
        over=(xy_gap.max(axis=1)<=0)&(above>=0)
        # Lowered through the opening: CoM inside the inner wall faces, collision extent below the rim, above the
        # floor. The rim (8.5-10.9 cm) sits above the scorer box top (7.9-10.2 cm), so without this column a part
        # lowered >1 cm below the rim but not yet in the scorer box left the drop stage (round-4 "back to carry").
        inside=(xy_gap.max(axis=1)<=self.cfg.opening_exit_m)&(above<0)&(target_local[:,2]>=self.floor[self.assigned])
        gap=np.linalg.norm(np.column_stack((xy_gap,np.maximum(-above,0))),axis=1)
        speed=np.array([np.linalg.norm(data.qvel[a:a+3]) for a in self.dof_adr])
        angular=np.array([np.linalg.norm(data.qvel[a+3:a+6]) for a in self.dof_adr])
        contact=self.contact_obs(data);direct=np.zeros(n,bool);neighbors=[set() for _ in self.names];resting=np.zeros(n,bool)
        touch_arm=np.zeros((n,2),bool);env_touch=np.zeros(n,bool)
        for c in data.contact[:data.ncon]:
            for obj,other in ((int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))):
                typ,k=self.owner[obj];ot,ok=self.owner[other]
                if typ!=0:continue
                arm=self.arm_of[other]
                if arm>=0:touch_arm[k,arm]=True
                else:env_touch[k]=True
                if ot!=1:resting[k]=True
                if other in self.bin_geoms[self.assigned[k]]:direct[k]=True
                elif ot==0 and k!=ok and self.assigned[k]==self.assigned[ok]:neighbors[k].add(ok)
        stable=correct&~contact['touched']&(speed<=self.cfg.at_rest_speed)&(angular<=self.cfg.at_rest_angular_speed)
        supported=stable&direct
        for _ in self.names:
            grown=supported|(stable&np.array([any(supported[j] for j in ns) for ns in neighbors],bool))
            if np.array_equal(grown,supported):break
            supported=grown
        # Handoff arrays (handoff_stage keys): giver = the arm that is not the receiver, for every part. The giver's
        # pinch-centre gap is measured only for cross-side parts the receiver holds with both fingers -- the only case
        # in which handoff_stage reads it (received, and progress once held); elsewhere it is +inf.
        cross,recv=self.handoff_roles(rest);give=1-recv;ind=np.arange(n)
        per_arm=contact['hits'].reshape(n,2,2).sum(axis=2)
        gap_give=np.full(n,np.inf)
        for k in np.flatnonzero(cross&(per_arm[ind,recv]==2)):
            l=self.link6[give[k]];centre=data.xpos[l]+data.xmat[l].reshape(3,3)@PAD
            gap_give[k]=self.surface_gap(data,k,centre)
        middle=np.linalg.norm(np.maximum(np.maximum(MIDDLE_LO-pos,pos-MIDDLE_HI),0),axis=1)
        handoff=dict(com=pos,cross=cross,receiver=recv,f_give=per_arm[ind,give],f_recv=per_arm[ind,recv],
                     touch_give=touch_arm[ind,give],touch_recv=touch_arm[ind,recv],gap_give=gap_give,env_touch=env_touch,
                     middle_gap=middle)
        return dict(xy=pos[:,:2],lift=pos[:,2]-rest[:n,2],speed=speed,angular_speed=angular,in_bin=correct,
                    wrong_bin=wrong,any_bin=membership.any(axis=1),supported=supported,resting_contact=resting,
                    over=over,inside=inside,gap=gap,xy_gap=xy_gap.max(axis=1),above_rim=above,height_to_floor=target_local[:,2]-self.floor[self.assigned],
                    clearance=lower-self.floor[self.assigned],
                    bin_moved=np.linalg.norm(binpos[:,:2]-rest[-2:,:2],axis=1),
                    bin_tilt=np.arccos(np.clip(rotations[:,2,2],-1,1)),**handoff,**contact)

def classify(obs,prev,cfg):
    near=(obs['dist']<=np.where(prev==P,cfg.reach_exit_m,cfg.reach_m))|obs['touched']
    out=np.full(prev.shape,R,np.int64);out[near]=P
    out[obs['grasped']&(obs['lift']>=cfg.lift_m)]=C
    # Cross-side parts (handoff amendment): held and lifted is handoff until the receiver holds it alone
    # (handoff_stage.received_in_air, or received after carry began); the receiver picking it itself meets that too.
    x=obs['cross'];h=cfg.handoff
    later=(prev>=C)&(prev<=DONE)
    received=hs.received_in_air(obs,h)|(hs.received(obs,h)&later)
    handing=x&(hs.handing(obs,prev==H,h)|(out==C))&~received
    out[handing]=H
    safe=obs['in_bin']&(obs['above_rim']<=0)
    # Entry needs the part over the opening (or already in the scorer box); once in drop, lowering inside the walls stays drop.
    over=obs['over']|safe|((prev==D)&(((obs['xy_gap']<=cfg.opening_exit_m)&(obs['above_rim']>=-cfg.opening_exit_m))|obs['inside']))
    # Cross-side: only the receiver may carry/drop it -- the giver touching it or no handoff yet keeps it out of drop.
    receiver_only=~x|(~obs['touch_give']&(later|received))
    out[over&obs['touched']&receiver_only]=D
    at_rest=(obs['speed']<=cfg.at_rest_speed)&(obs['angular_speed']<=cfg.at_rest_angular_speed)
    free=~obs['touched']&(~at_rest|~obs['resting_contact'])
    out[free]=prev[free]
    out[obs['supported']&~obs['touched']&at_rest]=DONE
    out[(obs['lift']<-cfg.fallen_m)&~obs['any_bin']]=FALLEN
    return out

def progress(stage,obs,cfg=None):
    cfg=cfg or StageConfig()
    if stage==R:return -obs['dist']
    if stage==P:return .25*obs['fingers']+.5*np.clip(obs['lift']/.03,0,1)*obs['grasped']
    if stage==H:return hs.progress(obs,cfg.handoff)-np.minimum(obs['middle_gap'],cfg.middle_m)
    if stage==C:return -np.minimum(obs['gap'],1.)
    if stage==D:return -np.clip(obs['height_to_floor'],0,.30)+.05*~obs['touched']
    raise ValueError(stage)

class NutsBoltsTracker(StageTracker):
    def __init__(self,scene,cfg=None):super().__init__(scene,classify,cfg or scene.cfg,start_label=R)
