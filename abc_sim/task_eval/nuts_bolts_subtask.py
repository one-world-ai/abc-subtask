"""One approved nuts/bolts subtask; real CPU contacts/velocities and numerical-failure latching.

Handoff amendment (2026-11-14): cross-side parts go reach -> pick -> handoff -> carry -> drop. In the handoff stage the
failures are ``handoff_stage.failure`` (dropped while > 4 cm up; touched anything but the two arms) plus the
task-wide ones; in carry/drop a cross-side part touched again by the giver is -1 "giver touched the part again".
Same-side parts are scored exactly as before.
"""
from __future__ import annotations
import numpy as np
from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.nuts_bolts import NutsBoltsSortingEvaluator
from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.nuts_bolts_stages import (STAGES,LABELS,FALLEN,STAGE_PROMPTS,R,H,C,D,StageConfig,NutsBoltsScene,NutsBoltsTracker,progress)
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator

class NutsBoltsSubtaskEvaluator(SubtaskEvaluator):
    STAGES,LABELS,FALLEN,STAGE_PROMPTS=STAGES,LABELS,FALLEN,STAGE_PROMPTS
    TARGET_KEY='part'
    def make_config(self):return StageConfig()
    def make_scene(self,model):return NutsBoltsScene(model,self.cfg)
    def make_tracker(self,scene):return NutsBoltsTracker(scene,self.cfg)
    def make_stock(self,model):return NutsBoltsSortingEvaluator(model=model,spec=self.spec)
    def progress(self,stage,obs):return progress(stage,obs,self.cfg)
    def sync_stock_to_episode(self):
        self.stock.set_active_object_joints([n+'_joint' for n,active in zip(self.scene.names,self.scene.active) if active])
    def set_active_object_joints(self,joints):
        if joints and not isinstance(joints[0],str):joints=joints[0]
        active=set(joints);self.scene.active=np.array([n+'_joint' in active for n in self.scene.names])
        self.stock.set_active_object_joints(joints)
    def pick_target(self,obs):
        ks=np.flatnonzero(self.scene.active&~obs['in_bin'])
        if not len(ks):raise RuntimeError('no unsorted active part remains')
        return int(ks[np.argmin(obs['dist'][ks])])
    def begin(self,model,data,target,rest_height=None,labels=None):
        super().begin(model,data,target,rest_height=rest_height,labels=labels)
        self._start_xy=self.tracker.obs['xy'][self.target].copy()
        self._sorted=self.tracker.obs['supported'].copy();self._sorted[self.target]=False
        self._released=False;self._was_touched=bool(self.tracker.obs['touched'][self.target])
        self._time=float(data.time);self._warnings=np.array([w.number for w in data.warning])
    def step_failure(self,k,obs,label):
        if np.any(obs['bin_moved']>self.cfg.bin_moved_m) or np.any(obs['bin_tilt']>self.cfg.bin_tilt_rad):return 'bin moved or tipped'
        if np.any(self._sorted&~obs['in_bin']):return 'knocked a sorted part out'
        self._sorted|=obs['supported'];self._sorted[k]=False
        if np.any(self.scene.active&(obs['lift']<-self.cfg.fallen_m)&~obs['any_bin']):return 'part fell off the table'
        if obs['wrong_bin'][k]:return 'wrong bin'
        touched=bool(obs['touched'][k])
        if self.stage<H:
            if not obs['grasped'][k] and (np.linalg.norm(obs['xy'][k]-self._start_xy)>self.cfg.push_m or label>=D and label!=FALLEN):return 'pushed'
        elif self.stage==H:
            why=hs.failure(obs,k,self.cfg.handoff)
            if why:return why
        elif obs['cross'][k] and obs['touch_give'][k]:return 'giver touched the part again'
        elif self._was_touched and not touched:
            if not (obs['over'][k] or obs['inside'][k] or obs['in_bin'][k] and obs['above_rim'][k]<=0):return 'released outside the opening'
            # Low release only (drop amendment): opening the fingers with the part in the air kicks it numerically.
            if self.stage==D and obs['clearance'][k]>self.cfg.release_max_clearance_m:return 'released too high'
            self._released=True
        if self._released and not touched and obs['resting_contact'][k] and obs['speed'][k]<=self.cfg.at_rest_speed and obs['angular_speed'][k]<=self.cfg.at_rest_angular_speed and not obs['in_bin'][k]:return 'landed outside the bin'
        self._was_touched=touched
        return ''
    def evaluate_model_data(self,model,data):
        bad=not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.time)
        if model is self.model and self.target is not None:
            bad=bool(bad or data.time<self._time-1e-10 or np.any(np.array([w.number for w in data.warning])>self._warnings))
        if bad or self._failed=='numerical instability':
            self._failed='numerical instability';self._success=False
            if self.target is None:
                self.target=-1;self._phi_chunk=0.
            phi=float(self._last.get('progress',0.))
            if not np.isfinite(phi):phi=0.
            self._last.update(subtask=self.stage_name,part=self.target,label=self._last.get('label','invalid'),
                              progress=phi,failed=self._failed,terminal=True,numerical_failure=True)
            return TaskEvalResult(reward=np.array([phi],np.float32),success=np.array([False]),metrics=self._last)
        result=super().evaluate_model_data(model,data)
        self._time=float(data.time);self._warnings=np.array([w.number for w in data.warning])
        return result
    def extra_metrics(self,k,obs):
        return dict(lift_m=float(obs['lift'][k]),cross=bool(obs['cross'][k]),receiver=int(obs['receiver'][k]),
                    f_give=int(obs['f_give'][k]),f_recv=int(obs['f_recv'][k]),touch_give=bool(obs['touch_give'][k]),
                    gap_give_m=float(min(obs['gap_give'][k],9.)),env_touch=bool(obs['env_touch'][k]),grasp_arm=int(obs['grasp_arm'][k]),touched=bool(obs['touched'][k]),
                    in_bin=bool(obs['in_bin'][k]),wrong_bin=bool(obs['wrong_bin'][k]),over_opening=bool(obs['over'][k]),
                    supported=bool(obs['supported'][k]),speed=float(obs['speed'][k]),clearance_m=float(obs['clearance'][k]),opening_gap_m=float(obs['gap'][k]),
                    bin_moved_m=float(obs['bin_moved'].max()),bin_tilt_rad=float(obs['bin_tilt'].max()))
    def stage_prompt(self):
        kind='nut' if self.scene.assigned[self.target]==0 else 'bolt';color='teal' if kind=='nut' else 'orange'
        side=('left','right')[int(self.tracker.obs['receiver'][self.target])]
        verbs=dict(reach=f'move a gripper to the closest unsorted {kind}',pick=f'grasp the {kind} and lift it',
                   handoff=f'hand the {kind} to the {side} gripper',
                   carry=f'carry the {kind} over the {color} bin',drop=f'release the {kind} into the {color} bin')
        return verbs[self.stage_name]
    def build_start(self,env,seed,rng=None):
        from abc_sim.task_eval.nuts_bolts_starts import build_start
        return build_start(env,seed,self.stage_name,rng=rng)
    def evaluate_qpos_batch(self,qpos_batch):
        raise RuntimeError('Nuts/bolts subtasks require real contacts and velocities through evaluate_model_data; qpos-only scoring is unsupported')
