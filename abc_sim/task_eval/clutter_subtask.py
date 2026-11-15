"""A single approved clutter subtask, with fixed prompted target and real CPU state.

Handoff (amendment 2026-11-14): failures are the scene failures (box moved/tipped, distractor off table, wrong object
lifted) and the task-independent ``handoff_stage.failure`` (dropped; touched anything but the two arms -- table, box
or clutter: setting it down and re-picking is not a handoff). Leaving the handoff region by more than 10 cm while
the receiver already has it = back to an earlier stage (-1), as in inhand.
"""
from __future__ import annotations
import numpy as np
from abc_sim.task_eval.grab_clutter import GrabClutterTargetInBoxEvaluator
from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.clutter_stages import C,H,PL,R,FALLEN,LABELS,STAGES,STAGE_PROMPTS,ClutterScene,ClutterTracker,StageConfig,progress
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator

class ClutterSubtaskEvaluator(SubtaskEvaluator):
    TASK, ENV_TASK, NAME = "grab_specific_object_from_clutter", "grab_clutter", "clutter_subtask"  # registration: task_eval/subtasks.py
    STAGES,LABELS,FALLEN,STAGE_PROMPTS=STAGES,LABELS,FALLEN,STAGE_PROMPTS
    TARGET_KEY='target'
    def reset(self,*,nworld=1):
        super().reset(nworld=nworld);self._last_sim_time=None
    def evaluate_model_data(self,model,data):
        if model is not self.model:self._last_sim_time=None
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or any(w.number for w in data.warning)
                or self._last_sim_time is not None and data.time<self._last_sim_time-1e-10 and not getattr(self,'allow_rewind',False)):
            raise RuntimeError('invalid MuJoCo numerical state or physics time reset')
        result=super().evaluate_model_data(model,data);self._last_sim_time=float(data.time);return result
    def make_config(self):return StageConfig()
    def make_scene(self,model):return ClutterScene(model,self.cfg)
    def make_tracker(self,scene):return ClutterTracker(scene,self.cfg)
    def make_stock(self,model):return GrabClutterTargetInBoxEvaluator(model=model,spec=self.spec)
    def configure_from_randomization(self,model,randomization):
        super().configure_from_randomization(model,randomization)
        self.scene.label=getattr(randomization,'metadata',{}).get('target_label','object')
    def pick_target(self,obs):return 0
    def begin(self,model,data,target,rest_height=None,labels=None):
        if int(target)!=0:raise ValueError('clutter target is fixed by the task prompt')
        super().begin(model,data,target,rest_height=rest_height,labels=labels)
        self._was_touched=bool(self.tracker.obs['touched'][target]);self._released=False
    def progress(self,stage,obs):return progress(stage,obs,self.cfg)
    def step_failure(self,k,obs,label):
        if obs['box_moved']>self.cfg.box_moved_m or obs['box_tilt']>self.cfg.box_tilt_rad:return 'box moved or tipped'
        if np.any(obs['off_table'][1:]):return 'distractor off table'
        if np.any(obs['lift'][1:]>=self.cfg.lift_m):return 'wrong object lifted'
        touched=bool(obs['touched'][k])
        if self.stage==H:
            self._was_touched=touched
            return hs.failure(obs,k,self.cfg.handoff)
        if self.stage<H:
            if not obs['grasped'][k] and (obs['moved_xy'][k]>self.cfg.push_m or label>=PL and label!=FALLEN):return 'pushed target'
        elif self._was_touched and not touched:
            if not (obs['in_opening'][k] or obs['in_box'][k]):return 'released outside opening'
            if obs['clearance'][k]>self.cfg.drop_clearance_m:return 'released too high'
            self._released=True
        if self._released and not touched and obs['resting_contact'][k] and obs['speed'][k]<=self.cfg.at_rest_speed and obs['angular_speed'][k]<=self.cfg.at_rest_angular_speed and not obs['in_box'][k]:return 'landed outside box'
        self._was_touched=touched
        return ''
    def extra_metrics(self,k,obs):
        return dict(target_label=self.scene.label,lift_m=float(obs['lift'][k]),grasp_arm=int(obs['grasp_arm'][k]),
                    touched=bool(obs['touched'][k]),in_box=bool(obs['in_box'][k]),floor_support=bool(obs['floor_support'][k]),
                    opening_gap_m=float(obs['gap'][k]),clearance_m=float(obs['clearance'][k]),speed=float(obs['speed'][k]),
                    box_moved_m=obs['box_moved'],box_tilt_rad=obs['box_tilt'],giver=int(obs['giver']),
                    f_give=int(obs['f_give'][k]),f_recv=int(obs['f_recv'][k]),touch_give=bool(obs['touch_give'][k]),
                    touch_recv=bool(obs['touch_recv'][k]),gap_give_m=float(obs['gap_give'][k]),
                    env_touch=bool(obs['env_touch'][k]),handoff_region_gap_m=float(obs['m_gap'][k]))
    def stage_prompt(self):
        return dict(reach=f'move a gripper to the {self.scene.label} in the clutter',pick=f'grasp the {self.scene.label} and lift it from the clutter',handoff=f'hand the {self.scene.label} to the other arm',carry=f'carry the {self.scene.label} over the box',place=f'place the {self.scene.label} inside the box and let go')[self.stage_name]
    def build_start(self,env,seed,rng=None):
        from abc_sim.task_eval.clutter_starts import build_start
        return build_start(env,seed,self.stage_name,rng=rng)
    def evaluate_qpos_batch(self,qpos_batch):raise RuntimeError('Clutter subtasks require real CPU contacts and velocities via evaluate_model_data')
