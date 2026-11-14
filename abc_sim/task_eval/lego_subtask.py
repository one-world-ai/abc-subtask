"""One approved LEGO stage on a fixed nearest unsorted active block.

Handoff (amendment 3, cross-side blocks only): failures are the shared ``handoff_stage.failure`` (dropped; touched
anything but the two arms -- table, bin or another block) plus "giver carried it across": the giver touches the
block while its CoM is more than ``region_exit_m`` past the region's edge on the receiver's side.
"""
from __future__ import annotations

import numpy as np

from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.lego_stages import (C, D, FALLEN, H, LABELS, P, STAGE_PROMPTS, STAGES,
                                         COLORS, LegoScene, LegoTracker, StageConfig, progress)
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


class LegoSubtaskEvaluator(SubtaskEvaluator):
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = 'block'

    def make_config(self): return StageConfig()
    def make_scene(self, model):
        self._active_joint_names = None
        return LegoScene(model, self.cfg)
    def make_tracker(self, scene): return LegoTracker(scene, self.cfg)
    def make_stock(self, model): return LegoBlocksSortingEvaluator(model=model, spec=self.spec)
    def progress(self, stage, obs): return progress(stage, obs, self.cfg)

    def pick_target(self, obs):
        candidates = np.flatnonzero(obs['active'] & ~obs['in_bin'])
        if not len(candidates): raise RuntimeError('no unsorted active LEGO block remains')
        return int(candidates[np.argmin(obs['dist'][candidates])])

    def set_active_object_joints(self, joints):
        if joints and not isinstance(joints[0], str):
            if len(joints) != 1: raise ValueError('LEGO subtasks are single-world')
            joints = joints[0]
        self._active_joint_names = list(joints)
        active = set(joints)
        self.scene.active = np.array([name+'_joint' in active for name in self.scene.names])
        self.stock.set_active_object_joints(joints)

    def sync_stock_to_episode(self):
        if self._active_joint_names is not None:
            self.stock.set_active_object_joints(self._active_joint_names)

    def begin(self, model, data, target, rest_height=None, labels=None):
        super().begin(model, data, target, rest_height=rest_height, labels=labels)
        self._start_xy = self.tracker.obs['xy'][self.target].copy()
        self._sorted = self.tracker.obs['in_bin'].copy(); self._sorted[self.target] = False
        self._wrong = self.tracker.obs['wrong_bin'].copy()
        self._released = False
        self._was_touched = bool(self.tracker.obs['touched'][self.target])

    def step_failure(self, k, obs, label):
        if np.any(obs['bin_moved'] > self.cfg.bin_moved_m) or np.any(obs['bin_tilt'] > self.cfg.bin_tilt_rad):
            return 'bin moved or tipped'
        if np.any(self._sorted & ~obs['in_bin']): return 'knocked a sorted block out'
        if obs['wrong_bin'][k] or np.any(obs['wrong_bin'] & ~self._wrong): return 'wrong bin'
        others = (np.arange(len(self.scene.names)) != k) & obs['active']
        if np.any((obs['lift'] < -self.cfg.fallen_m) & ~obs['in_bin'] & others):
            return 'another block fell off the table'
        touched = bool(obs['touched'][k])
        if self.stage == H:
            h = self.cfg.handoff
            why = hs.failure(obs, k, h)
            if not why and obs['touch_give'][k] and obs['y_recv'][k] > h.region_hi[1] + h.region_exit_m:
                why = 'giver carried it across'
            self._was_touched = touched
            return why
        if self.stage <= P:
            moved = np.linalg.norm(obs['xy'][k]-self._start_xy) > self.cfg.push_m
            # In reach any slide beyond push_m is a push: a real grasp ends reach first (grasp site within
            # reach_m), and two closed fingers dragging the block from outside are not a grasp (G5 round 3).
            if moved and self.stage == 0:
                return 'pushed'
            if not obs['grasped'][k] and (moved or label >= D and label != FALLEN):
                return 'pushed'
        elif self._was_touched and not touched:
            if not (obs['over'][k] or obs['in_bin'][k] and obs['above_rim'][k] <= 0):
                return 'released outside the opening'
            self._released = True
        if (self._released and not touched and obs['resting_contact'][k]
                and obs['speed'][k] <= self.cfg.at_rest_speed
                and obs['angular_speed'][k] <= self.cfg.at_rest_angular_speed and not obs['in_bin'][k]):
            return 'landed outside the bin'
        self._was_touched = touched
        return ''

    def extra_metrics(self, k, obs):
        return dict(target_name=self.scene.names[k], color=COLORS[self.scene.colors[k]],
                    lift_m=float(obs['lift'][k]), grasp_arm=int(obs['grasp_arm'][k]),
                    touched=bool(obs['touched'][k]), over_opening=bool(obs['over'][k]),
                    in_bin=bool(obs['in_bin'][k]), supported=bool(obs['supported'][k]),
                    wrong_bin=bool(obs['wrong_bin'][k]), opening_gap_m=float(obs['gap'][k]),
                    bin_moved_m=float(obs['bin_moved'].max()), bin_tilt_rad=float(obs['bin_tilt'].max()),
                    speed=float(obs['speed'][k]), cross_side=bool(obs['cross'][k]),
                    giver=int(obs['giver'][k]), receiver=int(obs['receiver'][k]), f_give=int(obs['f_give'][k]),
                    f_recv=int(obs['f_recv'][k]), touch_give=bool(obs['touch_give'][k]),
                    gap_give_m=float(obs['gap_give'][k]), env_touch=bool(obs['env_touch'][k]),
                    region_gap_m=float(obs['m_gap'][k]))

    def stage_prompt(self):
        name = self.scene.names[self.target]
        color, _, index = name.split('_')
        return dict(reach=f'move a gripper to {color} block {index}',
                    pick=f'grasp {color} block {index} and lift it',
                    handoff=f'hand {color} block {index} to the other arm',
                    carry=f'carry {color} block {index} over the {color} bin',
                    drop=f'drop {color} block {index} into the {color} bin')[self.stage_name]

    def build_start(self, env, seed, rng=None):
        from abc_sim.task_eval.lego_starts import build_start
        return build_start(env, seed, self.stage_name, rng=rng)

    def evaluate_qpos_batch(self, qpos_batch):
        raise RuntimeError('LEGO subtasks require real contacts and velocities via evaluate_model_data; '
                           'the qpos-only batched path is not supported')
