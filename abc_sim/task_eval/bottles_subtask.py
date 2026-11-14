"""One approved bottle stage, scored with the shared per-chunk reward contract."""
from __future__ import annotations

import numpy as np

from abc_sim.task_eval.bottles import PutBottlesInBinEvaluator
from abc_sim.task_eval.bottles_stages import (C, D, DONE, FALLEN, LABELS, R, STAGE_PROMPTS, STAGES,
                                              BottleScene, BottleTracker, StageConfig, progress)
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


class BottleSubtaskEvaluator(SubtaskEvaluator):
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = "bottle"

    def make_config(self):
        return StageConfig()

    def make_scene(self, model):
        return BottleScene(model, self.cfg)

    def make_tracker(self, scene):
        return BottleTracker(scene, self.cfg)

    def make_stock(self, model):
        return PutBottlesInBinEvaluator(model=model, spec=self.spec)

    def pick_target(self, obs):
        candidates = np.flatnonzero(~obs["in_bin"])
        if not len(candidates):
            raise RuntimeError("no unbinned bottle remains")
        return int(candidates[np.argmin(obs["dist"][candidates])])

    def begin(self, model, data, target, rest_height=None, labels=None):
        super().begin(model, data, target, rest_height=rest_height, labels=labels)
        self._start_xy = self.tracker.obs["xy"][self.target].copy()
        self._binned = self.tracker.obs["in_bin"].copy()
        self._binned[self.target] = False
        self._released = False
        self._was_touched = bool(self.tracker.obs["touched"][self.target])

    def progress(self, stage, obs):
        return progress(stage, obs)

    def step_failure(self, k, obs, label):
        if obs["bin_moved"] > self.cfg.bin_moved_m or obs["bin_tilt"] > self.cfg.bin_tilt_rad:
            return "bin moved or tipped"
        if np.any(self._binned & ~obs["in_bin"]):
            return "knocked a binned bottle out"
        touched = bool(obs["touched"][k])
        if self.stage < C:
            if not obs["grasped"][k] and (np.linalg.norm(obs["xy"][k] - self._start_xy) > self.cfg.push_m
                                            or label >= D and label != FALLEN):
                return "pushed"
        elif self._was_touched and not touched:
            if not (obs["over"][k] or obs["in_bin"][k] and obs["above_rim"][k] <= 0):
                return "released outside the opening"
            self._released = True
        if (self._released and not touched and obs["resting_contact"][k]
                and obs["speed"][k] <= self.cfg.at_rest_speed
                and obs["angular_speed"][k] <= self.cfg.at_rest_angular_speed and not obs["in_bin"][k]):
            return "landed outside the bin"
        self._was_touched = touched
        return ""

    def extra_metrics(self, k, obs):
        return dict(lift_m=float(obs["lift"][k]), grasp_arm=int(obs["grasp_arm"][k]),
                    touched=bool(obs["touched"][k]), over_opening=bool(obs["over"][k]),
                    in_bin=bool(obs["in_bin"][k]), opening_gap_m=float(obs["gap"][k]),
                    bin_moved_m=obs["bin_moved"], bin_tilt_rad=obs["bin_tilt"], speed=float(obs["speed"][k]))

    def stage_prompt(self):
        verb = dict(reach="move a gripper to", pick="grasp and lift", carry="carry over the bin opening",
                    drop="drop into the bin")
        return f"{verb[self.stage_name]} bottle {self.target + 1}"

    def build_start(self, env, seed, rng=None):
        from abc_sim.task_eval.bottles_starts import build_start
        return build_start(env, seed, self.stage_name, rng=rng)

    def evaluate_qpos_batch(self, qpos_batch):
        raise RuntimeError("Bottle subtasks require evaluate_model_data with real contacts and velocities; "
                           "the qpos-only batched path is not supported")
