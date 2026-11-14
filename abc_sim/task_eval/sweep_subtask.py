"""One approved sweep hand-pick stage (chain H), scored with the shared per-chunk reward contract."""
from __future__ import annotations

import numpy as np

from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator
from abc_sim.task_eval.sweep import SweepAwayEvaluator
from abc_sim.task_eval.sweep_stages import (C, D, FALLEN, LABELS, STAGE_PROMPTS, STAGES, StageConfig, SweepScene,
                                            SweepTracker, progress)


class SweepSubtaskEvaluator(SubtaskEvaluator):
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = "scrap"
    _active_joints = None

    def make_config(self):
        return StageConfig()

    def make_scene(self, model):
        scene = SweepScene(model, self.cfg)
        if self._active_joints is not None:
            scene.set_active(self._mask(scene.names, self._active_joints))
        return scene

    def make_tracker(self, scene):
        return SweepTracker(scene, self.cfg)

    def make_stock(self, model):
        return SweepAwayEvaluator(model=model, spec=self.spec)

    @staticmethod
    def _mask(names, joints):
        wanted = {str(j).removesuffix("_jnt") for j in joints}
        unknown = wanted - set(names)
        if unknown:
            raise ValueError(f"unknown sweep scraps {sorted(unknown)}")
        return np.array([n in wanted for n in names], dtype=bool)

    def set_active_trash_joints(self, joints):
        """Episode active set from the randomizer metadata (``env.reset`` calls this after every reset)."""
        self._active_joints = list(joints)
        self.scene.set_active(self._mask(self.scene.names, joints))
        self.sync_stock_to_episode()

    def sync_stock_to_episode(self):
        if self.stock is not None and self._active_joints is not None:
            self.stock.set_active_trash_joints(list(self._active_joints))

    def pick_target(self, obs):
        candidates = np.flatnonzero(obs["active"] & ~obs["in_bin"])
        if not len(candidates):
            raise RuntimeError("no unbinned active scrap remains")
        return int(candidates[np.argmin(obs["dist"][candidates])])

    def begin(self, model, data, target, rest_height=None, labels=None):
        super().begin(model, data, target, rest_height=rest_height, labels=labels)
        obs = self.tracker.obs
        if not obs["active"][self.target]:
            raise ValueError(f"scrap {self.target} is not active in this episode")
        self._start_xy = obs["xy"][self.target].copy()
        self._binned = obs["in_bin"] & obs["active"]
        self._binned[self.target] = False
        self._released = False
        self._was_touched = bool(obs["touched"][self.target])

    def progress(self, stage, obs):
        return progress(stage, obs)

    def step_failure(self, k, obs, label):
        if obs["bin_moved"] > self.cfg.bin_moved_m or obs["bin_tilt"] > self.cfg.bin_tilt_rad:
            return "bin moved or tipped"
        if np.any(self._binned & ~obs["in_bin"]):
            return "knocked a binned scrap out"
        others = obs["off_table"].copy()
        others[k] = False  # the target's own fall is the generic FALLEN failure
        if np.any(others):
            return "a scrap fell off the table"
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
                    bin_moved_m=obs["bin_moved"], bin_tilt_rad=obs["bin_tilt"], speed=float(obs["speed"][k]),
                    active_scraps=int(obs["active"].sum()), scraps_in_bin=int(obs["in_bin"].sum()))

    def stage_prompt(self):
        n = self.target + 1
        return dict(reach=f"move a gripper to paper scrap {n}", pick=f"grasp paper scrap {n} and lift it",
                    carry=f"carry paper scrap {n} over the bin opening",
                    drop=f"drop paper scrap {n} into the bin")[self.stage_name]

    def build_start(self, env, seed, rng=None):
        from abc_sim.task_eval.sweep_starts import build_start
        return build_start(env, seed, self.stage_name, rng=rng)

    def evaluate_qpos_batch(self, qpos_batch):
        raise RuntimeError("Sweep subtasks require evaluate_model_data with real contacts and velocities; "
                           "the qpos-only batched path is not supported")
