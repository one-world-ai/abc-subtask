"""One mug-rack stage on one target mug, scored with the shared per-chunk reward contract (stagekit.SubtaskEvaluator).

Failures (latched per action, -1, end the episode), on top of the kit's (off the table, back to an earlier stage):
  knocked over          reach/pick: the target's cup axis is > 60 deg from vertical while no finger touches it
  dropped               align/insert: no finger on the target for ``drop_steps`` actions and not threaded deep
  released, not hanging release: no finger, at rest, and the stock scorer does not count it as hanging
  knocked a hanging mug off   any stage: a mug hanging at the start stops hanging for ``knock_steps`` actions
"""
from __future__ import annotations

import numpy as np

from abc_sim.task_eval.mug_rack_stages import (A, DONE, FALLEN, I, LABELS, P, R, RL, STAGE_PROMPTS, STAGES,
                                               MugRackScene, MugRackTracker, StageConfig, progress)
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


class MugRackSubtaskEvaluator(SubtaskEvaluator):
    TASK, ENV_TASK, NAME = "hang_mug_on_mug_rack", "mug_tree", "mug_rack_subtask"  # registration: task_eval/subtasks.py
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = "mug"

    def make_config(self):
        return StageConfig()

    def make_scene(self, model):
        return MugRackScene(model, self.cfg)

    def make_tracker(self, scene):
        return MugRackTracker(scene, self.cfg)

    def make_stock(self, model):
        return None  # the scene runs the stock scorer every observation; stock_success is in extra_metrics

    def configure_from_randomization(self, model, randomization):
        if model is not self.model:
            self._bind(model)
        else:  # the tree's static pose is edited in place per reset: re-measure it (and the mug holes)
            self.scene.geometry.configure_from_randomization(model, randomization)
            self.scene._holes.clear()
            self.scene.measure_tree()

    def pick_target(self, obs):
        candidates = np.flatnonzero(obs["threadable"] & ~obs["hanging"])
        if not len(candidates):
            raise RuntimeError("no threadable mug left to hang (mug_1/7/10 cannot be threaded)")
        return int(candidates[np.argmin(obs["dist"][candidates])])

    def begin(self, model, data, target, rest_height=None, labels=None):
        super().begin(model, data, target, rest_height=rest_height, labels=labels)
        o = self.tracker.obs
        self._hung = o["hanging"].copy()
        self._hung[self.target] = False
        self._knock = np.zeros(len(self._hung), dtype=np.int64)
        self._untouched = 0
        self._depth0 = float(max(o["depth"][self.target], 0.0))
        self._phi_chunk = self._phi_best = self._phi()

    def progress(self, stage, obs):
        p = progress(stage, obs)
        if stage == RL and self.target is not None and hasattr(self, "_depth0"):
            lost = np.minimum(np.maximum(self._depth0 - obs["depth"], 0.0), 0.1)
            p = p - 0.5 * lost
        return p

    def step_failure(self, k, obs, label):
        cfg = self.cfg
        self._knock = np.where(self._hung & ~obs["hanging"], self._knock + 1, 0)
        if np.any(self._knock >= cfg.knock_steps):
            return "knocked a hanging mug off"
        touched = bool(obs["touched"][k])
        self._untouched = 0 if touched else self._untouched + 1
        if self.stage in (R, P) and not touched and obs["upz"][k] < cfg.knocked_upz:
            return "knocked over"
        deep = bool(obs["threaded"][k] and obs["depth"][k] >= cfg.depth_exit_m)
        if self.stage in (A, I) and self._untouched >= cfg.drop_steps and not deep:
            return "dropped"
        if (self.stage == RL and not touched and obs["speed"][k] <= cfg.at_rest_speed and not obs["hanging"][k]
                and self._untouched >= cfg.hold):
            return "released, not hanging"
        return ""

    def extra_metrics(self, k, obs):
        return dict(lift_m=float(obs["lift"][k]), grasp_arm=int(obs["grasp_arm"][k]), touched=bool(obs["touched"][k]),
                    handle_touch=bool(obs["handle_touch"][k]), upright_z=float(obs["upz"][k]),
                    peg=int(self.scene.pegs[int(obs["best_peg"][k])]) if len(self.scene.pegs) else -1,
                    radial_m=float(obs["radial"][k]), axial_m=float(obs["axial"][k]), angle_deg=float(obs["angle"][k]),
                    threaded=bool(obs["threaded"][k]), depth_m=float(obs["depth"][k]),
                    other_contact=bool(obs["other_contact"][k]), hanging=bool(obs["hanging"][k]),
                    speed=float(obs["speed"][k]), variant=self.scene.variants[k],
                    stock_success=bool(obs["hanging"].all()))

    def build_start(self, env, seed, rng=None):
        from abc_sim.task_eval.mug_rack_starts import build_start
        return build_start(env, seed, self.stage_name, rng=rng)

    def evaluate_qpos_batch(self, qpos_batch):
        raise RuntimeError("Mug-rack subtasks require evaluate_model_data with real contacts and velocities; "
                           "the qpos-only batched path is not supported")
