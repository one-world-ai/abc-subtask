"""One approved dish-rack stage (carry / insert / release), scored with the shared per-chunk reward contract.

Failures (latched, -1), checked every physics step before the kit's generic ones (fell off the table, back to an
earlier stage):
  all stages        rack moved > 3 cm or tilted > 10 deg; a plate that was in the rack at the start knocked out
  carry, insert     dropped (no jaw on the plate for 5 steps before it is seated); hit the table outside the rack
                    footprint
  insert            lying flat on the rack for > 2 chunks; jammed (held and pressed onto the rack with > 2 plate weights,
                    no 3 mm descent in 4 chunks)
  release           pulled out (held and lifted 2 cm above its seated height); left the slot or fell over after the
                    release (not in the stock in-rack geometry for 5 actions with no jaw on it)
"""
from __future__ import annotations

from collections import deque

import numpy as np

from abc_sim.task_eval.dishrack_stages import (C, FALLEN, I, LABELS, RL, STAGE_PROMPTS, STAGES, DishrackScene,
                                               DishrackTracker, StageConfig, progress, seated)
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


class DishrackSubtaskEvaluator(SubtaskEvaluator):
    TASK, ENV_TASK, NAME = "load_plates_into_dish_rack", "dishrack", "dishrack_subtask"  # registration: task_eval/subtasks.py
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = "plate"

    def make_config(self):
        return StageConfig()

    def make_scene(self, model):
        return DishrackScene(model, self.cfg)

    def make_tracker(self, scene):
        return DishrackTracker(scene, self.cfg)

    def pick_target(self, obs):
        """The unracked plate nearest a gripper (decision 3)."""
        candidates = np.flatnonzero(~obs["stock_ok"])
        if not len(candidates):
            raise RuntimeError("every plate is already in the rack")
        return int(candidates[np.argmin(obs["dist"][candidates])])

    def begin(self, model, data, target, rest_height=None, labels=None):
        if model is not self.model:
            self._bind(model)
        self.scene.set_target(data, target)
        self._nz0 = None
        super().begin(model, data, target, rest_height=rest_height, labels=labels)
        o, k = self.tracker.obs, self.target
        self._nz0 = float(o["nz"][k])
        self._low0 = float(o["low"][k])
        self._racked = o["stock_ok"].copy()
        self._racked[k] = False
        self._free_steps = self._flat_steps = self._left_steps = self._knock_steps = 0
        self._lows: deque = deque(maxlen=self.cfg.jam_steps)
        self._phi_chunk = self._phi_best = self._phi()

    def progress(self, stage, obs):
        k = self.target
        nz0 = self._nz0 if getattr(self, "_nz0", None) is not None else float(obs["nz"][k])
        return progress(stage, obs, nz0=nz0)

    def step_failure(self, k, obs, label):
        cfg = self.cfg
        if obs["rack_moved"] > cfg.rack_moved_m or obs["rack_tilt"] > cfg.rack_tilt_rad:
            return "rack moved or tipped"
        knocked = self._racked & ~obs["in_rack_geom"]
        self._knock_steps = self._knock_steps + 1 if knocked.any() else 0
        if self._knock_steps >= cfg.hold:
            return "knocked a racked plate out"
        touched = bool(obs["touched"][k])
        self._free_steps = 0 if touched else self._free_steps + 1
        if self.stage in (C, I):
            if self._free_steps >= cfg.drop_steps and label < RL:
                return "dropped"
            if obs["world_contact"][k] and not obs["over_fp"][k]:
                return "plate hit the table"
        if self.stage == I:
            flat = obs["nz"][k] >= cfg.flat_nz and obs["rack_contact"][k]
            self._flat_steps = self._flat_steps + 1 if flat else 0
            if self._flat_steps > cfg.flat_steps:
                return "lying flat on the rack"
            is_seated = bool(seated(obs, np.full(len(obs["nz"]), label), cfg)[k])
            pressed = obs["rack_force_w"][k] > cfg.jam_force_weights
            if touched and obs["rack_contact"][k] and pressed and not is_seated:
                self._lows.append(float(obs["low"][k]))
                if len(self._lows) == self._lows.maxlen and max(self._lows) - self._lows[-1] < cfg.jam_descent_m:
                    return "jammed on the rack"
            else:
                self._lows.clear()
        if self.stage == RL:
            if touched and obs["low"][k] > self._low0 + cfg.pull_m:
                return "pulled out"
            left = not touched and not obs["in_rack_geom"][k]     # fell over (|nz| > 0.55) or left the rack
            self._left_steps = self._left_steps + 1 if left else 0
            if self._left_steps >= cfg.left_steps:
                return "left the slot or fell over"
        return ""

    def extra_metrics(self, k, obs):
        return dict(touched=bool(obs["touched"][k]), grasped=bool(obs["grasped"][k]),
                    grasp_arm=int(obs["grasp_arm"][k]), normal_z=float(obs["nz"][k]), low_m=float(obs["low"][k]),
                    slot_dist=float(obs["slot_dist"][k]), seat_gap_m=float(obs["seat_gap"][k]),
                    aligned=bool(obs["aligned"][k]), at_slot=bool(obs["at_slot"][k]), seat_m=float(obs["seat"][k]),
                    in_rack_geom=bool(obs["in_rack_geom"][k]), target_stock_success=bool(obs["stock_ok"][k]),
                    jaw_gap_m=float(min(obs["jaw_gap"][k], 1.0)), speed=float(obs["speed"][k]),
                    rack_moved_m=obs["rack_moved"], rack_tilt_rad=obs["rack_tilt"])

    def build_start(self, env, seed, rng=None):
        from abc_sim.task_eval.dishrack_starts import build_start
        return build_start(env, seed, self.stage_name, rng=rng)

    def evaluate_qpos_batch(self, qpos_batch):
        raise RuntimeError("Dish-rack subtasks need evaluate_model_data with real contacts and velocities; "
                           "the qpos-only batched path is not supported")
