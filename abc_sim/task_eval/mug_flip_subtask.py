"""Evaluator for one mug-flip subtask (reach / pick / rotate / place) on one target mug, for single-stage RL.

Use with ``env.reset_to_stage(seed, stage)`` (or ``reset_to_snapshot`` + ``begin``), which picks the target mug and
the episode's rest heights. A plain ``env.reset`` also works for reach: the target is then the mug nearest a
gripper at the first evaluation.

Every physics step (``env.step`` evaluates every action) the stage tracker is updated and two failures are latched:
  dropped   the target mug touches no finger while above rest height + ``drop_margin_m`` (released in the air;
            rotate and place only: a mug let go mid-air can land upright inside one chunk, so it must be caught
            per step)
  regressed the debounced stage fell below the subtask's stage, or the mug left the table

Result: ``reward`` = the stage's progress number (dense, per step), ``success`` = the mug reached a later stage
(the subtask goal) without a latched failure, metrics carry label / progress / failed / terminal.

RL reward per chunk: call ``chunk_reward()`` after each chunk: +1 / -1 / 0 from the change in progress (dead band
``StageConfig.eps``), +1 on success and -1 on failure (both terminal). Options: ``continuous=True`` returns the raw
progress change instead of its sign; ``best_so_far=True`` gives +1 only when progress beats its best so far in the
episode by more than the dead band, else 0 (no -1 for moving back and forth; success/failure unchanged).
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.mug_flip_stages import (FALLEN, LABELS, PL, R, ROT, STAGE_PROMPTS, STAGES, MugScene,
                                               MugStageTracker, StageConfig, progress)
from abc_sim.task_eval.mugs import MugFlipUprightEvaluator
from abc_sim.task_specs import SimTaskSpec


class MugFlipSubtaskEvaluator:
    def __init__(self, *, model: mujoco.MjModel, spec: SimTaskSpec, stage: str, drop_margin_m: float = 0.01,
                 continuous: bool = False, best_so_far: bool = False) -> None:
        if stage not in STAGE_PROMPTS:
            raise ValueError(f"stage must be one of {tuple(STAGE_PROMPTS)}, got {stage!r}")
        self.spec, self.stage_name, self.stage = spec, stage, STAGES.index(stage)
        self.cfg = StageConfig()
        self.drop_margin_m = float(drop_margin_m)
        self.continuous = bool(continuous)
        self.best_so_far = bool(best_so_far)
        self._bind(model)

    def _bind(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.scene = MugScene(model)
        self.stock = MugFlipUprightEvaluator(model=model, spec=self.spec)  # stock success, for reference
        self.tracker = MugStageTracker(self.scene, self.cfg)
        self.mug: int | None = None
        self._scratch: mujoco.MjData | None = None
        self.reset()

    # -- episode control -----------------------------------------------------------------------------------------

    def reset(self, *, nworld: int = 1) -> None:
        if nworld != 1:
            raise ValueError("MugFlipSubtaskEvaluator is single-world")
        self.mug = None
        self._failed = ""
        self._success = False
        self._phi_chunk: float | None = None
        self._phi_best: float | None = None
        self._last: dict[str, Any] = {}
        self.stock.reset(nworld=1)

    def configure_from_randomization(self, model: mujoco.MjModel, randomization: object) -> None:
        if model is not self.model:
            self._bind(model)

    def begin(self, model: mujoco.MjModel, data: mujoco.MjData, mug: int, rest_height: np.ndarray | None = None,
              labels: np.ndarray | None = None) -> None:
        """Start the episode on ``mug``. A restored mid-episode start passes its saved rest heights and labels.
        ``model`` is the env's current one: mug scenes swap models when the mug count changes."""
        if model is not self.model:
            self._bind(model)
        self.reset()
        self.mug = int(mug)
        self.tracker.reset(data, rest_height=rest_height, labels=labels)
        self.stock._rest_height = self.tracker.rest_height[None].astype(np.float32)  # the episode's, not the restored
        self._phi_chunk = self._phi_best = self._phi()

    def _phi(self) -> float:
        return float(progress(self.stage, self.tracker.obs)[self.mug])

    # -- evaluation ----------------------------------------------------------------------------------------------

    def evaluate_model_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> TaskEvalResult:
        if model is not self.model:
            self._bind(model)
        if self.mug is None:  # plain reset: target the mug nearest a gripper
            self.tracker.reset(data)
            self.begin(model, data, int(np.argmin(self.tracker.obs["dist"])), rest_height=self.tracker.rest_height)
        else:
            self.tracker.update(data)
        k, o = self.mug, self.tracker.obs
        label = int(self.tracker.committed[k])
        if not (self._failed or self._success):
            in_air = bool(~o["touched"][k] & (o["lift"][k] > self.cfg.rest_lift_m + self.drop_margin_m))
            if self.stage in (ROT, PL) and in_air:
                self._failed = "dropped"
            elif label == FALLEN:
                self._failed = "fell off the table"
            elif label < self.stage:
                self._failed = f"back to {LABELS[label]}"
            elif label > self.stage:
                self._success = True
        phi = self._phi()
        stock = self.stock.evaluate_qpos_batch(np.asarray(data.qpos, np.float32)[None])
        metrics: dict[str, Any] = {
            "subtask": self.stage_name, "mug": k, "label": LABELS[label], "progress": phi,
            "failed": self._failed, "terminal": bool(self._failed or self._success),
            "upright_z": float(o["upz"][k]), "lift_m": float(o["lift"][k]), "grasp_arm": int(o["grasp_arm"][k]),
            "touched": bool(o["touched"][k]), "stock_success": bool(stock.success[0]),
        }
        self._last = metrics
        success = np.array([self._success and not self._failed])
        return TaskEvalResult(reward=np.array([phi], np.float32), success=success, metrics=metrics)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        """qpos-only path (contacts need a forward pass on a scratch MjData). Single world."""
        qpos_batch = np.asarray(qpos_batch)
        if qpos_batch.shape[0] != 1:
            raise ValueError("MugFlipSubtaskEvaluator is single-world")
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
        self._scratch.qpos[:] = qpos_batch[0]
        mujoco.mj_forward(self.model, self._scratch)
        return self.evaluate_model_data(self.model, self._scratch)

    def chunk_reward(self) -> tuple[float, bool]:
        """Reward for the chunk since the last call, and whether the episode is over. Call after each chunk."""
        if self.mug is None or self._phi_chunk is None:
            raise RuntimeError("no episode: evaluate (env.step) first")
        if self._failed:
            return -1.0, True
        if self._success:
            return 1.0, True
        phi = self._last.get("progress", self._phi_chunk)
        d = phi - self._phi_chunk
        self._phi_chunk = phi
        eps = self.cfg.eps[self.stage_name]
        if self.best_so_far:  # +1 only for a new best (by more than eps), else 0: wandering back and forth earns nothing
            if phi > self._phi_best + eps:
                self._phi_best = phi
                return 1.0, False
            return 0.0, False
        if self.continuous:
            return float(d), False
        return (1.0 if d > eps else -1.0 if d < -eps else 0.0), False
