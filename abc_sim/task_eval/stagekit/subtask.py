"""Evaluator for one subtask (one stage) on one target object, for single-stage RL.

Use with ``env.reset_to_stage(seed, start)`` (or ``reset_to_snapshot`` + ``begin``), which picks the target object and
the episode's rest heights. A plain ``env.reset`` also works for the first stage: the target is then the object
nearest a gripper at the first evaluation (``pick_target``).

Every physics step (``env.step`` evaluates every action) the stage tracker is updated and failures are latched:
  task-specific ``step_failure`` (e.g. "dropped"), then off the table (the FALLEN label), then a fall back to an
  earlier stage. Reaching a later stage is success. The first of these ends the episode.

Result: ``reward`` = the stage's progress number (dense, per step), ``success`` = the goal was reached without a
latched failure; metrics carry label / progress / failed / terminal plus the task's ``extra_metrics``.

RL reward per chunk: call ``chunk_reward()`` after each chunk. +1 on success and -1 on failure (both end the
episode); in between it depends on ``reward_mode``:
  "progress" (default)  shaping_total x (change in progress) / cfg.span[stage]. Potential-based shaping: the
                        in-between rewards sum to shaping_total x (progress gained) / span however many chunks it
                        takes, so going slowly or wobbling earns nothing extra.
  "sign"                +1 / -1 / 0 when progress rose / fell by more than the dead band cfg.eps[stage].
  "best"                +1 only when progress beats its best so far by more than the dead band, else 0.
"sign" and "best" pay each chunk of progress a full +1, so slow progress can out-earn finishing: use them with a
discount factor or a per-chunk cost.

A task subclasses this and sets STAGES / LABELS / FALLEN / STAGE_PROMPTS / TARGET_KEY and the hooks below.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.stagekit.tracker import StageTracker
from abc_sim.task_specs import SimTaskSpec


class SubtaskEvaluator:
    STAGES: tuple[str, ...] = ()  # trainable stages in order, then the terminal label(s) (e.g. "done")
    LABELS: tuple[str, ...] = ()  # STAGES + ("fallen",)
    FALLEN: int = -1  # label index of an object off the table
    STAGE_PROMPTS: dict[str, str] = {}
    TARGET_KEY = "object"  # metrics key naming the target (mug flip: "mug")

    def __init__(self, *, model: mujoco.MjModel, spec: SimTaskSpec, stage: str, drop_margin_m: float = 0.01,
                 reward_mode: str = "progress", shaping_total: float = 0.5) -> None:
        if stage not in self.STAGE_PROMPTS:
            raise ValueError(f"stage must be one of {tuple(self.STAGE_PROMPTS)}, got {stage!r}")
        self.spec, self.stage_name, self.stage = spec, stage, self.STAGES.index(stage)
        self.cfg = self.make_config()
        self.drop_margin_m = float(drop_margin_m)
        if reward_mode not in ("progress", "sign", "best"):
            raise ValueError(f"reward_mode must be progress, sign or best, got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.shaping_total = float(shaping_total)
        self._bind(model)

    # -- task hooks ----------------------------------------------------------------------------------------------

    def make_config(self):
        raise NotImplementedError

    def make_scene(self, model: mujoco.MjModel):
        raise NotImplementedError

    def make_tracker(self, scene) -> StageTracker:
        raise NotImplementedError

    def make_stock(self, model: mujoco.MjModel):
        """The task's stock evaluator (reported as ``stock_success``), or None."""
        return None

    def progress(self, stage: int, obs: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def step_failure(self, k: int, obs: dict[str, np.ndarray], label: int) -> str:
        """Task-specific failure checked every step before the generic ones; "" if none."""
        return ""

    def extra_metrics(self, k: int, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        return {}

    def pick_target(self, obs: dict[str, np.ndarray]) -> int:
        return int(np.argmin(obs["dist"]))

    def sync_stock_to_episode(self) -> None:
        """Called by ``begin``: e.g. give the stock evaluator the episode's rest heights, not the restored state's."""

    def stage_prompt(self) -> str:
        """The prompt of this stage for the current episode (default: the fixed ``STAGE_PROMPTS`` entry)."""
        return self.STAGE_PROMPTS[self.stage_name]

    def build_start(self, env, seed: int, rng: np.random.Generator | None = None) -> tuple[dict | None, list[dict]]:
        """Build a start of this stage for ``seed`` (``env.reset_to_stage`` without a saved start)."""
        raise NotImplementedError

    # -- binding -------------------------------------------------------------------------------------------------

    def _bind(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.scene = self.make_scene(model)
        self.stock = self.make_stock(model)
        self.tracker = self.make_tracker(self.scene)
        self.target: int | None = None
        self._scratch: mujoco.MjData | None = None
        self.reset()

    # -- episode control -----------------------------------------------------------------------------------------

    def reset(self, *, nworld: int = 1) -> None:
        if nworld != 1:
            raise ValueError(f"{type(self).__name__} is single-world")
        self.target = None
        self._failed = ""
        self._success = False
        self._phi_chunk: float | None = None
        self._phi_best: float | None = None
        self._last: dict[str, Any] = {}
        if self.stock is not None:
            self.stock.reset(nworld=1)

    def configure_from_randomization(self, model: mujoco.MjModel, randomization: object) -> None:
        if model is not self.model:
            self._bind(model)

    def begin(self, model: mujoco.MjModel, data: mujoco.MjData, target: int, rest_height: np.ndarray | None = None,
              labels: np.ndarray | None = None) -> None:
        """Start the episode on object ``target``. A restored mid-episode start passes its saved rest heights and
        labels. ``model`` is the env's current one: scenes with a variable object count swap models."""
        if model is not self.model:
            self._bind(model)
        self.reset()
        self.target = int(target)
        self.tracker.reset(data, rest_height=rest_height, labels=labels)
        self.sync_stock_to_episode()
        self._phi_chunk = self._phi_best = self._phi()

    def _phi(self) -> float:
        return float(self.progress(self.stage, self.tracker.obs)[self.target])

    # -- evaluation ----------------------------------------------------------------------------------------------

    def evaluate_model_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> TaskEvalResult:
        if model is not self.model:
            self._bind(model)
        if self.target is None:  # plain reset: target the object nearest a gripper
            self.tracker.reset(data)
            self.begin(model, data, self.pick_target(self.tracker.obs), rest_height=self.tracker.rest_height)
        else:
            self.tracker.update(data)
        k, o = self.target, self.tracker.obs
        label = int(self.tracker.committed[k])
        if not (self._failed or self._success):
            why = self.step_failure(k, o, label)
            if why:
                self._failed = why
            elif label == self.FALLEN:
                self._failed = "fell off the table"
            elif label < self.stage:
                self._failed = f"back to {self.LABELS[label]}"
            elif label > self.stage:
                self._success = True
        phi = self._phi()
        metrics: dict[str, Any] = {
            "subtask": self.stage_name, self.TARGET_KEY: k, "label": self.LABELS[label], "progress": phi,
            "failed": self._failed, "terminal": bool(self._failed or self._success),
            **self.extra_metrics(k, o),
        }
        if self.stock is not None:
            stock = self.stock.evaluate_qpos_batch(np.asarray(data.qpos, np.float32)[None])
            metrics["stock_success"] = bool(stock.success[0])
        self._last = metrics
        success = np.array([self._success and not self._failed])
        return TaskEvalResult(reward=np.array([phi], np.float32), success=success, metrics=metrics)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        """qpos-only path (contacts need a forward pass on a scratch MjData). Single world."""
        qpos_batch = np.asarray(qpos_batch)
        if qpos_batch.shape[0] != 1:
            raise ValueError(f"{type(self).__name__} is single-world")
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
        self._scratch.qpos[:] = qpos_batch[0]
        mujoco.mj_forward(self.model, self._scratch)
        return self.evaluate_model_data(self.model, self._scratch)

    def chunk_reward(self) -> tuple[float, bool]:
        """Reward for the chunk since the last call, and whether the episode is over. Call after each chunk."""
        if self.target is None or self._phi_chunk is None:
            raise RuntimeError("no episode: evaluate (env.step) first")
        if self._failed:
            return -1.0, True
        if self._success:
            return 1.0, True
        phi = self._last.get("progress", self._phi_chunk)
        d = phi - self._phi_chunk
        self._phi_chunk = phi
        if self.reward_mode == "progress":
            return float(self.shaping_total * d / self.cfg.span[self.stage_name]), False
        eps = self.cfg.eps[self.stage_name]
        if self.reward_mode == "best":  # +1 only for a new best (by more than eps): wandering earns nothing
            if phi > self._phi_best + eps:
                self._phi_best = phi
                return 1.0, False
            return 0.0, False
        return (1.0 if d > eps else -1.0 if d < -eps else 0.0), False
