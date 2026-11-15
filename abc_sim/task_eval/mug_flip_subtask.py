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

RL reward per chunk: call ``chunk_reward()`` after each chunk. +1 on success and -1 on failure (both end the
episode); in between it depends on ``reward_mode``:
  "progress" (default)  shaping_total x (change in progress) / StageConfig.span[stage]. Potential-based shaping:
                        the in-between rewards sum to shaping_total x (progress gained) / span however many chunks
                        it takes, so going slowly or wobbling earns nothing extra. With shaping_total 0.5 a whole
                        stage of progress is worth ~0.5, less than the +1 for finishing.
  "sign"                +1 / -1 / 0 when progress rose / fell by more than the dead band StageConfig.eps. Each chunk
                        of progress earns a full +1, so slow progress can out-earn finishing: use with a discount
                        factor or a per-chunk cost.
  "best"                +1 only when progress beats its best so far by more than the dead band, else 0 (same caveat).
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.mug_flip_stages import (FALLEN, LABELS, PL, ROT, STAGE_PROMPTS, STAGES, MugScene,
                                               MugStageTracker, StageConfig, progress)
from abc_sim.task_eval.mugs import MugFlipUprightEvaluator
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


class MugFlipSubtaskEvaluator(SubtaskEvaluator):
    TASK, ENV_TASK, NAME = "turn_mug_right_side_up", "mug_flip", "mug_flip_subtask"  # registration: task_eval/subtasks.py
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = "mug"

    def make_config(self) -> StageConfig:
        return StageConfig()

    def make_scene(self, model: mujoco.MjModel) -> MugScene:
        return MugScene(model)

    def make_tracker(self, scene: MugScene) -> MugStageTracker:
        return MugStageTracker(scene, self.cfg)

    def make_stock(self, model: mujoco.MjModel) -> MugFlipUprightEvaluator:
        return MugFlipUprightEvaluator(model=model, spec=self.spec)  # stock success, for reference

    def progress(self, stage: int, obs: dict[str, np.ndarray]) -> np.ndarray:
        return progress(stage, obs)

    def step_failure(self, k: int, obs: dict[str, np.ndarray], label: int) -> str:
        # Let go in the air (rotate, place): a mug released mid-air can land upright inside one chunk, so per step.
        in_air = bool(~obs["touched"][k] & (obs["lift"][k] > self.cfg.rest_lift_m + self.drop_margin_m))
        return "dropped" if self.stage in (ROT, PL) and in_air else ""

    def extra_metrics(self, k: int, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"upright_z": float(obs["upz"][k]), "lift_m": float(obs["lift"][k]),
                "grasp_arm": int(obs["grasp_arm"][k]), "touched": bool(obs["touched"][k])}

    def sync_stock_to_episode(self) -> None:
        self.stock._rest_height = self.tracker.rest_height[None].astype(np.float32)  # the episode's, not the restored

    def build_start(self, env, seed: int, rng: np.random.Generator | None = None):
        from abc_sim.task_eval.mug_flip_starts import build_start, load_grasp_library

        if not hasattr(env, "_mug_grasp_library"):
            env._mug_grasp_library = load_grasp_library()
        return build_start(env, seed, self.stage_name, env._mug_grasp_library, rng=rng)

    @property
    def mug(self) -> int | None:
        return self.target
