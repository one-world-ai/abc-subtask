"""Evaluators for one open_drawer / close_drawer subtask on the prompted drawer, for single-stage RL.

Generic behaviour (latched failures, chunk reward modes) is ``stagekit.SubtaskEvaluator``; stages are
``drawer_stages``. Failures checked every control step, before the generic ones (back to an earlier stage):
  wrong drawer        another drawer opened more than ``other_max_m`` (stock 5 cm; approved D4), any stage
  pulled further open close_drawer: the target more than ``pulled_past_m`` (3 cm) past its episode start opening
                      (approved D5), any stage
"Back to an earlier stage" covers: pull - the hand left the drawer before 12 cm; release - pushed back below
11 cm (open) / pulled back above 2.5 cm (close); grasp / push - the hand moved away (> 5 cm, no contact).

The episode's target drawer comes from the randomization metadata (``configure_from_randomization``), like the stock
``DrawerOpenCloseEvaluator``, which is also run for ``stock_success`` and whose success is the pull / push goal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.drawer_stages import (CLOSE_LIVE, CLOSE_PROMPTS, CLOSE_STAGES, DRAWERS, JOINTS, OPEN_LIVE,
                                             OPEN_PROMPTS, OPEN_STAGES, DrawerScene, DrawerTracker, StageConfig,
                                             classify_close, classify_open, progress_close, progress_open)
from abc_sim.task_eval.openwam_tasks import DrawerOpenCloseEvaluator
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


def _metadata(randomization: object) -> Mapping:
    meta = getattr(randomization, "metadata", None)
    if meta is None and isinstance(randomization, Mapping):
        meta = randomization.get("metadata")
    return meta if isinstance(meta, Mapping) else {}


class _DrawerSubtaskEvaluator(SubtaskEvaluator):
    TARGET_KEY = "drawer"
    MODE = ""
    LIVE: dict[str, str] = {}

    def __init__(self, **kwargs) -> None:
        self.meta: Mapping = {}
        super().__init__(**kwargs)

    def make_config(self) -> StageConfig:
        return StageConfig()

    def make_scene(self, model: mujoco.MjModel) -> DrawerScene:
        return DrawerScene(model, self.cfg)

    def make_stock(self, model: mujoco.MjModel) -> DrawerOpenCloseEvaluator:
        c = self.cfg
        return DrawerOpenCloseEvaluator(model=model, spec=self.spec, mode=self.MODE, open_m=c.open_m,
                                        closed_m=c.closed_m, other_max_m=c.other_max_m)

    def configure_from_randomization(self, model: mujoco.MjModel, randomization: object) -> None:
        super().configure_from_randomization(model, randomization)
        meta = _metadata(randomization)
        if meta:
            self.meta = dict(meta)
        self.stock.configure_from_randomization(model, randomization)

    def target_index(self) -> int:
        if "target_drawer" not in self.meta:
            raise RuntimeError("drawer subtask: no target drawer yet (reset the env with randomize=True)")
        return DRAWERS.index(self.meta["target_drawer"])

    def start_opening(self) -> np.ndarray:
        """Opening of each drawer at the episode's (stock) start, from the randomizer."""
        st = self.meta.get("drawer_start", {})
        return np.asarray([float(st.get(JOINTS[d], 0.0)) for d in DRAWERS])

    def pick_target(self, obs: dict[str, np.ndarray]) -> int:
        return self.target_index()

    def step_failure(self, k: int, obs: dict[str, np.ndarray], label: int) -> str:
        if obs["others"][k] > self.cfg.other_max_m:
            return "wrong drawer opened"
        return ""

    def extra_metrics(self, k: int, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"target_drawer": DRAWERS[k], "q_m": float(obs["q"][k]), "others_max_q_m": float(obs["others"][k]),
                "pinch_bar_m": float(obs["dist"][k]), "hand_bar_m": float(obs["hand_dist"][k]), "near_arm": int(obs["near_arm"][k]),
                "fingers_on_handle": int(obs["fingers"][k]), "grasp_arm": int(obs["grasp_arm"][k]),
                "hand_on": bool(obs["hand_on"][k]), "released": bool(obs["released"][k])}

    def stage_prompt(self) -> str:
        target = self.meta.get("target_drawer")
        return self.LIVE[self.stage_name].format(target=target) if target else self.STAGE_PROMPTS[self.stage_name]

    def build_start(self, env, seed: int, rng: np.random.Generator | None = None):
        from abc_sim.task_eval.drawer_starts import build_start

        return build_start(env, seed, self.MODE, self.stage_name, rng=rng)


class DrawerOpenSubtaskEvaluator(_DrawerSubtaskEvaluator):
    TASK, ENV_TASK, NAME = "open_drawer", "drawer_open", "open_drawer_subtask"  # registration: task_eval/subtasks.py
    STAGES = OPEN_STAGES
    FALLEN = len(OPEN_STAGES)  # never produced: drawers cannot fall
    LABELS = OPEN_STAGES + ("fallen",)
    STAGE_PROMPTS = OPEN_PROMPTS
    LIVE = OPEN_LIVE
    MODE = "open"

    def make_tracker(self, scene: DrawerScene) -> DrawerTracker:
        return DrawerTracker(scene, classify_open, self.cfg)

    def progress(self, stage: int, obs: dict[str, np.ndarray]) -> np.ndarray:
        return progress_open(stage, obs, self.cfg)


class DrawerCloseSubtaskEvaluator(_DrawerSubtaskEvaluator):
    TASK, ENV_TASK, NAME = "close_drawer", "drawer_close", "close_drawer_subtask"  # registration: task_eval/subtasks.py
    STAGES = CLOSE_STAGES
    FALLEN = len(CLOSE_STAGES)
    LABELS = CLOSE_STAGES + ("fallen",)
    STAGE_PROMPTS = CLOSE_PROMPTS
    LIVE = CLOSE_LIVE
    MODE = "close"

    def make_tracker(self, scene: DrawerScene) -> DrawerTracker:
        return DrawerTracker(scene, classify_close, self.cfg)

    def progress(self, stage: int, obs: dict[str, np.ndarray]) -> np.ndarray:
        return progress_close(stage, obs, self.cfg, self.start_opening())

    def step_failure(self, k: int, obs: dict[str, np.ndarray], label: int) -> str:
        why = super().step_failure(k, obs, label)
        if not why and obs["q"][k] > self.start_opening()[k] + self.cfg.pulled_past_m:
            why = "pulled further open"
        return why


__all__ = ["DrawerOpenSubtaskEvaluator", "DrawerCloseSubtaskEvaluator"]
