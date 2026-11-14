"""Evaluator for one put_relative subtask (reach / pick / handoff / carry / place) on the episode's mover, for
single-stage RL.

Generic behaviour (latched failures, chunk reward modes) is ``stagekit.SubtaskEvaluator``; stages are
``put_relative_stages``. Failures checked every physics step, before the generic ones (off the table, back to an
earlier stage):
  moved the reference   the reference object moved more than ``ref_moved_m`` from its rest spot (any stage)
  dropped               carry / place: no finger on the mover while it is more than rest_lift_m + drop_margin_m above
                        its rest height (let go in the air; "put", not "throw")
  pushed                reach / pick: the mover moved more than ``push_m`` while not held, or reached the goal zone
                        without being carried there
Cross-table episodes (giver G = the arm on the mover's side, receiver V; ``put_relative_stages``):
  handoff               ``handoff_stage.failure``: dropped (free and > 4 cm up); touched anything but the two arms
                        (table, other objects: the handoff happens in the air); giver carried it into the goal zone
                        itself (G touches it while its xy is in the zone)
  carry / place         giver touched the mover again (after the handoff only V may touch it)

The episode's mover, reference and direction come from the randomization metadata (``configure_from_randomization``),
like the stock ``PutRelativeEvaluator``, which is also run for ``stock_success``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_eval.put_relative_stages import (C, FALLEN, LABELS, H, P, PL, R, STAGE_PROMPTS, STAGES, DONE,
                                                   PutRelativeScene, PutRelativeTracker, StageConfig, live_prompt,
                                                   progress)
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


def goal_from_randomization(randomization: object) -> dict | None:
    metadata = getattr(randomization, "metadata", None)
    if metadata is None and isinstance(randomization, Mapping):
        metadata = randomization.get("metadata")
    if not isinstance(metadata, Mapping) or "mover_object" not in metadata:
        return None
    offset = np.asarray(metadata["direction_offset_xy"], dtype=np.float64)
    return {"mover": str(metadata["mover_object"]), "reference": str(metadata["reference_object"]),
            "axis": offset / np.linalg.norm(offset), "direction": str(metadata.get("direction", "")),
            "mover_label": str(metadata.get("mover_label", "object")),
            "reference_label": str(metadata.get("reference_label", "object"))}


class PutRelativeSubtaskEvaluator(SubtaskEvaluator):
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = "target"

    def __init__(self, **kwargs) -> None:
        self.goal: dict | None = None
        super().__init__(**kwargs)

    def make_config(self) -> StageConfig:
        return StageConfig()

    def make_scene(self, model: mujoco.MjModel) -> PutRelativeScene:
        scene = PutRelativeScene(model, self.cfg)
        if self.goal is not None:
            scene.set_goal(self.goal)
        return scene

    def make_tracker(self, scene: PutRelativeScene) -> PutRelativeTracker:
        return PutRelativeTracker(scene, self.cfg)

    def make_stock(self, model: mujoco.MjModel) -> PutRelativeEvaluator:
        return PutRelativeEvaluator(model=model, spec=self.spec)

    def configure_from_randomization(self, model: mujoco.MjModel, randomization: object) -> None:
        super().configure_from_randomization(model, randomization)
        goal = goal_from_randomization(randomization)
        if goal is not None:
            self.goal = goal
            self.scene.set_goal(goal)
        self.stock.configure_from_randomization(model, randomization)

    def pick_target(self, obs: dict[str, np.ndarray]) -> int:
        if self.scene.mover < 0:
            raise RuntimeError("put_relative subtask: no directive yet (reset the env with randomize=True)")
        return self.scene.mover

    def begin(self, model, data, target, rest_height=None, labels=None) -> None:
        super().begin(model, data, target, rest_height=rest_height, labels=labels)
        self._start_xy = np.array(self.tracker.obs["xy"][self.target], dtype=np.float64)

    def progress(self, stage: int, obs: dict[str, np.ndarray]) -> np.ndarray:
        return progress(stage, obs, self.cfg)

    def step_failure(self, k: int, obs: dict[str, np.ndarray], label: int) -> str:
        cfg = self.cfg
        if obs["ref_moved"] > cfg.ref_moved_m:
            return "moved the reference"
        held_up = bool(obs["grasped"][k] and obs["lift"][k] >= cfg.lift_m)
        cross = bool(obs["cross"][k]) if "cross" in obs else False
        if self.stage == H:
            why = hs.failure(obs, k, cfg.handoff) if cross else ""
            if not why and cross and obs["touch_give"][k] and obs["gap"][k] <= 0.0:
                why = "giver carried it to the goal"
            return why
        if cross and self.stage in (C, PL) and obs["touch_give"][k]:
            return "giver touched the mover again"
        if self.stage in (C, PL):
            if not obs["touched"][k] and obs["lift"][k] > cfg.rest_lift_m + self.drop_margin_m:
                return "dropped"
        else:  # reach, pick
            if label in (PL, DONE) and not held_up:
                return "pushed"
            moved = float(np.linalg.norm(obs["xy"][k] - self._start_xy))
            if moved > cfg.push_m and not obs["grasped"][k] and obs["lift"][k] < cfg.lift_m:
                return "pushed"
        return ""

    def extra_metrics(self, k: int, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        out = {"lift_m": float(obs["lift"][k]), "grasp_arm": int(obs["grasp_arm"][k]),
               "touched": bool(obs["touched"][k]), "zone_gap_m": float(obs["gap"][k]),
               "along_m": float(obs["along"][k]), "perp_m": float(obs["perp"][k]),
               "ref_moved_m": float(obs["ref_moved"]), "speed": float(obs["speed"][k])}
        if "cross" in obs:
            out["cross"] = bool(obs["cross"][k])
        if "f_give" in obs:
            out.update(f_give=int(obs["f_give"][k]), f_recv=int(obs["f_recv"][k]),
                       touch_give=bool(obs["touch_give"][k]), touch_recv=bool(obs["touch_recv"][k]),
                       gap_give_m=float(obs["gap_give"][k]), env_touch=bool(obs["env_touch"][k]),
                       middle_gap_m=float(obs["mid_gap"][k]))
        return out

    def stage_prompt(self) -> str:
        return live_prompt(self.stage_name, self.goal or {})

    def build_start(self, env, seed: int, rng: np.random.Generator | None = None):
        from abc_sim.task_eval.put_relative_starts import build_start

        return build_start(env, seed, self.stage_name, rng=rng)


__all__ = ["PutRelativeSubtaskEvaluator", "goal_from_randomization", "R", "P", "H", "C", "PL", "DONE"]
