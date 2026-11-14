"""Evaluator for one inhand_transfer subtask (reach / pick / present / handoff / carry / place), for single-stage RL.

Generic behaviour (latched failures, chunk reward modes) is ``stagekit.SubtaskEvaluator``; stages are
``inhand_stages`` (the handoff stage's tests are the task-independent ``handoff_stage``). Failures checked every
physics step before the generic ones (off the table, back to an earlier stage), per the approved stage table:

  reach, pick   pushed: the item moved > push_m (xy) while A does not hold it with both fingers
  reach         touched the item with a closed gripper (A's fingers < 50 % open; G5 repair, not in the table)
                receiver touched it: any B geom on the item (decision 6: the spawn-side arm gives)
  present       dropped: no robot touches it while > 4 cm above rest
                carried past the middle: A touches it while its origin is in the far band (decision 1: the transfer
                is required)
  handoff       ``handoff_stage.failure``: dropped; touched the table (decision 2: in the air only)
  carry         dropped (any release > 4 cm up, before reaching place); A touches it again
  place         A touches it again; released too high (decision 3: lowest point > 5 cm above the table when the
                robot lets go and nothing else supports it); landed outside the far band (released, at rest, not
                the stock success)

The episode's side comes from the randomization metadata (``configure_from_randomization``); the fixed stock scorer
(``InhandTransferOtherSideEvaluator``) is the place goal and is reported as ``stock_success``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.inhand_stages import (C, DONE, FALLEN, H, LABELS, P, PL, PR, R, STAGE_PROMPTS, STAGES,
                                             InhandScene, InhandTracker, StageConfig, progress)
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


def side_of(randomization: object) -> str | None:
    metadata = getattr(randomization, "metadata", None)
    if metadata is None and isinstance(randomization, Mapping):
        metadata = randomization.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    side = metadata.get("side")
    return side if side in ("left", "right") else None


class InhandSubtaskEvaluator(SubtaskEvaluator):
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = "target"

    def __init__(self, **kwargs) -> None:
        self._side: str | None = None
        self._randomization = None
        super().__init__(**kwargs)

    def make_config(self) -> StageConfig:
        return StageConfig()

    def make_scene(self, model: mujoco.MjModel) -> InhandScene:
        scene = InhandScene(model, self.cfg)
        if self._side is not None:
            scene.set_side(self._side, self._randomization)
        return scene

    def make_tracker(self, scene: InhandScene) -> InhandTracker:
        return InhandTracker(scene, self.cfg)

    def configure_from_randomization(self, model: mujoco.MjModel, randomization: object) -> None:
        side = side_of(randomization)
        if side is not None:
            self._side, self._randomization = side, randomization
        super().configure_from_randomization(model, randomization)
        if side is not None:
            self.scene.set_side(side, randomization)

    @property
    def giver(self) -> int:
        return self.scene.giver

    def pick_target(self, obs: dict[str, np.ndarray]) -> int:
        return 0

    def begin(self, model, data, target, rest_height=None, labels=None) -> None:
        super().begin(model, data, target, rest_height=rest_height, labels=labels)
        o = self.tracker.obs
        self._start_xy = np.array(o["xy"][0], dtype=np.float64)
        self._was_touched = bool(o["touch_give"][0] or o["touch_recv"][0])
        self._released = False

    def progress(self, stage: int, obs: dict[str, np.ndarray]) -> np.ndarray:
        return progress(stage, obs, self.cfg)

    def step_failure(self, k: int, obs: dict[str, np.ndarray], label: int) -> str:
        cfg, st = self.cfg, self.stage
        touched = bool(obs["touch_give"][k] or obs["touch_recv"][k])
        free_up = not touched and obs["lift"][k] > cfg.handoff.drop_m
        why = ""
        if st in (R, P):
            if obs["touch_recv"][k]:
                why = "receiver arm touched the item"
            elif obs["f_give"][k] < 2 and np.linalg.norm(obs["xy"][k] - self._start_xy) > cfg.push_m:
                why = "pushed"
            elif st == R and obs["f_give"][k] > 0 and obs["open_give"][k] < cfg.closed_open_frac:
                # Reach means "ready to grasp". Without this, a closed fist bumping the item (the push hack) met the
                # touch clause of the reach goal: +1 on 17/20 (G5 round 0).
                why = "touched the item with a closed gripper"
        elif st == PR:
            if free_up:
                why = "dropped"
            elif obs["touch_give"][k] and obs["ys"][k] <= -cfg.band_m:
                why = "giver carried it past the middle"
        elif st == H:
            why = hs.failure(obs, k, cfg.handoff)
        elif st == C:
            if obs["touch_give"][k]:
                why = "giver touched the item again"
            elif free_up:
                why = "dropped"
        elif st == PL:
            if obs["touch_give"][k]:
                why = "giver touched the item again"
            elif self._was_touched and not touched and not obs["env_touch"][k] and obs["low"][k] > cfg.release_low_m:
                why = "released too high"
            else:
                if self._was_touched and not touched:
                    self._released = True
                if (self._released and not touched and obs["env_touch"][k] and obs["speed"][k] <= cfg.at_rest_speed
                        and not obs["stock_done"][k]):
                    why = "landed outside the far band"
        self._was_touched = touched
        return why

    def extra_metrics(self, k: int, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"lift_m": float(obs["lift"][k]), "giver": int(self.scene.giver), "f_give": int(obs["f_give"][k]),
                "f_recv": int(obs["f_recv"][k]), "touch_give": bool(obs["touch_give"][k]),
                "touch_recv": bool(obs["touch_recv"][k]), "env_touch": bool(obs["env_touch"][k]),
                "gap_give_m": float(obs["gap_give"][k]), "gap_recv_m": float(obs["gap_recv"][k]),
                "middle_gap_m": float(obs["m_gap"][k]), "far_gap_m": float(obs["far_gap"][k]),
                "low_m": float(obs["low"][k]), "speed": float(obs["speed"][k]),
                "stock_success": bool(obs["stock_done"][k])}

    def build_start(self, env, seed: int, rng: np.random.Generator | None = None):
        from abc_sim.task_eval.inhand_starts import build_start

        return build_start(env, seed, self.stage_name, rng=rng)


__all__ = ["InhandSubtaskEvaluator", "R", "P", "PR", "H", "C", "PL", "DONE"]
