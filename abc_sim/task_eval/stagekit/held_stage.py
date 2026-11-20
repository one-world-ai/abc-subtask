"""Single-stage tasks that begin at a verified held-object snapshot."""

import os
from pathlib import Path

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching
from abc_sim.task_eval.stagekit.starts import ENV_ROOT, load_starts
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator


class HeldStage(SubtaskEvaluator):
    """Share snapshot validation, physical-time debounce and sparse terminal rewards."""

    TARGET_KEY = "target"
    HOLD_SECONDS = 0.5
    BUDGET_SECONDS = 12.0

    def __init__(self, *, model, spec, stage):
        self.spec, self.stage_name = spec, stage
        self._bind(model)

    def _bind(self, model):
        self.model = model
        self.scene = ObjectScene(model, free_joints_matching(model, self.object_pattern))
        self.stock = self.make_stock(model)
        self.goal_joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.velocity_indices = model.jnt_dofadr[self.goal_joints, None] + np.arange(6)
        self.reset()

    def reset(self, *, nworld=1):
        self.target = None
        self._failed, self._success, self._last = "", False, {}
        self.stable_since = None
        self.stock.reset(nworld=nworld)

    def valid_start(self, data, target, rest):
        """Require a slow, lifted two-finger grasp and the task's initial object arrangement."""
        obs = self.scene.observe(data, rest)
        dof = self.scene.dof_adr[target]
        return (obs["grasped"][target] and obs["lift"][target] >= 0.03
                and np.linalg.norm(data.qvel[dof:dof+3]) < 0.05
                and self.initial_arrangement(data, target))

    def begin(self, model, data, target, rest_height=None, labels=None):
        if model is not self.model:
            self._bind(model)
        self.reset()
        if not self.valid_start(data, target, rest_height):
            raise ValueError("Invalid held-object start")
        self.target = target
        self.started = data.time
        self.init_events(data)

    def build_start(self, env, seed, rng=None):
        """Resolve a saved seed deterministically, or sample among its grasps with an explicit RNG."""
        default = Path(__file__).resolve().parents[3] / "subtask_rewards/held_starts"
        rows = load_starts(os.environ.get(ENV_ROOT, default), self.TASK, self.stage_name)
        matches = [row for row in rows if row["seed"] == seed]
        if not matches:
            return None, [{"seed": seed, "reason": "no saved held-object start"}]
        return matches[0 if rng is None else int(rng.integers(len(matches)))], []

    def next_stage(self):
        return None

    def evaluate_qpos_batch(self, qpos_batch):
        raise ValueError("Held stages need time, contacts and velocities")

    def check_physics_step(self, data):
        """Discard the episode on the first excessive goal-object velocity."""
        if self.target is None or self.failure or self.success:
            return
        vel = data.qvel[self.velocity_indices]
        if (not np.isfinite(vel).all() or (np.linalg.norm(vel[:, :3], axis=1) > 4).any()
                or (np.linalg.norm(vel[:, 3:], axis=1) > 60).any()):
            self._failed = "sim_fault"

    def evaluate_model_data(self, model, data):
        """Latch failures, or success after the goal holds continuously in simulated time."""
        if model is not self.model:
            self._bind(model)
        if self.target is not None and not (self.failure or self.success):
            self.check_physics_step(data)
            if not self.failure:
                good, self._last = self.observe_goal(data)
                if not good:
                    self.stable_since = None
                elif self.stable_since is None:
                    self.stable_since = data.time
                if self.stable_since is not None and data.time - self.stable_since >= self.HOLD_SECONDS:
                    self._success = True
                elif data.time - self.started >= self.BUDGET_SECONDS:
                    self._failed = "timeout"
        reward, done = self.chunk_reward()
        self._last.update(episode_done=done, terminal=done, failed=self.failure, sim_fault=self.failure == "sim_fault")
        return TaskEvalResult(reward=np.array([reward]), success=np.array([self.success]), metrics=dict(self._last))

    def chunk_reward(self):
        return float(int(self.success) - int(bool(self.failure))), bool(self.success or self.failure)
