"""Staged (partial-credit) reward for turn_mug_right_side_up, for RL.

The stock ``MugFlipUprightEvaluator`` pays only for mugs that are already upright and set down. This evaluator keeps
its success flag unchanged and replaces the reward with a staged one. Each mug moves through six stages, and the reward
is the mean per-mug value:

    stage     weight  counts when
    reach     0.05    closest grasp site approaches the mug (shaped from the start distance; max so far)
    grasp     0.15    both fingers of one gripper touch the mug (latched)
    lift      0.10    the mug is grasped and >= lift_m above its rest height (latched)
    rotate    0.30    tilt progress (upright_z - start) / (upright_z_min - start), counted only after a grasp and
                      while a finger touches the mug (max so far)
    place     0.25    the stock per-mug condition: upright and <= max_lift_m above rest (live, not latched)
    release   0.15    placed and no finger touching the mug (live)

A placed mug implies the earlier stages, so a mug flipped without a lift still gets them. Place and release are
live: a mug knocked back over loses them. Reach, grasp, lift and rotate never go down, so grasping and dropping
repeatedly earns nothing extra. As in put_relative, the reward is 1.0 exactly when the stock check succeeds and
is capped at 0.99 otherwise.

The env scores every physics step (``env.step`` calls the evaluator), so latches see every step.
Contacts come from MjData: ``evaluate_model_data`` (the single-world CPU env calls it) uses the live data, and
``evaluate_qpos_batch`` runs ``mj_forward`` on a scratch MjData for each row, as dishrack.py does.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.mugs import MugFlipUprightEvaluator
from abc_sim.task_specs import SimTaskSpec

STAGES = ("reach", "grasp", "lift", "rotate", "place", "release", "done")
DEFAULT_WEIGHTS = {"reach": 0.05, "grasp": 0.15, "lift": 0.10, "rotate": 0.30, "place": 0.25, "release": 0.15}
# Text for the stage a mug is working on (index = number of stages completed). Only a hint in the metrics: the
# reward never depends on the prompt.
STAGE_PROMPTS = (
    "reach for the mug",
    "grasp the mug",
    "lift the mug",
    "turn the mug right side up",
    "put the mug down upright",
    "release the mug",
    "done",
)
ARMS = ("left", "right")


def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in out:
            out.add(body)
    return out


def _geoms_of(model: mujoco.MjModel, bodies: set[int]) -> np.ndarray:
    return np.asarray([g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies], dtype=np.int32)


class MugFlipStagedEvaluator:
    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        upright_z_min: float = 0.95,
        max_lift_m: float = 0.03,
        lift_m: float = 0.03,
        reach_done_m: float = 0.04,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.spec = spec
        self.upright_z_min = float(upright_z_min)
        self.max_lift_m = float(max_lift_m)
        self.lift_m = float(lift_m)
        self.reach_done_m = float(reach_done_m)
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"stage weights must sum to 1, got {self.weights}")
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.base = MugFlipUprightEvaluator(
            model=model, spec=self.spec, upright_z_min=self.upright_z_min, max_lift_m=self.max_lift_m
        )
        self.mug_names = list(self.base.mug_names)
        self._mug_bodies = []
        self._mug_geoms = []
        for name in self.mug_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_jnt")
            body = int(model.jnt_bodyid[joint])
            self._mug_bodies.append(body)
            self._mug_geoms.append(_geoms_of(model, _subtree(model, body)))
        # Finger geoms: (arm, side) -> geom ids; side 0 = link_left_finger subtree, 1 = link_right_finger subtree.
        self._finger_geoms: dict[tuple[int, int], np.ndarray] = {}
        for a, arm in enumerate(ARMS):
            for s, side in enumerate(("link_left_finger", "link_right_finger")):
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_{side}")
                if body < 0:
                    raise ValueError(f"body {arm}_{side} not found")
                self._finger_geoms[(a, s)] = _geoms_of(model, _subtree(model, body))
        self._grasp_sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grasp_site") for arm in ARMS]
        if min(self._grasp_sites) < 0:
            raise ValueError("left/right_grasp_site not found")
        # geom id -> (kind, index): kind 0 = mug k, kind 1 = finger (arm*2 + side).
        self._geom_owner = np.full((model.ngeom, 2), -1, dtype=np.int32)
        for k, geoms in enumerate(self._mug_geoms):
            self._geom_owner[geoms] = (0, k)
        for (a, s), geoms in self._finger_geoms.items():
            self._geom_owner[geoms] = (1, a * 2 + s)
        self._scratch: mujoco.MjData | None = None

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self.base.reset(nworld=self._nworld)
        n, m = self._nworld, len(self.mug_names)
        self._start_dist = np.full((n, m), np.nan, dtype=np.float32)
        self._start_upz = np.full((n, m), np.nan, dtype=np.float32)
        self._reach = np.zeros((n, m), dtype=np.float32)
        self._grasped_ever = np.zeros((n, m), dtype=bool)
        self._lifted_ever = np.zeros((n, m), dtype=bool)
        self._rotate = np.zeros((n, m), dtype=np.float32)
        self._max_reward = np.zeros((n,), dtype=np.float32)

    def configure_from_randomization(self, model: mujoco.MjModel, randomization: object) -> None:
        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

    # -- per-world geometry ------------------------------------------------------------------------------------------

    def _contacts(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(grasped[m], touched[m]): both fingers of one arm on mug k / any finger on mug k."""
        m = len(self.mug_names)
        hits = np.zeros((m, 4), dtype=bool)  # mug k touched by finger (arm*2 + side)
        for i in range(int(data.ncon)):
            c = data.contact[i]
            o1, o2 = self._geom_owner[int(c.geom1)], self._geom_owner[int(c.geom2)]
            if o1[0] == 0 and o2[0] == 1:
                hits[o1[1], o2[1]] = True
            elif o1[0] == 1 and o2[0] == 0:
                hits[o2[1], o1[1]] = True
        grasped = (hits[:, 0] & hits[:, 1]) | (hits[:, 2] & hits[:, 3])
        return grasped, hits.any(axis=1)

    def _site_dist(self, data: mujoco.MjData) -> np.ndarray:
        sites = np.asarray(data.site_xpos[self._grasp_sites], dtype=np.float64)  # (2, 3)
        mugs = np.asarray(data.subtree_com[self._mug_bodies], dtype=np.float64)  # mug CoM (top body is massless)
        return np.linalg.norm(mugs[:, None, :] - sites[None, :, :], axis=-1).min(axis=1).astype(np.float32)

    # -- evaluation --------------------------------------------------------------------------------------------------

    def evaluate_model_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> TaskEvalResult:
        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=1)
        if self._nworld != 1:
            self.reset(nworld=1)
        qpos = np.asarray(data.qpos, dtype=np.float32)[None, :]
        grasped, touched = self._contacts(data)
        return self._score(qpos, grasped[None], touched[None], self._site_dist(data)[None])

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}")
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
        d = self._scratch
        grasped, touched, dist = [], [], []
        for row in qpos_batch:
            d.qpos[:] = row
            mujoco.mj_forward(self.model, d)
            g, t = self._contacts(d)
            grasped.append(g)
            touched.append(t)
            dist.append(self._site_dist(d))
        return self._score(qpos_batch, np.asarray(grasped), np.asarray(touched), np.asarray(dist))

    def _score(self, qpos: np.ndarray, grasped: np.ndarray, touched: np.ndarray, dist: np.ndarray) -> TaskEvalResult:
        base = self.base.evaluate_qpos_batch(qpos)
        b, m = qpos.shape[0], len(self.mug_names)
        if m == 0:
            return TaskEvalResult(reward=base.reward, success=base.success, metrics=dict(base.metrics))
        upz = np.asarray(base.metrics["mug_upright_z"], dtype=np.float32)
        lift = np.asarray(base.metrics["mug_lift_m"], dtype=np.float32)
        placed = np.asarray(base.metrics["mug_upright_mask"], dtype=bool)

        first = np.isnan(self._start_dist)
        self._start_dist[first] = dist[first]
        self._start_upz[first] = upz[first]

        span = np.maximum(self._start_dist - self.reach_done_m, 1e-3)
        reach_now = np.clip((self._start_dist - dist) / span, 0.0, 1.0)
        self._reach = np.maximum(self._reach, reach_now)
        self._grasped_ever |= grasped
        self._lifted_ever |= grasped & (lift >= self.lift_m)
        rot_span = self.upright_z_min - self._start_upz
        rot_now = np.where(rot_span > 1e-3, np.clip((upz - self._start_upz) / np.maximum(rot_span, 1e-3), 0, 1), 1.0)
        # Only while a finger touches the mug: a dropped mug tumbling through upright must not latch the credit.
        self._rotate = np.maximum(self._rotate, np.where(self._grasped_ever & touched, rot_now, 0.0))
        released = placed & ~touched

        w = self.weights
        reach = np.where(placed | self._grasped_ever, 1.0, self._reach)
        grasp = (placed | self._grasped_ever).astype(np.float32)
        lifted = (placed | self._lifted_ever).astype(np.float32)
        rotate = np.where(placed, 1.0, self._rotate)
        per_mug = (w["reach"] * reach + w["grasp"] * grasp + w["lift"] * lifted + w["rotate"] * rotate
                   + w["place"] * placed + w["release"] * released).astype(np.float32)

        success = base.success
        reward = np.where(success, 1.0, np.minimum(per_mug.mean(axis=1), 0.99)).astype(np.float32)
        self._max_reward = np.maximum(self._max_reward, reward)

        # Stage index per mug = number of stages completed in order (6 = done).
        done = np.stack([reach >= 1.0, grasp > 0, lifted > 0, rotate >= 1.0, placed, released], axis=-1)
        stage = np.argmin(np.concatenate([done, np.zeros((b, m, 1), bool)], axis=-1), axis=-1).astype(np.int32)
        stage[done.all(axis=-1)] = 6
        active = [int(np.argmin(np.where(s == 6, 99, s))) if (s < 6).any() else 0 for s in stage]
        metrics: dict[str, Any] = dict(base.metrics)
        metrics.update({
            "base_reward": base.reward,
            "mug_stage": stage,
            "mug_grasped": grasped,
            "mug_touched": touched,
            "mug_released": released,
            "mug_reach": reach.astype(np.float32),
            "mug_rotate": rotate.astype(np.float32),
            "mug_staged_value": per_mug,
            "grasp_site_dist_m": dist.astype(np.float32),
            "active_stage": [STAGES[stage[i, active[i]]] for i in range(b)],
            "stage_prompt": [STAGE_PROMPTS[stage[i, active[i]]] for i in range(b)],
            "max_reward_so_far": self._max_reward.copy(),
        })
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)
