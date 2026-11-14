"""In-hand transfer task evaluator."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class InhandTransferOtherSideEvaluator:
    """Score the item resting on the table opposite its sampled spawn side.

    The item must cross the configured y band, remain within the tabletop,
    and sit below the resting-height limit. The start side comes from reset
    metadata.

    It must also be released and at rest (phase-0 audit, subtask_rewards/inhand):
    no robot geom may touch it, and with MjData (``evaluate_model_data``, the
    path ``env.evaluate_task`` takes) its linear speed must be below
    ``max_rest_speed_mps``. Without these, an item still held low over the far
    side, or flying/sliding through the band, scored success and ended the
    episode. The qpos-only path (``evaluate_qpos_batch``) recomputes contacts
    with ``mj_forward`` but has no velocities, so it cannot check rest.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        object_joint_name: str = "task_object_joint",
        table_geom_name: str = "table_plane",
        other_side_min_y_m: float = 0.10,
        max_resting_height_m: float = 0.10,
        max_rest_speed_mps: float = 0.02,
    ) -> None:
        self.spec = spec
        self._object_joint_name = object_joint_name
        self._table_geom_name = table_geom_name
        self.other_side_min_y_m = float(other_side_min_y_m)
        self.max_resting_height_m = float(max_resting_height_m)
        self.max_rest_speed_mps = float(max_rest_speed_mps)
        self._start_sign: float | None = None
        self._spawn_y: float | None = None
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._start_sign = None
        self._spawn_y = None
        self._object_adr = self._resolve_free_joint_qpos_adr(
            model, self._object_joint_name
        )
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._object_joint_name)
        self._object_dof_adr = int(model.jnt_dofadr[joint_id])
        self._object_geom_mask = self._subtree_geom_mask(model, [int(model.jnt_bodyid[joint_id])])
        robot_roots = [
            body
            for body in (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in ("left_arm", "right_arm")
            )
            if body >= 0
        ]
        if len(robot_roots) != 2:
            raise ValueError("left_arm/right_arm bodies not found; cannot check that the item is released")
        self._robot_geom_mask = self._subtree_geom_mask(model, robot_roots)
        self._scratch_data: mujoco.MjData | None = None
        (
            self._table_x_bounds,
            self._table_y_bounds,
            self._table_z,
        ) = self._resolve_table(model, self._table_geom_name)

    @staticmethod
    def _subtree_geom_mask(model: mujoco.MjModel, roots: list[int]) -> np.ndarray:
        in_tree = np.zeros(model.nbody, dtype=bool)
        in_tree[roots] = True
        for body in range(1, model.nbody):  # parents precede children
            if in_tree[int(model.body_parentid[body])]:
                in_tree[body] = True
        return in_tree[np.asarray(model.geom_bodyid)]

    def _robot_touches_object(self, data: mujoco.MjData) -> bool:
        if int(data.ncon) == 0:
            return False
        g1 = np.asarray(data.contact.geom1[: data.ncon])
        g2 = np.asarray(data.contact.geom2[: data.ncon])
        obj, rob = self._object_geom_mask, self._robot_geom_mask
        return bool(np.any((obj[g1] & rob[g2]) | (obj[g2] & rob[g1])))

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Adopt the episode's start side (and any reloaded model)."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)

        metadata = getattr(randomization, "metadata", None)
        if metadata is None and isinstance(randomization, Mapping):
            metadata = randomization.get("metadata")
        if not isinstance(metadata, Mapping):
            return
        side = metadata.get("side")
        if side not in ("left", "right"):
            return
        # The randomizer's "left" is +y (the left arm's side of the table).
        self._start_sign = 1.0 if side == "left" else -1.0

        object_states = getattr(randomization, "object_states", None)
        if object_states is None and isinstance(randomization, Mapping):
            object_states = randomization.get("object_states")
        self._spawn_y = None
        if isinstance(object_states, Mapping):
            state = object_states.get(self._object_joint_name)
            if isinstance(state, Mapping) and "pos" in state:
                self._spawn_y = float(state["pos"][1])

    def debug_spec(self) -> dict[str, object] | None:
        return None

    @staticmethod
    def _resolve_free_joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_table(
        model: mujoco.MjModel,
        table_geom_name: str,
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, table_geom_name)
        if geom_id < 0:
            raise ValueError(f"Geom {table_geom_name!r} not found in model")
        if int(model.geom_bodyid[geom_id]) != 0:
            raise ValueError(
                f"Geom {table_geom_name!r} must live on the world body; the "
                "world-frame side test assumes a static table"
            )
        pos = np.asarray(model.geom_pos[geom_id], dtype=np.float64)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        return (
            (float(pos[0] - size[0]), float(pos[0] + size[0])),
            (float(pos[1] - size[1]), float(pos[1] + size[1])),
            float(pos[2]),
        )

    def evaluate_model_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> TaskEvalResult:
        """Score the live simulation: contacts and velocities come from ``data``."""
        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)
        qpos = np.asarray(data.qpos, dtype=np.float32)[None, :]
        touching = np.asarray([self._robot_touches_object(data)])
        dof = self._object_dof_adr
        speed = np.asarray([float(np.linalg.norm(data.qvel[dof : dof + 3]))], dtype=np.float32)
        return self._evaluate(qpos, touching, speed)

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        touching = np.zeros((qpos_batch.shape[0],), dtype=bool)
        if self._start_sign is not None:
            if self._scratch_data is None:
                self._scratch_data = mujoco.MjData(self.model)
            for world, qpos in enumerate(qpos_batch):
                self._scratch_data.qpos[:] = qpos
                mujoco.mj_forward(self.model, self._scratch_data)
                touching[world] = self._robot_touches_object(self._scratch_data)
        return self._evaluate(qpos_batch, touching, None)

    def _evaluate(
        self,
        qpos_batch: np.ndarray,
        robot_touching: np.ndarray,
        speed: np.ndarray | None,
    ) -> TaskEvalResult:
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        batch = qpos_batch.shape[0]
        if self._start_sign is None:
            zeros = np.zeros((batch,), dtype=np.float32)
            return TaskEvalResult(
                reward=zeros,
                success=np.zeros((batch,), dtype=bool),
                metrics={
                    "goal_configured": False,
                    "ever_success": self._ever_success.copy(),
                },
            )

        pos = qpos_batch[:, self._object_adr : self._object_adr + 3]
        sign = self._start_sign
        crossed = (pos[:, 1] * sign) <= -self.other_side_min_y_m
        placed = pos[:, 2] <= self._table_z + self.max_resting_height_m
        x_lo, x_hi = self._table_x_bounds
        y_lo, y_hi = self._table_y_bounds
        on_table = (
            (pos[:, 0] >= x_lo)
            & (pos[:, 0] <= x_hi)
            & (pos[:, 1] >= y_lo)
            & (pos[:, 1] <= y_hi)
        )
        released = ~np.asarray(robot_touching, dtype=bool)
        at_rest = (
            np.ones((batch,), dtype=bool)
            if speed is None
            else np.asarray(speed) <= self.max_rest_speed_mps
        )
        success = crossed & placed & on_table & released & at_rest
        self._ever_success |= success

        # Progress: the object's signed travel from its spawn y to the far
        # band's near edge; latched success keeps a later bump from erasing it.
        spawn_y = self._spawn_y if self._spawn_y is not None else sign * 0.25
        span = max(spawn_y * sign + self.other_side_min_y_m, 1e-6)
        travel = (spawn_y - pos[:, 1]) * sign
        reward = np.clip(travel / span, 0.0, 1.0).astype(np.float32)
        reward = np.where(success | self._ever_success, 1.0, np.minimum(reward, 0.99))
        reward = reward.astype(np.float32)

        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "goal_configured": True,
                "start_side": "left" if sign > 0 else "right",
                "object_y": pos[:, 1].astype(np.float32),
                "object_z": pos[:, 2].astype(np.float32),
                "crossed_to_other_side": crossed,
                "object_placed": placed,
                "object_on_table": on_table,
                "object_released": released,
                "object_at_rest": at_rest,
                "rest_checked": speed is not None,
                "ever_success": self._ever_success.copy(),
            },
        )
