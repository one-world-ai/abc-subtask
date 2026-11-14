"""Grab-target-from-clutter task evaluator."""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import (
    _body_subtree_ids,
    _quat_to_rotmat,
    _quat_to_rotmat_batch,
    _resolve_bottle_center_offsets,
    _subtree_body_transforms,
)
from abc_sim.task_specs import SimTaskSpec


class GrabClutterTargetInBoxEvaluator:
    """Score success when the prompted target object is placed in the target box."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        goal_site_name: str = "grab_clutter_target_box_goal_region",
    ) -> None:
        self.model = model
        self.spec = spec
        self._target_qpos_addr = self._resolve_target_qpos_addr(model)
        self._box_qpos_addr = self._resolve_joint_qpos_addr(
            model,
            "grab_clutter_target_box_joint",
        )
        self._goal_local_center, self._goal_half_size = self._resolve_goal_site(
            model,
            goal_site_name,
        )
        self._target_center_offset = _resolve_bottle_center_offsets(model, ["target_object"])[0]
        self._target_collision_points = self._resolve_target_collision_points(model)
        self._box_floor_z, self._box_rim_z = self._resolve_box_heights(model)
        self.reset(nworld=1)

    @staticmethod
    def _resolve_target_collision_points(model: mujoco.MjModel) -> np.ndarray:
        """Collision-mesh vertices in the target freejoint frame.

        Primitive geoms use their bounding-box corners (conservative for curved
        primitives). The test only rejects objects wholly above the opening;
        using an outer bound cannot reject a real intersection with the box.
        """
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        transforms = _subtree_body_transforms(model, root)
        bodies = _body_subtree_ids(model, root)
        corners = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        points = []
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) not in bodies:
                continue
            if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
                continue
            kind = int(model.geom_type[g])
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = int(model.geom_dataid[g])
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                local = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
            else:
                size = np.asarray(model.geom_size[g], dtype=np.float64)
                if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    extent = np.full(3, size[0])
                elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                    extent = np.array([size[0], size[0], size[1] + (size[0] if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0)])
                else:
                    extent = size
                local = corners * extent
            bp, br = transforms[int(model.geom_bodyid[g])]
            gr = _quat_to_rotmat(model.geom_quat[g])
            points.append((local @ gr.T + model.geom_pos[g]) @ br.T + bp)
        if not points:
            raise ValueError("Target object has no collision geometry")
        return np.concatenate(points).astype(np.float32)

    @staticmethod
    def _resolve_box_heights(model: mujoco.MjModel) -> tuple[float, float]:
        """Physical floor top and wall rim in the box frame, not the visual goal site."""
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "grab_clutter_target_box_floor")
        walls = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"grab_clutter_target_box_{s}_wall")
                 for s in ("back", "front", "left", "right")]
        if floor < 0 or min(walls) < 0:
            raise ValueError("Target box floor/wall collision geoms are missing")
        def top(g):
            return float(model.geom_pos[g, 2] + np.abs(_quat_to_rotmat(model.geom_quat[g])[2]) @ model.geom_size[g])
        return top(floor), max(top(g) for g in walls)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    @staticmethod
    def _resolve_target_qpos_addr(model: mujoco.MjModel) -> int:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "target_object_joint",
        )
        if joint_id < 0:
            raise ValueError("Joint 'target_object_joint' not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Joint 'target_object_joint' must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} must be a free joint")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_goal_site(
        model: mujoco.MjModel,
        goal_site_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            goal_site_name,
        )
        if site_id < 0:
            raise ValueError(f"Site {goal_site_name!r} not found in model")
        return (
            np.asarray(model.site_pos[site_id], dtype=np.float32).copy(),
            np.asarray(model.site_size[site_id], dtype=np.float32).copy(),
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        target_pos = qpos_batch[:, self._target_qpos_addr: self._target_qpos_addr + 3]
        box_pos = qpos_batch[:, self._box_qpos_addr: self._box_qpos_addr + 3]
        box_quat = qpos_batch[:, self._box_qpos_addr + 3: self._box_qpos_addr + 7]
        rot_world_from_box = _quat_to_rotmat_batch(box_quat)
        rel_world = target_pos - box_pos
        rel_box = np.einsum("bij,bj->bi", np.swapaxes(rot_world_from_box, 1, 2), rel_world)
        delta = np.abs(rel_box - self._goal_local_center[None, :])
        in_goal_region = np.all(delta <= self._goal_half_size[None, :], axis=1)
        # The goal site's deliberately generous vertical extent includes empty
        # air above the rim and space below the floor. Those points alone do not
        # constitute an object in the box. Preserve tall/partly protruding true
        # placements by measuring each target's actual collision geometry.
        target_quat = qpos_batch[:, self._target_qpos_addr + 3:self._target_qpos_addr + 7]
        rot_box_from_target = np.einsum("bji,bjk->bik", rot_world_from_box, _quat_to_rotmat_batch(target_quat))
        lowest_target_z = rel_box[:, 2] + np.min(
            np.einsum("bj,nj->bn", rot_box_from_target[:, 2], self._target_collision_points), axis=1
        )
        target_center_z = rel_box[:, 2] + np.einsum(
            "bj,j->b", rot_box_from_target[:, 2], self._target_center_offset
        )
        geometry_in_box = (target_center_z >= self._box_floor_z - 0.005) & (lowest_target_z <= self._box_rim_z + 0.005)
        success = in_goal_region & geometry_in_box
        self._ever_success |= success

        world_goal_center = box_pos + np.einsum(
            "bij,j->bi",
            rot_world_from_box,
            self._goal_local_center,
        )
        xy_distance = np.linalg.norm(target_pos[:, :2] - world_goal_center[:, :2], axis=1)
        xy_radius = max(float(np.linalg.norm(self._goal_half_size[:2])), 1e-6)
        reward = np.clip(1.0 - (xy_distance / xy_radius), 0.0, 1.0).astype(np.float32)
        reward = np.where(success, 1.0, reward).astype(np.float32)
        return TaskEvalResult(
            reward=reward,
            success=success,
            metrics={
                "target_pos": target_pos.astype(np.float32),
                "target_box_pos": box_pos.astype(np.float32),
                "target_box_center": np.broadcast_to(
                    world_goal_center,
                    target_pos.shape,
                ).copy(),
                "target_box_half_size": np.broadcast_to(
                    self._goal_half_size,
                    target_pos.shape,
                ).copy(),
                "target_box_xy_distance_m": xy_distance.astype(np.float32),
                "target_in_box": success.copy(),
                "target_in_goal_region": in_goal_region.copy(),
                "target_geometry_in_box": geometry_in_box.copy(),
                "target_lowest_z_in_box_frame": lowest_target_z.astype(np.float32),
                "ever_success": self._ever_success.copy(),
            },
        )
