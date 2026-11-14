"""LEGO block color-sorting task evaluator."""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch, _subtree_body_transforms
from abc_sim.task_specs import SimTaskSpec


_LEGO_COLORS = ("red", "yellow", "blue")


class LegoBlocksSortingEvaluator:
    """Score success when every LEGO block is in its matching color bin."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
    ) -> None:
        self.model = model
        self.spec = spec
        self._block_qpos_addrs: dict[str, np.ndarray] = {}
        self._bin_qpos_addrs: dict[str, int] = {}
        self._goal_regions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._bin_collision_planes: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._bin_xy_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.block_names: dict[str, list[str]] = {}
        for color in _LEGO_COLORS:
            names, addrs = self._resolve_block_qpos_addrs(model, color)
            self.block_names[color] = names
            self._block_qpos_addrs[color] = addrs
            self._bin_qpos_addrs[color] = self._resolve_joint_qpos_addr(
                model,
                f"{color}_lego_sorting_bin_joint",
            )
            self._goal_regions[color] = self._resolve_goal_site(
                model,
                f"{color}_lego_bin_goal_region",
            )
            self._bin_collision_planes[color], self._bin_xy_bounds[color] = self._measure_bin_cavity(
                model, f"{color}_lego_sorting_bin"
            )
        self.reset(nworld=1)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._active = {
            color: np.ones((self._nworld, len(names)), dtype=bool)
            for color, names in self.block_names.items()
        }

    def set_active_object_joints(self, active_joints_by_world: Any) -> None:
        """Score only each world's active blocks; the rest sit parked off the table."""
        if self._nworld == 1 and active_joints_by_world and isinstance(active_joints_by_world[0], str):
            active_joints_by_world = [active_joints_by_world]
        for world, joints in enumerate(active_joints_by_world):
            active = set(joints)
            for color, names in self.block_names.items():
                self._active[color][world] = [f"{name}_joint" in active for name in names]

    @staticmethod
    def _resolve_block_qpos_addrs(
        model: mujoco.MjModel,
        color: str,
    ) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        pattern = re.compile(rf"{re.escape(color)}_lego_(\d+)_joint")
        for joint_id in range(model.njnt):
            joint_name = model.jnt(joint_id).name
            if not joint_name:
                continue
            match = pattern.fullmatch(joint_name)
            if match is None:
                continue
            index = int(match.group(1))
            entries.append((index, f"{color}_lego_{index}", int(model.jnt_qposadr[joint_id])))

        if not entries:
            raise ValueError(f"No {color}_lego_*_joint freejoints found in model")

        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

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
        site_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site {site_name!r} not found in model")
        return (
            np.asarray(model.site_pos[site_id], dtype=np.float32),
            np.asarray(model.site_size[site_id], dtype=np.float32),
        )

    def _inside_bin(
        self,
        positions: np.ndarray,
        qpos_batch: np.ndarray,
        color: str,
    ) -> np.ndarray:
        bin_qpos_addr = self._bin_qpos_addrs[color]
        goal_local_center, goal_size = self._goal_regions[color]
        bin_pos = qpos_batch[:, bin_qpos_addr: bin_qpos_addr + 3]
        bin_quat = qpos_batch[:, bin_qpos_addr + 3: bin_qpos_addr + 7]
        rel_world = positions - bin_pos[:, None, :]
        rot_world_from_bin = _quat_to_rotmat_batch(bin_quat)
        rel_bin = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_bin, 1, 2), rel_world)
        # A sideways or inverted container is not a sorting receptacle. The
        # opening is local +z; yaw and ordinary rocking preserve its upward
        # direction. Use only a numerical margin here, leaving stricter tilt
        # limits to the subtask's bin-movement failure policy.
        opening_up = rot_world_from_bin[:, 2, 2] > 1e-6
        # The authored site is too narrow: physically settled blocks against
        # an inner wall can sit outside its xy box. Use the actual collision
        # cavity instead. A segment from the cavity centre to a block centre
        # must not cross any of the convex wall/floor collision pieces.
        lo, hi = self._bin_xy_bounds[color]
        inside = opening_up[:, None] & np.all((rel_bin[..., :2] >= lo) & (rel_bin[..., :2] <= hi), axis=-1)
        inside &= np.abs(rel_bin[..., 2] - goal_local_center[2]) <= goal_size[2]
        if not np.any(inside):
            return inside
        for normals, offsets in self._bin_collision_planes[color]:
            a = np.einsum("...i,fi->...f", rel_bin[..., :2], normals[:, :2])
            b = rel_bin[..., 2, None] * normals[:, 2] + offsets
            parallel = np.abs(a) < 1e-10
            possible = ~np.any(parallel & (b > 1e-7), axis=-1)
            ratio = -b / np.where(parallel, 1.0, a)
            entry = np.maximum(0.0, np.max(np.where(a < -1e-10, ratio, -np.inf), axis=-1))
            exit = np.minimum(1.0, np.min(np.where(a > 1e-10, ratio, np.inf), axis=-1))
            crosses_solid = possible & (entry <= exit + 1e-7)
            inside &= ~crosses_solid
        return inside

    @staticmethod
    def _measure_bin_cavity(model: mujoco.MjModel, name: str):
        """Convex collision-piece planes in bin coordinates, measured per model.

        MuJoCo collides against each mesh's convex hull, even when its exported
        triangle faces are not convex. Recover hull planes from the small set
        of distinct collision vertices without adding a geometry dependency.
        """
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if root < 0:
            raise ValueError(f"Bin body {name!r} not found")
        transforms = _subtree_body_transforms(model, root)
        pieces, all_vertices = [], []
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            if body not in transforms or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError(f"Expected convex mesh collider for {name}, got {model.geom(g).name}")
            mesh = int(model.geom_dataid[g])
            vertices = np.asarray(model.mesh_vert[model.mesh_vertadr[mesh]:
                                                  model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]], dtype=np.float64)
            geom_rot = _quat_to_rotmat_batch(np.asarray(model.geom_quat[g])[None, :])[0].astype(np.float64)
            pos, rot = transforms[body]
            vertices = (vertices @ geom_rot.T + model.geom_pos[g]) @ rot.T + pos
            graph = int(model.mesh_graphadr[mesh])
            if graph >= 0:
                # MuJoCo 3.8 user_mesh.cc MakeGraph/CopyGraph stores hull
                # triangles after the vertex and adjacency graph arrays:
                # https://github.com/google-deepmind/mujoco/blob/3.8.0/src/user/user_mesh.cc
                nv, nf = map(int, model.mesh_graph[graph:graph + 2])
                fa = graph + 2 + 3 * nv + 3 * nf
                faces = np.asarray(model.mesh_graph[fa:fa + 3 * nf]).reshape(-1, 3)
                triangles = vertices[faces]
            else:
                vertices = np.unique(np.round(vertices, 9), axis=0)
                if len(vertices) > 128:
                    raise ValueError(f"Unexpectedly large uncompiled collision hull for {model.geom(g).name}")
                triangles = vertices[np.asarray(list(combinations(range(len(vertices)), 3)), dtype=np.int64)]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            valid = lengths > 1e-12
            normals = normals[valid] / lengths[valid, None]
            offsets = -np.einsum("ij,ij->i", normals, triangles[valid, 0])
            signed = vertices @ normals.T + offsets
            supported = np.all(signed <= 1e-8, axis=0) | np.all(signed >= -1e-8, axis=0)
            normals, offsets = normals[supported], offsets[supported]
            flip = normals @ vertices.mean(axis=0) + offsets > 0
            normals[flip] *= -1; offsets[flip] *= -1
            if np.max(vertices @ normals.T + offsets) > 2e-6:
                raise ValueError(f"Invalid collision hull {model.geom(g).name}")
            planes = np.unique(np.round(np.column_stack([normals, offsets]), 9), axis=0)
            pieces.append((planes[:, :3], planes[:, 3]))
            all_vertices.append(vertices)
        if not pieces:
            raise ValueError(f"No collision meshes for {name}")
        vertices = np.concatenate(all_vertices)
        return pieces, (vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0))

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        num_correct = np.zeros((qpos_batch.shape[0],), dtype=np.int32)
        num_wrong_bin = np.zeros((qpos_batch.shape[0],), dtype=np.int32)
        metrics: dict[str, np.ndarray] = {}
        total_blocks = np.zeros((qpos_batch.shape[0],), dtype=np.int32)
        for color in _LEGO_COLORS:
            positions = np.stack(
                [qpos_batch[:, addr : addr + 3] for addr in self._block_qpos_addrs[color]],
                axis=1,
            )
            active = self._active[color]
            in_matching_bin = active & self._inside_bin(positions, qpos_batch, color)
            color_sorted = in_matching_bin.sum(axis=1).astype(np.int32)
            metrics[f"num_{color}_sorted"] = color_sorted
            num_correct += color_sorted
            total_blocks += active.sum(axis=1)

            for other_color in _LEGO_COLORS:
                if other_color == color:
                    continue
                num_wrong_bin += (active & self._inside_bin(positions, qpos_batch, other_color)).sum(
                    axis=1
                ).astype(np.int32)

        success = (num_correct == total_blocks) & (num_wrong_bin == 0)
        self._ever_success |= success
        metrics.update(
            {
                "num_correct": num_correct,
                "num_wrong_bin": num_wrong_bin,
                "ever_success": self._ever_success.copy(),
            }
        )
        return TaskEvalResult(
            reward=(num_correct / np.maximum(total_blocks, 1)).astype(np.float32),
            success=success,
            metrics=metrics,
        )
