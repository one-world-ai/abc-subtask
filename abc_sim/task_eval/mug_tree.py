"""Mug-on-rack task evaluator."""

from __future__ import annotations

import re

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


class MugOnRackEvaluator:
    """Score hang_mug_on_mug_rack by mugs hanging in the tree's peg band.

    The scene deals one to three mugs on the table beside a peg tree the
    randomizer shifts and yaws per reset. A mug hangs when its centre sits
    inside a cylinder around the tree's trunk axis, within the height band
    from just above the tree's lowest point up past its top -- measured
    against the tree's live pose, so the same criterion follows the tree
    wherever a reset put it.

    The band alone does not separate hanging from resting: the tree's base sits
    on the table, so a mug standing on the table (or on the base disc) beside
    the trunk is inside it. A mug therefore also has to touch the tree above
    its base (a peg or the trunk) and touch nothing else except other mugs --
    no table, floor, wall, base disc or robot. Contacts are measured from the
    scored qpos with ``mj_geomDistance`` and a small tolerance, so they do not
    depend on the solver's penetration depth.

    Touching only the tree is still not hanging. A mug dropped onto the tree
    usually comes to rest with a peg or the trunk top inside its cup, or
    balanced on its rim. So a mug must also be held by its handle: some point
    where the handle touches the tree is at least ``hang_above_com_m`` above
    the mug's centre of mass. The handle is found per mug model from its
    collision mesh: the cup axis is the centre of the base ring of vertices,
    the wall radius is the median over 15-degree sectors of the farthest vertex
    from that axis, and the handle is the sectors reaching more than
    ``handle_min_reach_m`` beyond the wall. A touch point counts as on the
    handle when it lies in a handle sector (or next to one) and more than
    ``handle_clearance_m`` outside the wall.

    A handle touching the tree is still not hanging either: a mug resting
    with a peg in its cup or its body on a peg can lean its handle on the
    trunk or another peg above its centre of mass (subtask_rewards/mug_rack,
    2026-11-13: 9 of 59 accepted random settles; with the handle's collision
    removed none of them moved). So a tree piece must also pass through the
    handle's hole. The hole is measured per mug model in the handle's middle
    plane (the plane through the cup axis and the handle, centred on the
    handle's thickness): grid points outside the collision pieces from which
    the handle or wall is hit in all four in-plane directions (outwards,
    inwards, up, down). A mug is threaded when such a point lies inside a
    tree piece above the base. Success requires every dealt mug to be hanging
    at once.

    The tree is a static body whose pose the randomizer edits in place, so
    the evaluator re-measures the tree's world frame on every
    ``configure_from_randomization`` -- rebinding only on model identity
    would keep a stale pose after an in-place perturbation.
    """

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        tree_body_name: str = "mug_tree",
        mug_joint_regex: str = r"mug_(\d+)_jnt",
        hang_radius_m: float = 0.115,
        base_clearance_m: float = 0.005,
        top_clearance_m: float = 0.05,
        touch_tolerance_m: float = 0.002,
        base_geom_height_m: float = 0.01,
        hang_above_com_m: float = 0.01,
        handle_min_reach_m: float = 0.008,
        handle_clearance_m: float = 0.003,
        hole_grid_m: float = 0.001,
    ) -> None:
        self.spec = spec
        self.touch_tolerance_m = float(touch_tolerance_m)
        self.base_geom_height_m = float(base_geom_height_m)
        self.hang_above_com_m = float(hang_above_com_m)
        self.handle_min_reach_m = float(handle_min_reach_m)
        self.handle_clearance_m = float(handle_clearance_m)
        self.hole_grid_m = float(hole_grid_m)
        self._tree_body_name = tree_body_name
        self._mug_joint_regex = mug_joint_regex
        self.hang_radius_m = float(hang_radius_m)
        self.base_clearance_m = float(base_clearance_m)
        self.top_clearance_m = float(top_clearance_m)
        self._bind_model(model)
        self.reset(nworld=1)

    def _bind_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.mug_names, self._mug_qpos_addrs = self._resolve_mug_qpos_addrs(
            model, self._mug_joint_regex
        )
        self._tree_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self._tree_body_name
        )
        if self._tree_body_id < 0:
            raise ValueError(f"Body {self._tree_body_name!r} not found in model")
        if int(model.body_jntnum[self._tree_body_id]) != 0:
            raise ValueError(
                f"Body {self._tree_body_name!r} must be a static body; a jointed "
                "tree would need its pose read from qpos"
            )
        # Collision-mesh vertices in the tree's own frame, measured once; the
        # world pose composes in per reset.
        self._tree_local_vertices = self._collect_local_vertices(
            model, self._tree_body_id
        )
        collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self._mug_geoms: list[np.ndarray] = []
        self._mug_bodies: list[np.ndarray] = []
        for name in self.mug_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_jnt")
            root_body = int(model.jnt_bodyid[joint_id])
            subtree = self._subtree(model, root_body)
            self._mug_bodies.append(
                np.array([root_body, *sorted(set(subtree) - {root_body})], dtype=int)
            )
            self._mug_geoms.append(
                np.flatnonzero(collidable & np.isin(model.geom_bodyid, list(subtree)))
            )
        self._all_mug_geoms = (
            np.concatenate(self._mug_geoms) if self._mug_geoms else np.zeros(0, dtype=int)
        )
        self._tree_geoms = np.flatnonzero(
            collidable & (model.geom_bodyid == self._tree_body_id)
        )
        self._collidable = np.flatnonzero(collidable)
        self._measure_tree(model)
        self._measure_mug_handles(model)

    def _measure_mug_handles(self, model: mujoco.MjModel) -> None:
        # Mug shapes are rigid, so the handle is measured in each mug's own frame at qpos0.
        rest = mujoco.MjData(model)
        mujoco.mj_kinematics(model, rest)
        self._mug_handles = [
            self._measure_mug_handle(model, rest, index) for index in range(len(self.mug_names))
        ]

    @staticmethod
    def _subtree(model: mujoco.MjModel, root: int) -> set[int]:
        ids = {root}
        for body_id in range(int(model.nbody)):
            current = body_id
            while current > 0:
                current = int(model.body_parentid[current])
                if current == root:
                    ids.add(body_id)
                    break
        return ids

    def _measure_tree(self, model: mujoco.MjModel) -> None:
        """Compose the tree's current world pose into the hang band."""

        pos = np.asarray(model.body_pos[self._tree_body_id], dtype=np.float64)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_quat[self._tree_body_id])
        world = self._tree_local_vertices @ rot.reshape(3, 3).T + pos
        self._axis_xy = world[:, :2].mean(axis=0)
        base_z = float(world[:, 2].min())
        top_z = float(world[:, 2].max())
        self._hang_z_lo = base_z + self.base_clearance_m
        self._hang_z_hi = top_z + self.top_clearance_m
        if self._hang_z_lo >= self._hang_z_hi:
            raise ValueError(
                f"Tree {self._tree_body_name!r} measured an empty hang band"
            )
        # The randomizer only shifts and yaws the tree, so which collision
        # pieces form the base (reach down to table level) is fixed per model,
        # but it is cheap to recompute with the pose.
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        base, hooks = [], []
        for geom_id in self._tree_geoms:
            low = self._geom_world_min_z(model, data, int(geom_id))
            (base if low <= base_z + self.base_geom_height_m else hooks).append(int(geom_id))
        self._tree_hook_geoms = np.asarray(hooks, dtype=int)
        self._tree_hook_planes = [self._geom_planes(model, data, g) for g in hooks]
        allowed = np.concatenate([self._tree_hook_geoms, self._all_mug_geoms])
        self._support_geoms = np.setdiff1d(self._collidable, allowed)
        if self._tree_hook_geoms.size == 0:
            raise ValueError(f"Tree {self._tree_body_name!r} has no geoms above its base")

    @staticmethod
    def _geom_world_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
        mesh_id = int(model.geom_dataid[geom_id])
        if int(model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_MESH or mesh_id < 0:
            return float(data.geom_xpos[geom_id][2] - model.geom_rbound[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        verts = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
        world = verts @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        return float(world[:, 2].min())

    @staticmethod
    def _local_planes(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Outward face planes (normals, offsets) of a convex collision mesh in its geom frame."""

        mesh_id = int(model.geom_dataid[geom_id])
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH) or mesh_id < 0:
            return None
        verts = np.asarray(
            model.mesh_vert[model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]],
            dtype=np.float64,
        )
        faces = np.asarray(
            model.mesh_face[model.mesh_faceadr[mesh_id] : model.mesh_faceadr[mesh_id] + model.mesh_facenum[mesh_id]]
        )
        a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        normals = np.cross(b - a, c - a)
        length = np.linalg.norm(normals, axis=1)
        keep = length > 1e-12
        normals = normals[keep] / length[keep, None]
        offsets = np.einsum("ij,ij->i", normals, a[keep])
        flip = normals @ verts.mean(axis=0) > offsets  # orient outwards (pieces are convex)
        normals[flip] *= -1.0
        offsets[flip] *= -1.0
        return normals, offsets

    def _geom_planes(self, model: mujoco.MjModel, data: mujoco.MjData, geom_id: int):
        """Face planes of a convex collision mesh in world coordinates at ``data``'s pose."""

        local = self._local_planes(model, int(geom_id))
        if local is None:
            return None
        normals, offsets = local
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        world_normals = normals @ rot.T
        return world_normals, offsets + world_normals @ data.geom_xpos[geom_id]

    @staticmethod
    def _inside(points: np.ndarray, planes, margin: float = 0.0) -> np.ndarray:
        normals, offsets = planes
        return np.all(points @ normals.T - offsets <= -margin, axis=1)

    def _touches(self, data: mujoco.MjData, geoms_a: np.ndarray, geoms_b: np.ndarray) -> bool:
        """Whether any geom in ``geoms_a`` is within the touch tolerance of one in ``geoms_b``."""

        return bool(self._touch_points(data, geoms_a, geoms_b, first_only=True))

    def _touch_points(
        self,
        data: mujoco.MjData,
        geoms_a: np.ndarray,
        geoms_b: np.ndarray,
        *,
        first_only: bool = False,
    ) -> list[np.ndarray]:
        """World points (midpoints of the nearest-point pairs) where the two geom sets touch."""

        if geoms_a.size == 0 or geoms_b.size == 0:
            return []
        tol = self.touch_tolerance_m
        rb_a = self.model.geom_rbound[geoms_a]
        rb_b = self.model.geom_rbound[geoms_b]
        centre_gap = np.linalg.norm(
            data.geom_xpos[geoms_a][:, None, :] - data.geom_xpos[geoms_b][None, :, :], axis=-1
        )
        # rbound 0 marks planes (unbounded): never pruned.
        near = (centre_gap <= rb_a[:, None] + rb_b[None, :] + tol) | (rb_b[None, :] == 0)
        points: list[np.ndarray] = []
        fromto = np.zeros(6, dtype=np.float64)
        for i, j in zip(*np.nonzero(near)):
            dist = mujoco.mj_geomDistance(
                self.model, data, int(geoms_a[i]), int(geoms_b[j]), tol, fromto
            )
            if dist < tol:
                points.append(0.5 * (fromto[:3] + fromto[3:]))
                if first_only:
                    break
        return points

    def _measure_mug_handle(self, model: mujoco.MjModel, data: mujoco.MjData, mug_index: int) -> dict:
        """Cup axis, wall radius and handle sectors of one mug, in its joint body's frame."""

        root = int(self._mug_bodies[mug_index][0])
        rot = data.xmat[root].reshape(3, 3)
        verts = []
        for geom_id in self._mug_geoms[mug_index]:
            mesh_id = int(model.geom_dataid[geom_id])
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH) or mesh_id < 0:
                continue
            start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            world = (
                np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
                @ data.geom_xmat[geom_id].reshape(3, 3).T
                + data.geom_xpos[geom_id]
            )
            verts.append((world - data.xpos[root]) @ rot)
        if not verts:
            return {"handle_sectors": np.zeros(0, dtype=int)}
        local = np.concatenate(verts)
        base = local[local[:, 2] < local[:, 2].min() + 0.004, :2]
        centre = 0.5 * (base.min(axis=0) + base.max(axis=0))
        offset = local[:, :2] - centre
        radius = np.linalg.norm(offset, axis=1)
        sector = self._sector(offset)
        reach = np.zeros(24)
        np.maximum.at(reach, sector, radius)
        wall = float(np.median(reach[reach > 0]))
        handle = np.flatnonzero(reach > wall + self.handle_min_reach_m)
        allowed = np.unique(np.concatenate([handle, (handle - 1) % 24, (handle + 1) % 24]))
        info = {"centre": centre, "wall": wall, "handle_sectors": handle, "allowed": allowed}
        # The hole is measured lazily (first time this mug is a hanging candidate): it costs ~0.1-1 s per model and
        # scale, and most evaluations never get that far.
        info["hole"] = None
        info["_hole_args"] = (local, offset, radius)
        return info

    def _hole(self, mug_index: int) -> np.ndarray:
        info = self._mug_handles[mug_index]
        if info.get("hole") is None:
            if "_hole_args" not in info:
                info["hole"] = np.zeros((0, 3))
            else:
                data = mujoco.MjData(self.model)
                mujoco.mj_kinematics(self.model, data)
                root = int(self._mug_bodies[mug_index][0])
                info["hole"] = self._measure_hole(
                    self.model, data, mug_index, root, data.xmat[root].reshape(3, 3), *info["_hole_args"], info
                )
        return info["hole"]

    def _measure_hole(self, model, data, mug_index, root, rot, local, offset, radius, info) -> np.ndarray:
        """Points of the handle's hole in the mug's frame (see the class docstring); (0, 3) if none."""

        if info["handle_sectors"].size == 0:
            return np.zeros((0, 3))
        angle = float(np.mean(-np.pi + (info["handle_sectors"] + 0.5) * np.pi / 12))
        out_dir = np.array([np.cos(angle), np.sin(angle)])
        side = np.array([-out_dir[1], out_dir[0]])
        on_handle = (radius > info["wall"] + self.handle_clearance_m) & np.isin(self._sector(offset), info["allowed"])
        middle = float(np.median(offset[on_handle] @ side)) if on_handle.any() else 0.0
        planes = []
        for geom_id in self._mug_geoms[mug_index]:
            if not self._near_handle(model, data, int(geom_id), root, rot, info):
                continue
            world = self._geom_planes(model, data, int(geom_id))
            if world is None:
                continue
            normals, offsets = world
            # into the mug's joint-body frame
            planes.append((normals @ rot, offsets - normals @ data.xpos[root]))
        step = self.hole_grid_m
        handle_z = local[on_handle, 2] if on_handle.any() else local[:, 2]
        rs = np.arange(0.5 * info["wall"], float(radius.max()) + 2 * step, step)
        zs = np.arange(float(handle_z.min()) - 2 * step, float(handle_z.max()) + 2 * step, step)
        rr, zz = np.meshgrid(rs, zs, indexing="ij")
        found = []
        for shift in (-0.003, 0.0, 0.003):
            base = info["centre"] + side * (middle + shift)
            pts = np.stack(
                [base[0] + rr * out_dir[0], base[1] + rr * out_dir[1], zz], axis=-1
            ).reshape(-1, 3)
            solid = np.zeros(len(pts), dtype=bool)
            for piece in planes:
                solid |= self._inside(pts, piece, margin=-1e-4)
            solid = solid.reshape(rr.shape)
            # a solid cell further out / in / up / down along the grid lines
            out_hit = np.flip(np.maximum.accumulate(np.flip(solid, 0), axis=0), 0)
            in_hit = np.maximum.accumulate(solid, axis=0)
            up_hit = np.flip(np.maximum.accumulate(np.flip(solid, 1), axis=1), 1)
            down_hit = np.maximum.accumulate(solid, axis=1)
            enclosed = ~solid & out_hit & in_hit & up_hit & down_hit
            found.append(pts.reshape(*rr.shape, 3)[enclosed])
        return np.concatenate(found) if found else np.zeros((0, 3))

    def _near_handle(self, model, data, geom_id, root, rot, info) -> bool:
        """Whether a mug collision piece reaches into the handle sectors (the rest cannot bound the hole)."""

        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return False
        verts = np.asarray(
            model.mesh_vert[model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id]],
            dtype=np.float64,
        )
        local = (verts @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id] - data.xpos[root]) @ rot
        offset = local[:, :2] - info["centre"]
        return bool(np.isin(self._sector(offset), info["allowed"]).any())

    def _threaded(self, data: mujoco.MjData, mug_index: int) -> bool:
        """A tree piece above the base passes through mug ``mug_index``'s handle hole."""

        hole = self._hole(mug_index)
        if len(hole) == 0:
            return False
        root = int(self._mug_bodies[mug_index][0])
        world = hole @ data.xmat[root].reshape(3, 3).T + data.xpos[root]
        for geom_id, planes in zip(self._tree_hook_geoms, self._tree_hook_planes):
            if planes is None:
                continue
            # only points inside the piece's bounding sphere can be inside it (same verdict, ~50x cheaper)
            near = np.sum((world - data.geom_xpos[geom_id]) ** 2, axis=1) <= self.model.geom_rbound[geom_id] ** 2
            if near.any() and bool(self._inside(world[near], planes).any()):
                return True
        return False

    @staticmethod
    def _sector(offset_xy: np.ndarray) -> np.ndarray:
        angle = np.arctan2(offset_xy[..., 1], offset_xy[..., 0])
        return (np.floor((angle + np.pi) / (np.pi / 12)).astype(int)) % 24

    def _held_by_handle(self, data: mujoco.MjData, mug_index: int, points: list[np.ndarray]) -> bool:
        info = self._mug_handles[mug_index]
        if not points or info["handle_sectors"].size == 0:
            return False
        root = int(self._mug_bodies[mug_index][0])
        rot = data.xmat[root].reshape(3, 3)
        world = np.asarray(points)
        local = (world - data.xpos[root]) @ rot
        offset = local[:, :2] - info["centre"]
        on_handle = (np.linalg.norm(offset, axis=1) > info["wall"] + self.handle_clearance_m) & np.isin(
            self._sector(offset), info["allowed"]
        )
        above = world[:, 2] >= self._mug_com_z(data, mug_index) + self.hang_above_com_m
        return bool((on_handle & above).any())

    def _mug_com_z(self, data: mujoco.MjData, mug_index: int) -> float:
        bodies = self._mug_bodies[mug_index]
        mass = self.model.body_mass[bodies]
        return float(np.dot(mass, data.xipos[bodies, 2]) / max(float(mass.sum()), 1e-9))

    @staticmethod
    def _collect_local_vertices(model: mujoco.MjModel, body_id: int) -> np.ndarray:
        vertices = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_group[geom_id]) != 3:
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            mesh_vertices = np.asarray(
                model.mesh_vert[start : start + count], dtype=np.float64
            )
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
            vertices.append(
                mesh_vertices @ rot.reshape(3, 3).T
                + np.asarray(model.geom_pos[geom_id], dtype=np.float64)
            )
        if not vertices:
            raise ValueError(
                f"Body id {body_id} has no collision-mesh vertices to measure"
            )
        return np.concatenate(vertices, axis=0)

    @staticmethod
    def _resolve_mug_qpos_addrs(
        model: mujoco.MjModel,
        mug_joint_regex: str,
    ) -> tuple[list[str], np.ndarray]:
        pattern = re.compile(mug_joint_regex)
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = pattern.fullmatch(name)
            if match is None:
                continue
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Joint {name!r} must be a free joint")
            entries.append(
                (
                    int(match.group(1)),
                    name.removesuffix("_jnt"),
                    int(model.jnt_qposadr[joint_id]),
                )
            )
        if not entries:
            raise ValueError(f"No joints matching {mug_joint_regex!r} found in model")
        entries.sort(key=lambda item: item[0])
        names = [name for _, name, _ in entries]
        addrs = np.asarray([addr for _, _, addr in entries], dtype=np.int32)
        return names, addrs

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_hanging = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def configure_from_randomization(
        self,
        model: mujoco.MjModel,
        randomization: object,
    ) -> None:
        """Re-measure the tree (its static pose is edited in place per reset) and the mug handles."""

        if model is not self.model:
            self._bind_model(model)
            self.reset(nworld=self._nworld)
        else:
            self._measure_tree(model)
            self._measure_mug_handles(model)

    def debug_spec(self) -> dict[str, object] | None:
        return None

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        mug_pos = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._mug_qpos_addrs], axis=1
        ).astype(np.float64)
        radial = np.linalg.norm(mug_pos[..., :2] - self._axis_xy[None, None, :], axis=-1)
        in_band = (
            (radial <= self.hang_radius_m)
            & (mug_pos[..., 2] >= self._hang_z_lo)
            & (mug_pos[..., 2] <= self._hang_z_hi)
        )
        on_tree = np.zeros_like(in_band)
        held_by_handle = np.zeros_like(in_band)
        threaded = np.zeros_like(in_band)
        supported_elsewhere = np.zeros_like(in_band)
        data = mujoco.MjData(self.model)
        for world_index, qpos in enumerate(qpos_batch):
            if not in_band[world_index].any():
                continue
            data.qpos[:] = qpos
            mujoco.mj_kinematics(self.model, data)
            for mug_index, geoms in enumerate(self._mug_geoms):
                if not in_band[world_index, mug_index]:
                    continue
                points = self._touch_points(data, geoms, self._tree_hook_geoms)
                on_tree[world_index, mug_index] = bool(points)
                held_by_handle[world_index, mug_index] = self._held_by_handle(
                    data, mug_index, points
                )
                threaded[world_index, mug_index] = bool(
                    held_by_handle[world_index, mug_index] and self._threaded(data, mug_index)
                )
                supported_elsewhere[world_index, mug_index] = self._touches(
                    data, geoms, self._support_geoms
                )
        hanging = in_band & on_tree & held_by_handle & threaded & ~supported_elsewhere

        num_hanging = hanging.sum(axis=1).astype(np.int32)
        total = len(self.mug_names)
        self._max_hanging = np.maximum(self._max_hanging, num_hanging)
        success = num_hanging >= total
        self._ever_success |= success

        mugs_hanging = [
            [name for name, ok in zip(self.mug_names, world) if bool(ok)]
            for world in hanging
        ]
        return TaskEvalResult(
            reward=num_hanging.astype(np.float32) / float(max(total, 1)),
            success=success,
            metrics={
                "num_mugs_hanging": num_hanging,
                "num_active_mugs": np.full_like(num_hanging, total),
                "max_mugs_hanging_so_far": self._max_hanging.copy(),
                "mugs_hanging": mugs_hanging,
                "mug_in_band_mask": in_band.tolist(),
                "mug_tree_contact_mask": on_tree.tolist(),
                "mug_held_by_handle_mask": held_by_handle.tolist(),
                "mug_threaded_mask": threaded.tolist(),
                "mug_other_contact_mask": supported_elsewhere.tolist(),
                "mug_names": list(self.mug_names),
                "hang_radius_m": self.hang_radius_m,
                "hang_z_band": [self._hang_z_lo, self._hang_z_hi],
                "ever_success": self._ever_success.copy(),
            },
        )
