"""Stages of hang_mug_on_mug_rack for single-stage RL (approved table: subtask_rewards/mug_rack/stage_table.md).

One target mug per episode (the nearest threadable mug that is not hanging yet). Each mug is in exactly one stage,
computed from the sim state alone (plus hysteresis on the previous stage):

    stage    a mug is in it when
    reach    none of the below
    pick     a grasp site is within reach_m of its CoM or a finger touches it
    align    both fingers of one arm on it, no finger on the handle, lifted >= lift_m
    insert   held, and its handle hole is lined up with a free peg (hole centre <= align_radial_m from the peg's
             axis line, 0..align_axial_m beyond the tip, hole axis <= align_angle_deg from the peg axis, touching
             nothing but the fingers), or a free peg already passes through the hole
    release  a free peg passes through the hole (scorer's threading test, per peg) with the hole centre >= depth_m
             along the peg past its tip, and a finger still touches the mug
    done     the stock scorer's per-mug hanging test, no finger, at rest (debounced: ``hold`` actions)
    fallen   lifted < -fallen_m (off the table)

"free peg": no other mug is threaded on it. The six pegs are the tree pieces whose axis is within 45 deg of
horizontal; the trunk pieces never count as pegs (the stock scorer also accepts a handle hooked on the trunk top).

Per-chunk reward: kit ``SubtaskEvaluator.chunk_reward`` on these progress numbers (all over the target):
    reach    -(distance from the nearest grasp site to the CoM)
    pick     0.25 x fingers + 0.5 x min(lift / lift_m, 1) x (grasped with the handle free)
    align    -min over free pegs of (radial gap + 0.5 x axial gap + 0.02 x angle / 30 deg)
    insert   -min over free pegs of (radial gap + (depth_m - depth)+)
    release  0.025 x (2 - fingers) - 0.5 x min((depth lost since the start)+, 0.1)  (evaluator: needs the start depth)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.task_eval.mug_tree import MugOnRackEvaluator
from abc_sim.task_eval.stagekit.holding import object_variant
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching
from abc_sim.task_eval.stagekit.tracker import StageTracker

STAGES = ("reach", "pick", "align", "insert", "release", "done")
R, P, A, I, RL, DONE = range(6)
FALLEN = len(STAGES)
LABELS = STAGES + ("fallen",)
# Task prompts are lookup keys: every one must normalize to a string no other task uses.
STAGE_PROMPTS = {
    "reach": "mug rack subtask: move a gripper to the target mug",
    "pick": "mug rack subtask: grasp the target mug away from its handle and lift it",
    "align": "mug rack subtask: line the target mug's handle up with a free peg",
    "insert": "mug rack subtask: slide the target mug's handle onto the peg",
    "release": "mug rack subtask: let go of the target mug on the rack",
}
# Handle hole narrower than a peg (phase 0, holes.json: 8 / 11 / 12 mm vs peg 13.2-13.9 mm): never a target
# (approved decision 2).
UNTHREADABLE = frozenset({"mug_1", "mug_7", "mug_10"})


@dataclass(frozen=True)
class StageConfig:
    # Grasp site close enough to start grasping. Mug flip measured 5-10 cm (median 7.6 cm) from a held mug's CoM;
    # mug-rack rim pinches measured in phase 3 (STATUS.md) sit 4-8 cm from the CoM.
    reach_m: float = 0.10
    reach_exit_m: float = 0.12
    lift_m: float = 0.03  # lifted (mug flip / bottles convention)
    fallen_m: float = 0.05  # below rest by this much: off the table
    knocked_upz: float = 0.5  # cup axis more than 60 deg from vertical while not held: knocked over (reach/pick)
    # Align region (approved table). Hole 18-30 mm vs peg ~13.5 mm leaves 2-8 mm radial slack.
    align_radial_m: float = 0.015
    align_axial_m: float = 0.04
    align_angle_deg: float = 30.0
    # Once in insert, a mug stays there while held and roughly lined up (contacts with the peg are expected).
    insert_exit_radial_m: float = 0.025
    insert_exit_axial_m: float = 0.06
    insert_exit_angle_deg: float = 40.0
    depth_m: float = 0.02  # threaded this far past the tip (settled real hangs sit 2-6 cm along the peg)
    depth_exit_m: float = 0.01
    at_rest_speed: float = 0.02  # m/s (bottles / put_relative convention)
    drop_steps: int = 8  # align/insert: no finger this many consecutive actions (and not threaded deep) = dropped
    knock_steps: int = 5  # another hanging mug not hanging this many consecutive actions = knocked off
    hold: int = 5  # actions a new label must persist
    eps: dict = field(default_factory=lambda: dict(reach=0.005, pick=0.05, align=0.005, insert=0.002, release=0.005))
    span: dict = field(default_factory=lambda: dict(reach=0.30, pick=1.0, align=0.40, insert=0.06, release=0.10))


def _peg_axis(model, data, g, axis_xy):
    mid = int(model.geom_dataid[g])
    v = np.asarray(model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]], float)
    w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    c = w.mean(0)
    ax = np.linalg.svd(w - c)[2][0]
    if np.linalg.norm((c + 0.02 * ax)[:2] - axis_xy) < np.linalg.norm((c - 0.02 * ax)[:2] - axis_xy):
        ax = -ax  # point out of the trunk
    t = (w - c) @ ax
    radius = float(np.median(np.linalg.norm((w - c) - np.outer(t, ax), axis=1)))
    return ax, c + ax * t.max(), c + ax * t.min(), radius


class MugRackScene(ObjectScene):
    """Per-mug observations: contacts, lift, tilt, handle contact, handle hole relative to each peg, stock hanging."""

    def __init__(self, model: mujoco.MjModel, cfg: StageConfig | None = None) -> None:
        super().__init__(model, free_joints_matching(model, r"mug_(\d+)_jnt"))
        self.cfg = cfg or StageConfig()
        self.geometry = MugOnRackEvaluator(model=model, spec=None)
        self.variants = [object_variant(model, b) for b in self.bodies]
        self.threadable = np.array([v not in UNTHREADABLE for v in self.variants], dtype=bool)
        self._holes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.measure_tree()

    def measure_tree(self) -> None:
        """Pegs of the tree at its current pose (the randomizer edits the static tree's pose in place)."""
        g = self.geometry
        g._measure_tree(self.model)
        data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, data)
        self.pegs, self.peg_axis, self.peg_tip, self.peg_root, self.peg_radius, self.peg_planes = [], [], [], [], [], []
        for geom, planes in zip(g._tree_hook_geoms, g._tree_hook_planes):
            ax, tip, root, radius = _peg_axis(self.model, data, int(geom), g._axis_xy)
            if abs(ax[2]) > np.sin(np.deg2rad(45)) or planes is None:
                continue  # trunk piece
            self.pegs.append(int(geom))
            self.peg_axis.append(ax)
            self.peg_tip.append(tip)
            self.peg_root.append(root)
            self.peg_radius.append(radius)
            self.peg_planes.append(planes)
        self.peg_axis, self.peg_tip = np.asarray(self.peg_axis), np.asarray(self.peg_tip)
        self.peg_root, self.peg_radius = np.asarray(self.peg_root), np.asarray(self.peg_radius)
        self.tree_geoms = set(int(x) for x in g._tree_geoms)

    def hole(self, k: int):
        """(hole points (n, 3), hole centre (3,), hole axis (3,)) of mug k in its joint-body frame."""
        if k not in self._holes:
            pts = self.geometry._hole(k)
            if len(pts) == 0:
                self._holes[k] = (pts, np.zeros(3), np.array([0.0, 0.0, 1.0]))
            else:
                c = pts.mean(0)
                axis = np.linalg.svd(pts - c)[2][2]  # the slices are 3 mm apart along it: least spread
                self._holes[k] = (pts, c, axis)
        return self._holes[k]

    def handle_hits(self, data: mujoco.MjData) -> np.ndarray:
        """(m,) a finger contact lies on mug k's handle (scorer's handle region, from the collision mesh)."""
        m = len(self.names)
        out = np.zeros(m, dtype=bool)
        g = self.geometry
        for i in range(int(data.ncon)):
            c = data.contact[i]
            o1, o2 = self.owner[int(c.geom1)], self.owner[int(c.geom2)]
            if o1[0] == 0 and o2[0] == 1:
                k = int(o1[1])
            elif o1[0] == 1 and o2[0] == 0:
                k = int(o2[1])
            else:
                continue
            if not out[k] and g._mug_handles[k]["handle_sectors"].size:
                info = g._mug_handles[k]
                root = int(g._mug_bodies[k][0])
                local = (np.asarray(c.pos) - data.xpos[root]) @ data.xmat[root].reshape(3, 3)
                off = local[:2] - info["centre"]
                out[k] = bool(np.linalg.norm(off) > info["wall"] + g.handle_clearance_m
                              and np.isin(g._sector(off[None]), info["allowed"])[0])
        return out

    def other_contact(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(m,) mug k touches something other than a finger; (m,) mug k touches the tree."""
        m = len(self.names)
        other, tree = np.zeros(m, dtype=bool), np.zeros(m, dtype=bool)
        for i in range(int(data.ncon)):
            c = data.contact[i]
            for ga, gb in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                oa, ob = self.owner[ga], self.owner[gb]
                if oa[0] == 0 and not (ob[0] == 0 and ob[1] == oa[1]) and ob[0] != 1:
                    other[oa[1]] = True
                    tree[oa[1]] |= gb in self.tree_geoms
        return other, tree

    def peg_relations(self, data: mujoco.MjData, k: int):
        """Mug k's hole vs every peg: radial gap, axial position (+ = beyond the tip), angle (deg), threaded."""
        pts, c, axis = self.hole(k)
        n = len(self.pegs)
        root = self.bodies[k]
        Rm = data.xmat[root].reshape(3, 3)
        centre = Rm @ c + data.xpos[root]
        hax = Rm @ axis
        rel = centre[None] - self.peg_tip
        axial = np.einsum("ij,ij->i", rel, self.peg_axis) if n else np.zeros(0)
        radial = np.linalg.norm(rel - axial[:, None] * self.peg_axis, axis=1) if n else np.zeros(0)
        angle = np.degrees(np.arccos(np.clip(np.abs(self.peg_axis @ hax), 0, 1))) if n else np.zeros(0)
        threaded = np.zeros(n, dtype=bool)
        if len(pts):
            world = None
            for j in range(n):
                # only pegs that can reach the hole: centre within the peg's length + hole size of the peg segment
                seg = np.clip(axial[j], -np.linalg.norm(self.peg_tip[j] - self.peg_root[j]), 0.0)
                if np.linalg.norm(rel[j] - seg * self.peg_axis[j]) > 0.05:
                    continue
                if world is None:
                    world = pts @ Rm.T + data.xpos[root]
                g = self.pegs[j]
                near = np.sum((world - data.geom_xpos[g]) ** 2, axis=1) <= self.model.geom_rbound[g] ** 2
                threaded[j] = bool(near.any() and self.geometry._inside(world[near], self.peg_planes[j]).any())
        return dict(centre=centre, hole_axis=hax, radial=radial, axial=axial, angle=angle, threaded=threaded)

    def stock_hanging(self, data: mujoco.MjData) -> np.ndarray:
        res = self.geometry.evaluate_qpos_batch(np.asarray(data.qpos, np.float32)[None])
        return np.asarray(res.metrics["mug_threaded_mask"][0], dtype=bool), \
            np.array([n in res.metrics["mugs_hanging"][0] for n in self.geometry.mug_names], dtype=bool)

    def observe(self, data: mujoco.MjData, rest_height: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self.cfg
        m = len(self.names)
        base = super().observe(data, rest_height)
        upz = np.array([data.xmat[b].reshape(3, 3)[2, 2] for b in self.bodies])
        speed = np.array([np.linalg.norm(data.qvel[a:a + 3]) for a in self.dof_adr])
        handle = self.handle_hits(data)
        other, tree = self.other_contact(data)
        stock_threaded, hanging = self.stock_hanging(data)
        n = len(self.pegs)
        rel = {key: np.full((m, n), np.nan) for key in ("radial", "axial", "angle")}
        thr = np.zeros((m, n), dtype=bool)
        for k in range(m):
            # holes are measured lazily: threadable mugs, and others only when the scorer finds them in its band
            if self.threadable[k] or stock_threaded[k]:
                r = self.peg_relations(data, k)
                for key in rel:
                    rel[key][k] = r[key]
                thr[k] = r["threaded"]
        # a peg is free for mug k when no other mug is threaded on it
        occupied = thr.any(axis=0)
        free = np.ones((m, n), dtype=bool)
        for k in range(m):
            free[k] = ~(thr[np.arange(m) != k].any(axis=0)) if m > 1 else True
        big = 1e3
        radial = np.where(free, np.nan_to_num(rel["radial"], nan=big), big)
        axial = np.nan_to_num(rel["axial"], nan=big)
        angle = np.nan_to_num(rel["angle"], nan=180.0)
        axial_gap = np.maximum(np.maximum(-axial, axial - cfg.align_axial_m), 0.0)
        align_cost = radial + 0.5 * axial_gap + 0.02 * angle / 30.0
        depth = np.where(thr & free, -axial, -big)
        insert_cost = radial + np.maximum(cfg.depth_m - np.where(free, -axial, -big), 0.0)
        best = np.argmin(align_cost, axis=1) if n else np.zeros(m, dtype=int)
        idx = np.arange(m)
        threaded_free = (thr & free).any(axis=1) if n else np.zeros(m, dtype=bool)
        return dict(
            base, upz=upz, speed=speed, handle_touch=handle, grasped_free=base["grasped"] & ~handle,
            other_contact=other, tree_contact=tree, hanging=hanging, threadable=self.threadable.copy(),
            align_cost=align_cost.min(axis=1) if n else np.full(m, big),
            insert_cost=insert_cost.min(axis=1) if n else np.full(m, big),
            best_peg=best, radial=radial[idx, best] if n else np.full(m, big),
            axial=axial[idx, best] if n else np.full(m, big), angle=angle[idx, best] if n else np.full(m, 180.0),
            threaded=threaded_free, depth=depth.max(axis=1) if n else np.full(m, -big),
            occupied=np.repeat(occupied[None], m, axis=0),
        )


def classify(obs, prev, cfg):
    near = (obs["dist"] <= np.where(prev >= P, cfg.reach_exit_m, cfg.reach_m)) | obs["touched"]
    out = np.full(prev.shape, R, dtype=np.int64)
    out[near] = P
    out[obs["grasped_free"] & (obs["lift"] >= cfg.lift_m)] = A
    stay = prev >= I
    lined = np.where(
        stay,
        (obs["radial"] <= cfg.insert_exit_radial_m) & (np.abs(obs["axial"]) <= cfg.insert_exit_axial_m)
        & (obs["angle"] <= cfg.insert_exit_angle_deg),
        (obs["radial"] <= cfg.align_radial_m) & (obs["axial"] >= 0) & (obs["axial"] <= cfg.align_axial_m)
        & (obs["angle"] <= cfg.align_angle_deg) & ~obs["other_contact"])
    out[obs["grasped"] & (lined | obs["threaded"])] = I
    deep = obs["threaded"] & (obs["depth"] >= np.where(prev >= RL, cfg.depth_exit_m, cfg.depth_m))
    out[deep & obs["touched"]] = RL
    at_rest = obs["speed"] <= cfg.at_rest_speed
    # Let go: a mug still moving keeps its stage (it may be settling onto the peg or falling).
    free = ~obs["touched"] & ~at_rest & (prev >= A) & (prev < FALLEN)
    out[free] = prev[free]
    out[deep & ~obs["touched"] & ~at_rest & (prev < RL)] = RL
    out[obs["hanging"] & ~obs["touched"] & at_rest] = DONE
    out[obs["lift"] < -cfg.fallen_m] = FALLEN
    return out


def progress(stage, obs):
    if stage == R:
        return -obs["dist"]
    if stage == P:
        return 0.25 * obs["fingers"] + 0.5 * np.clip(obs["lift"] / 0.03, 0, 1) * obs["grasped_free"]
    if stage == A:
        return -np.minimum(obs["align_cost"], 1.0)
    if stage == I:
        return -np.minimum(obs["insert_cost"], 1.0)
    if stage == RL:
        return 0.025 * (2 - obs["fingers"])  # the evaluator subtracts 0.5 x depth lost since the start
    raise ValueError(stage)


class MugRackTracker(StageTracker):
    def __init__(self, scene, cfg=None):
        super().__init__(scene, classify, cfg or scene.cfg, start_label=R)
