"""Physics-defined stages for load_plates_into_dish_rack (approved 2026-11-14, ``subtask_rewards/dishrack``).

Chain per target plate: pick (STOPPED, label only) -> carry -> insert -> release -> done.
  pick     the plate is not held (lying on the table, or loose)
  carry    held by both jaws of one gripper, off the table (start: an upright plate pinched at its top edge)
  insert   carry goal: held, upright, plane within 15 deg of a slot plane, centre within 2 cm (along the slot axis)
           of a slot centre of the kinematic slot map, over the rack footprint, lowest point <= slot entry + 3 cm
  release  insert goal (stage table): the stock in-rack geometry (spatially in the rack, centre and lowest point over
           the tight footprint, touching the rack, upright) with the plate still touched by a jaw, at a slot of the
           slot map (centre within 2 cm along the slot axis, plane within 15 deg) and its lowest point <= 1 cm above
           that slot's measured seat depth at the plate's position ("seated")
  done     release goal (stage table): the fixed stock success for this plate (in the rack, upright, no gripper
           contact), reached from the insert goal, plate slower than 1 cm/s for 64 consecutive actions (2 chunks),
           every jaw >= 5 cm from the plate
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.task_eval.dishrack import (DishrackEvaluator, _body_subtree_ids, _geom_ids_for_bodies, _geom_points_world,
                                        _gripper_geom_ids, _plate_body_name, _plate_joint_name)
from abc_sim.task_eval.dishrack_geom import compute_slot_map, local_points
from abc_sim.task_eval.stagekit.robot import ARMS
from abc_sim.task_eval.stagekit.scene import ObjectScene
from abc_sim.task_eval.stagekit.tracker import StageTracker

STAGES = ("pick", "carry", "insert", "release", "done")
P, C, I, RL, DONE = range(5)
FALLEN = len(STAGES)
LABELS = STAGES + ("fallen",)
STAGE_PROMPTS = {
    "carry": "dish rack subtask: carry the plate upright above the dish rack",
    "insert": "dish rack subtask: lower the plate into the rack slot",
    "release": "dish rack subtask: release the plate in the rack",
}


@dataclass(frozen=True)
class StageConfig:
    hold: int = 5
    done_hold: int = 64                 # release goal: still for 2 consecutive chunks (stage table)
    upright_nz: float = 0.55            # stock upright_normal_z_max
    flat_nz: float = 0.75               # stock horizontal_normal_z_min
    align_t_m: float = 0.02             # slot play 0-35 mm for a 1.3 cm plate (dr_slots.py)
    align_t_exit_m: float = 0.025
    align_deg: float = 15.0
    align_exit_deg: float = 20.0
    entry_above_m: float = 0.03         # lowest point <= slot entry + 3 cm
    entry_above_exit_m: float = 0.04
    seat_m: float = 0.01                # seated: lowest point <= 1 cm above the slot's seat depth (stage table)
    seat_exit_m: float = 0.015
    still_speed: float = 0.01           # 1 cm/s
    gap_m: float = 0.05                 # every jaw >= 5 cm from the plate
    fallen_m: float = 0.05
    rack_moved_m: float = 0.03
    rack_tilt_rad: float = float(np.deg2rad(10))
    drop_steps: int = 5                 # no jaw on the plate for 5 steps = let go (debounced like the labels)
    flat_steps: int = 64                # insert: lying flat on the rack > 2 chunks
    jam_steps: int = 128                # insert: no descent for 4 chunks while held against the rack
    jam_descent_m: float = 0.003
    jam_force_weights: float = 2.0      # jam: plate-rack normal force > 2 x the plate's weight (pressed, not resting)
    pull_m: float = 0.02                # release: lowest point raised 2 cm above its seated height = pulled out
    left_steps: int = 5
    eps: dict = field(default_factory=lambda: dict(carry=0.005, insert=0.003, release=0.01))
    span: dict = field(default_factory=lambda: dict(carry=0.30, insert=0.15, release=1.0))


_SLOT_CACHE: dict = {}


def _plate_objects(model):
    out = []
    for i in DishrackEvaluator._resolve_plate_indices(model):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _plate_joint_name(i))
        out.append((_plate_body_name(i), int(j)))
    return out


class DishrackScene(ObjectScene):
    def __init__(self, model, cfg=None):
        super().__init__(model, _plate_objects(model))
        self.cfg = cfg or StageConfig()
        self.rack_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dishrack")
        rack_j = int(model.body_jntadr[self.rack_body])
        self.rack_adr = int(model.jnt_qposadr[rack_j])
        rack_geoms = _geom_ids_for_bodies(model, _body_subtree_ids(model, self.rack_body))
        self.rack_geoms = rack_geoms
        self.rack_col = sorted(g for g in rack_geoms if model.geom_contype[g] or model.geom_conaffinity[g])
        self.gripper_geoms = _gripper_geom_ids(model)
        self.world_geoms = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == 0
                            and (model.geom_contype[g] or model.geom_conaffinity[g])}
        self.plate_col = []
        for b in self.bodies:
            gs = _geom_ids_for_bodies(model, _body_subtree_ids(model, b))
            self.plate_col.append(sorted(g for g in gs if model.geom_contype[g] or model.geom_conaffinity[g]))
        self.jaw_geoms = []
        for a, _ in enumerate(ARMS):
            gs = [g for g in range(model.ngeom) if self.owner[g, 0] == 1 and self.owner[g, 1] // 2 == a
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
            self.jaw_geoms.append(gs)
        self._owner_any = {}   # any geom of plate k (the stock checks contacts on all of a plate's geoms)
        for k, b in enumerate(self.bodies):
            for g in _geom_ids_for_bodies(model, _body_subtree_ids(model, b)):
                self._owner_any[int(g)] = k
        self._plate_mass = [float(model.body_subtreemass[b]) for b in self.bodies]
        self._geom_ready = False
        self.target: int | None = None
        self.slots: list[dict] = []
        self._scratch = mujoco.MjData(model)

    # -- constant local geometry (rigid subtrees), measured once from the first state --------------------------------
    def _ensure_geometry(self, data):
        if self._geom_ready:
            return
        m = self.model
        self.plate_pts = [local_points(m, data, g, b) for g, b in zip(self.plate_col, self.bodies)]
        self.plate_pc = [(p.min(0) + p.max(0)) / 2 for p in self.plate_pts]
        self.plate_sub = []
        for p in self.plate_pts:   # jaw-distance subsample (fixed), <= 2000 points
            idx = np.random.default_rng(0).choice(len(p), min(len(p), 2000), replace=False)
            self.plate_sub.append(p[np.sort(idx)])
        self.rack_pts = local_points(m, data, self.rack_col, self.rack_body)
        self.fp_lo = self.rack_pts[:, :2].min(0)
        self.fp_hi = self.rack_pts[:, :2].max(0)
        rp, Rr = data.xpos[self.rack_body], data.xmat[self.rack_body].reshape(3, 3)
        loc = []   # the stock's bounding-sphere half extents (constant in the rack frame)
        for g in self.rack_geoms:
            lg = (data.geom_xpos[g] - rp) @ Rr
            rb = float(m.geom_rbound[g])
            loc += [[lg[0] - rb, lg[1] - rb], [lg[0] + rb, lg[1] + rb]]
        loc = np.asarray(loc)
        self.half_ext = (float(np.max(np.abs(loc[:, 0]))), float(np.max(np.abs(loc[:, 1]))))
        self._geom_ready = True

    def set_target(self, data, k):
        """Choose the episode's plate and load its slot map (cached per rack and plate geometry)."""
        self._ensure_geometry(data)
        self.target = int(k)
        key = (self.rack_pts.shape, round(float(self.rack_pts.sum()), 5),
               self.plate_pts[k].shape, round(float(self.plate_pts[k].sum()), 5))
        if key not in _SLOT_CACHE:
            if len(_SLOT_CACHE) > 256:
                _SLOT_CACHE.clear()
            _SLOT_CACHE[key] = compute_slot_map(self.model, data, self.rack_body, self.rack_col, self.bodies[k],
                                                int(self.qpos_adr[k]), self.plate_col[k])
        self.slot_map = _SLOT_CACHE[key]
        self.slots = self.slot_map["slots"]

    def _allgeoms_of(self, k) -> set:
        return {g for g, o in self._owner_any.items() if o == k}

    def observe_centre(self, data, k) -> np.ndarray:
        """World xy of plate k's bbox centre."""
        self._ensure_geometry(data)
        R = np.asarray(data.xmat[self.bodies[k]]).reshape(3, 3)
        return (np.asarray(data.xpos[self.bodies[k]]) + R @ self.plate_pc[k])[:2]

    def observe_low(self, data, k) -> float:
        """Rack-frame height of plate k's lowest collision point."""
        self._ensure_geometry(data)
        R = np.asarray(data.xmat[self.bodies[k]]).reshape(3, 3)
        pts = self.plate_pts[k] @ R.T + np.asarray(data.xpos[self.bodies[k]])
        rp, Rr = np.asarray(data.xpos[self.rack_body]), np.asarray(data.xmat[self.rack_body]).reshape(3, 3)
        return float(((pts - rp) @ Rr[:, 2]).min())

    def heights(self, data):
        """Plate origin heights, then the rack's position and z axis (the episode's rack anchor)."""
        z = np.asarray([data.qpos[a + 2] for a in self.qpos_adr], dtype=np.float64)
        rack_z = np.asarray(data.xmat[self.rack_body]).reshape(3, 3)[:, 2]
        return np.concatenate([z, data.qpos[self.rack_adr:self.rack_adr + 3], rack_z])

    # -- per-step measurements ---------------------------------------------------------------------------------
    def observe(self, data, rest):
        self._ensure_geometry(data)
        cfg = self.cfg
        m = len(self.names)
        rest = np.asarray(rest, dtype=np.float64)
        rp = np.asarray(data.xpos[self.rack_body]).copy()
        Rr = np.asarray(data.xmat[self.rack_body]).reshape(3, 3).copy()
        lo, hi = self.fp_lo - 0.01, self.fp_hi + 0.01          # stock footprint_tolerance_m
        c = self.contact_obs(data)
        rack_c = np.zeros(m, bool)
        grip_c = np.zeros(m, bool)
        world_c = np.zeros(m, bool)
        owner_any = self._owner_any
        for i in range(int(data.ncon)):
            con = data.contact[i]
            for a, b in ((int(con.geom1), int(con.geom2)), (int(con.geom2), int(con.geom1))):
                k = owner_any.get(a)
                if k is None:
                    continue
                if b in self.rack_geoms:
                    rack_c[k] = True
                if b in self.gripper_geoms:
                    grip_c[k] = True
                if b in self.world_geoms:
                    world_c[k] = True
        nz = np.zeros(m)
        low = np.zeros(m)
        centre_r = np.zeros((m, 3))
        normal_r = np.zeros((m, 3))
        spatial = np.zeros(m, bool)
        over_fp = np.zeros(m, bool)
        lowest_world = np.zeros(m)
        speed = np.zeros(m)
        for k in range(m):
            x = np.asarray(data.xpos[self.bodies[k]])
            R = np.asarray(data.xmat[self.bodies[k]]).reshape(3, 3)
            pts = self.plate_pts[k] @ R.T + x
            zmin = pts[:, 2].min()
            bottom = pts[pts[:, 2] <= zmin + 0.003].mean(0)
            origin_l = (x - rp) @ Rr
            bottom_l = (bottom - rp) @ Rr
            centre_r[k] = ((x + R @ self.plate_pc[k]) - rp) @ Rr
            normal_r[k] = R[:, 2] @ Rr
            nz[k] = abs(R[2, 2])
            low[k] = float(((pts - rp) @ Rr[:, 2]).min())      # lowest point above the rack base (rack frame)
            lowest_world[k] = zmin
            spatial[k] = (self.half_ext[0] + 0.05 - abs(origin_l[0]) >= 0 and self.half_ext[1] + 0.05 - abs(origin_l[1]) >= 0
                          and origin_l[2] + 0.06 >= 0 and 0.45 - origin_l[2] >= 0)       # stock margin / band
            over_fp[k] = bool(np.all(origin_l[:2] >= lo) and np.all(origin_l[:2] <= hi)
                              and np.all(bottom_l[:2] >= lo) and np.all(bottom_l[:2] <= hi))
            dof = int(self.dof_adr[k])
            speed[k] = float(np.linalg.norm(data.qvel[dof:dof + 3]))
        upright = nz <= cfg.upright_nz
        in_rack_geom = spatial & over_fp & rack_c & upright                # stock rule without the gripper check
        stock_ok = in_rack_geom & ~grip_c                                   # the fixed stock per-plate success
        z = np.asarray([data.qpos[a + 2] for a in self.qpos_adr], dtype=np.float64)
        obs = dict(**c, lift=z - rest[:m],
                   nz=nz, upright=upright, low=low, lowest_world=lowest_world, centre_r=centre_r, normal_r=normal_r,
                   spatial=spatial, over_fp=over_fp, rack_contact=rack_c, gripper_contact=grip_c, world_contact=world_c,
                   in_rack_geom=in_rack_geom, stock_ok=stock_ok, speed=speed,
                   rack_moved=float(np.linalg.norm(rp[:2] - rest[m:m + 2])) if len(rest) >= m + 6 else 0.0,
                   rack_tilt=float(np.arccos(np.clip(Rr[:, 2] @ rest[m + 3:m + 6], -1, 1))) if len(rest) >= m + 6 else 0.0)
        # target-only measurements (slot alignment, sinking, support, jaw distance)
        t = self.target
        obs["rack_force_w"] = np.zeros(m)   # plate-rack normal force in plate weights (target only)
        if t is not None and t < m and rack_c[t]:
            f6 = np.zeros(6)
            total = 0.0
            for i in range(int(data.ncon)):
                con = data.contact[i]
                g1, g2 = int(con.geom1), int(con.geom2)
                if (owner_any.get(g1) == t and g2 in self.rack_geoms) or (owner_any.get(g2) == t and g1 in self.rack_geoms):
                    mujoco.mj_contactForce(self.model, data, i, f6)
                    total += abs(f6[0])
            obs["rack_force_w"][t] = total / (self._plate_mass[t] * 9.81)
        obs.update(aligned=np.zeros(m, bool), aligned_exit=np.zeros(m, bool), slot_dist=np.full(m, 1.0),
                   seat_gap=np.full(m, 1.0), seat=np.full(m, np.nan), at_slot=np.zeros(m, bool),
                   at_slot_exit=np.zeros(m, bool), jaw_gap=np.full(m, np.inf))
        if t is not None and t < m:
            self._target_obs(data, obs, t, rp, Rr)
            if obs["stock_ok"][t] and not obs["touched"][t]:
                obs["jaw_gap"][t] = self._jaw_gap(data, t)
        return obs

    def _slot_terms(self, obs, t):
        """Per slot: (along-axis offset, lateral offset, plane angle error rad, height above entry)."""
        n = obs["normal_r"][t]
        cr = obs["centre_r"][t]
        rows = []
        for s in self.slots:
            a = s["axis"]
            ang = float(np.arccos(np.clip(abs(n[a]) / max(np.linalg.norm(n), 1e-9), 0, 1)))
            rows.append((abs(cr[a] - s["t"]), abs(cr[1 - a] - s["u"]), ang, obs["low"][t] - s["entry"]))
        return np.asarray(rows).reshape(-1, 4)

    def _target_obs(self, data, obs, t, rp, Rr):
        cfg = self.cfg
        if not self.slots:
            return
        terms = self._slot_terms(obs, t)
        score = np.hypot(terms[:, 0], terms[:, 1]) + 0.5 * terms[:, 2]
        obs["slot_dist"][t] = float(score.min())
        # slots on the nearest measured line per axis (lines are 25 % of the footprint width apart)
        cr = obs["centre_r"][t]
        on_line = np.zeros(len(self.slots), bool)
        for a in (0, 1):
            us = np.asarray([s["u"] for s in self.slots if s["axis"] == a])
            if len(us):
                line_u = us[int(np.argmin(np.abs(us - cr[1 - a])))]
                on_line |= np.asarray([s["axis"] == a and abs(s["u"] - line_u) < 1e-9 for s in self.slots])
        near_in = on_line & (terms[:, 0] <= cfg.align_t_m) & (terms[:, 2] <= np.deg2rad(cfg.align_deg))
        near_out = on_line & (terms[:, 0] <= cfg.align_t_exit_m) & (terms[:, 2] <= np.deg2rad(cfg.align_exit_deg))
        base = bool(obs["grasped"][t] and obs["upright"][t] and obs["over_fp"][t])
        obs["aligned"][t] = base and bool((near_in & (terms[:, 3] <= cfg.entry_above_m)).any())
        obs["aligned_exit"][t] = base and bool((near_out & (terms[:, 3] <= cfg.entry_above_exit_m)).any())
        obs["at_slot"][t] = bool(near_in.any())
        obs["at_slot_exit"][t] = bool(near_out.any())
        # seat depth at the plate's position along the slot's line (the slot map's zlow profile), for the best slot
        # among the at-slot ones (else among the nearest line's, else all)
        pool = np.flatnonzero(near_out) if near_out.any() else np.flatnonzero(on_line)
        if not len(pool):
            pool = np.arange(len(self.slots))
        s = self.slots[int(pool[np.argmin(score[pool])])]
        line = self.slot_map["lines"][s["line"]]
        ok = np.isfinite(line["zlow"])
        tt = float(np.clip(cr[s["axis"]], s["t_lo"], s["t_hi"]))
        seat = float(np.interp(tt, line["t"][ok], line["zlow"][ok])) if ok.any() else s["seat"]
        obs["seat"][t] = seat
        obs["seat_gap"][t] = max(0.0, float(obs["low"][t] - seat))

    def _jaw_gap(self, data, t) -> float:
        x = np.asarray(data.xpos[self.bodies[t]])
        R = np.asarray(data.xmat[self.bodies[t]]).reshape(3, 3)
        pts = self.plate_sub[t] @ R.T + x
        best = np.inf
        for gs in self.jaw_geoms:
            jp = np.concatenate([_geom_points_world(self.model, data, g) for g in gs])
            dd = np.sqrt(((pts[None, :, :] - jp[:, None, :]) ** 2).sum(-1)).min()
            best = min(best, float(dd))
        return best


def seated(obs, prev, cfg):
    """Insert goal geometry (without the jaw condition): stock in-rack geometry, at a slot, <= 1 cm above its seat."""
    exit_ = prev >= RL
    at = np.where(exit_, obs["at_slot_exit"], obs["at_slot"])
    return obs["in_rack_geom"] & at & (obs["seat_gap"] <= np.where(exit_, cfg.seat_exit_m, cfg.seat_m))


def classify(obs, prev, cfg):
    out = np.asarray(prev, dtype=np.int64).copy()
    seat = seated(obs, prev, cfg)
    held = obs["grasped"]
    aligned = np.where(prev == I, obs["aligned_exit"], obs["aligned"])
    free = ~obs["touched"]
    # free and at rest anywhere but in the rack: lying loose (the pick label)
    out[free & (obs["speed"] <= cfg.still_speed) & ~obs["stock_ok"]] = P
    out[held & ~obs["world_contact"]] = C
    out[held & aligned] = I
    out[seat & obs["touched"]] = RL
    # after the insert goal, a released plate counts while the fixed stock rule holds (decision 2: it may lean in the
    # slot's play); falling over or leaving the rack is the release failure (stage table)
    released = obs["stock_ok"] & free & (prev >= RL)
    out[released] = RL
    out[released & (obs["speed"] <= cfg.still_speed) & (obs["jaw_gap"] >= cfg.gap_m)] = DONE
    out[obs["lift"] < -cfg.fallen_m] = FALLEN
    return out


def progress(stage, obs, nz0=None):
    if stage == C:
        return -obs["slot_dist"]
    if stage == I:
        return -obs["seat_gap"]
    if stage == RL:   # nz0: the plate's |normal_z| at the start (the evaluator passes it)
        released = obs["in_rack_geom"] & ~obs["touched"]
        return 0.5 * released - np.maximum(obs["nz"] - nz0, 0)
    raise ValueError(stage)


class DishrackTracker(StageTracker):
    """StageTracker whose DONE label needs ``cfg.done_hold`` consecutive updates (still for 2 chunks)."""

    def __init__(self, scene, cfg=None):
        super().__init__(scene, classify, cfg or scene.cfg, start_label=P)

    def update(self, data):
        self.obs = self.scene.observe(data, self.rest_height)
        raw = self.classify(self.obs, self.committed, self.cfg)
        self.count = np.where(raw == self.candidate, self.count + 1, 1)
        self.candidate = raw
        need = np.where(raw == DONE, self.cfg.done_hold, self.cfg.hold)
        changed = np.flatnonzero((raw != self.committed) & (self.count >= need))
        self.committed[changed] = raw[changed]
        return changed.tolist()
