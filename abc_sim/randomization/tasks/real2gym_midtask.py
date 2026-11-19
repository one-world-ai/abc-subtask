"""Randomizer of the Real2Gym mid-task starts (abc_sim/real2gym_midtask.py): the task's normal reset, then the target's
predecessors placed at their demo poses (relative to their anchor), demo joint values set, and a second settle.

Placement of one predecessor p with demo pose T_p and anchor pose T_a (same demo frame) and anchor pose now T_a':
  T_p' = Rz(dyaw) about the anchor: p' = p_a' + Rz(dyaw) (p_p - p_a), q' = qz(dyaw) q_p, with dyaw the yaw of
  R_a' R_a^T (the layout jitter; the anchor's small settle tilt is not carried over a lever of up to 50 cm).
Candidates in order, first without conflict wins: that pose, the same without the yaw, the demo pose as is (no
jitter); then for an object in a container ("in" goal) the first pose nudged by 2 / 4 / 6 cm in 8 directions, for an
object whose goal is on the table ("off", "upright": the releases' tables are deeper than abc's, demo end poses can lie
beyond its back edge) the free table spots of a 2 cm grid nearest the first pose, same orientation. Conflict: a contact
of p with anything (robot included) deeper than PEN_MAX_M; for table goals also a footprint (conservative box) closer
than SPOT_CLEARANCE_M to another object's, off the table or in an arm base keep-out. Then the scene settles
SETTLE_MIN_STEPS (or the task's settle_steps if longer) with the arms held; velocities zeroed, time 0.
"""
from __future__ import annotations

import copy
from typing import Any

import mujoco
import numpy as np

from ...real2gym_midtask import midtask_record
from .real2gym_tasks import (ARM_BASES_XY, ARM_KEEPOUT_M, TABLE_MARGIN, TABLE_X, TABLE_Y, Real2GymRandomizer, _corners,
                              _overlap, _quad_disc, load_library)

PEN_MAX_M = 0.006
SETTLE_MIN_STEPS = 500
SPOT_CLEARANCE_M = 0.01
SPOT_CLEARANCE_TIGHT_M = 0.003  # second pass when no spot keeps 1 cm
JOINT_CLEAR_M = 0.0  # a staged joint's part may not touch the robot
GRID_X = np.round(np.arange(0.33, 0.871, 0.02), 3)
GRID_Y = np.round(np.arange(-0.62, 0.621, 0.02), 3)
NUDGES = [(0.0, 0.0)] + [(r * np.cos(a), r * np.sin(a)) for r in (0.02, 0.04, 0.06) for a in np.arange(8) * np.pi / 4]


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _qmul(a, b) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _subtree_geoms(m, b0: int) -> set[int]:
    bodies = {b0}
    for b in range(b0 + 1, m.nbody):
        if int(m.body_parentid[b]) in bodies:
            bodies.add(b)
    return {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies}


def _hold_arms(env) -> None:
    env.data.ctrl[:] = 0.0
    for i, idx in enumerate(env._ctrl_indices):
        env.data.ctrl[idx] = env.data.qpos[env._qpos_indices[i]] if i not in env._gripper_set else 0.0475


def stage_midtask(env, task: str, target: str, rec: dict[str, Any], settle_steps: int = 0) -> dict[str, Any]:
    """Stage rec's predecessors and joints in env (after the normal reset settle). Returns a report."""
    m, d = env.model, env.data
    lib = {o["name"]: o for o in load_library(task)["objects"]}

    def free_adr(name):
        fj = lib[name]["free_joint"]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, fj) if fj else -1
        return (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])) if j >= 0 else (None, None)

    def pose_now(name):
        qa, _ = free_adr(name)
        if qa is not None:
            return d.qpos[qa: qa + 3].copy(), d.qpos[qa + 3: qa + 7].copy()
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        return d.xpos[b].copy(), d.xquat[b].copy()

    def heading(R):
        return float(np.arctan2(R[1, 0], R[0, 0]))

    colliding = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)

    def body_corners(name):
        """Corners of the colliding geoms' boxes of the object's subtree, in its root body frame."""
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        Rb, xb = d.xmat[b].reshape(3, 3), d.xpos[b]
        pts = []
        for g in _subtree_geoms(m, b):
            if not colliding(g):
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            w = (signs * h + c) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pts.append((w - xb) @ Rb)
        return np.concatenate(pts) if pts else np.zeros((1, 3))

    def quad(pts_body, pos, quat):
        """Footprint rectangle (in the body's heading frame) of the body-frame points at a pose -> 4 x 2 corners."""
        R = _mat(quat)
        w = pts_body @ R.T + np.asarray(pos)
        yaw = heading(R)
        c, s_ = np.cos(yaw), np.sin(yaw)
        loc = w[:, :2] @ np.array([[c, -s_], [s_, c]])  # world xy -> heading frame
        lo, hi = loc.min(0), loc.max(0)
        box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        return box @ np.array([[c, s_], [-s_, c]])

    lo_t = np.array([TABLE_X[0], TABLE_Y[0]]) + TABLE_MARGIN
    hi_t = np.array([TABLE_X[1], TABLE_Y[1]]) - TABLE_MARGIN
    present = [k for k in lib if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k) >= 0]

    def footprint_free(q, others, gap):
        if np.any(q.min(0) < lo_t) or np.any(q.max(0) > hi_t):
            return False
        if any(_quad_disc(q, c, ARM_KEEPOUT_M) for c in ARM_BASES_XY):
            return False
        return not any(_overlap(q, qo, gap=gap) for qo in others)

    robot = set()
    for g in range(m.ngeom):
        root = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_rootid[int(m.geom_bodyid[g])])) or ""
        if root.startswith(("left_", "right_")) or "arm" in root or "station" in root:
            robot.add(g)
    report = {"target": target, "staged": [], "joints": {}, "ok": True}
    for j, q in rec.get("joints", {}).items():
        # the demo's value, moved back toward the start value in 0.05 rad steps while the moved part touches the robot
        # (an open door resting on an arm would push it: the demo held it open with that arm)
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr, dof = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
        part = _subtree_geoms(m, int(m.jnt_bodyid[jid]))
        q0 = float(d.qpos[adr])
        val = float(q)
        while True:
            d.qpos[adr], d.qvel[dof] = val, 0.0
            mujoco.mj_forward(m, d)
            hit = any(((int(c.geom1) in part and int(c.geom2) in robot) or (int(c.geom2) in part and int(c.geom1) in robot))
                      and c.dist < JOINT_CLEAR_M for c in (d.contact[i] for i in range(d.ncon)))
            if not hit or abs(val - q0) < 0.05:
                break
            val += 0.05 * np.sign(q0 - val)
        report["joints"][j] = round(val, 4)
        report.setdefault("joints_demo", {})[j] = float(q)
    for s in rec["staged"]:
        p = s["name"]
        qa, da = free_adr(p)
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p)
        own = _subtree_geoms(m, b)
        pa, qa_ = pose_now(s["anchor"])
        Ra, Ra0 = _mat(qa_), _mat(s["anchor_quat"])
        Rd = Ra @ Ra0.T
        dyaw = float(np.arctan2(Rd[1, 0], Rd[0, 0]))
        rel = np.asarray(s["pos"], float) - np.asarray(s["anchor_pos"], float)
        pd, qd = np.asarray(s["pos"], float), np.asarray(s["quat"], float)
        c, sn = np.cos(dyaw), np.sin(dyaw)
        Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])
        base = [("anchor", pa + Rz @ rel, _qmul(_yaw_quat(dyaw), qd)),
                ("anchor_no_yaw", pa + rel, qd.copy()),
                ("demo_pose", pd.copy(), qd.copy())]
        inside = s.get("goal") == "in"  # in a container: the demo pose relative to it; else anywhere free on the table
        if inside:
            cands = base + [(f"anchor+nudge{k}", base[0][1] + np.array([dx, dy, 0.0]), base[0][2])
                            for k, (dx, dy) in enumerate(NUDGES) if k]
        else:
            x0 = base[0][1][:2]
            grid = sorted(((x, y) for x in GRID_X for y in GRID_Y), key=lambda q: (q[0] - x0[0]) ** 2 + (q[1] - x0[1]) ** 2)
            cands = base + [(f"table_spot{'' if gap == SPOT_CLEARANCE_M else '_tight'}", np.array([x, y, base[0][1][2]]),
                             base[0][2], gap) for gap in (SPOT_CLEARANCE_M, SPOT_CLEARANCE_TIGHT_M) for x, y in grid]
        chosen = None
        if not inside:
            pts = body_corners(p)
            others = []
            for k in present:
                if k != p:
                    pk, qk = pose_now(k)
                    others.append(quad(body_corners(k), pk, qk))
        for label, pos, quat, *gap in cands:
            quat = quat / np.linalg.norm(quat)
            if not inside and not footprint_free(quad(pts, pos, quat), others, gap[0] if gap else SPOT_CLEARANCE_M):
                continue
            d.qpos[qa: qa + 3], d.qpos[qa + 3: qa + 7] = pos, quat
            d.qvel[da: da + 6] = 0.0
            mujoco.mj_forward(m, d)
            pen = 0.0
            for i in range(d.ncon):
                ct = d.contact[i]
                if (int(ct.geom1) in own) != (int(ct.geom2) in own):
                    pen = max(pen, float(-ct.dist))
            if pen <= PEN_MAX_M:
                chosen = (label, pen, pos)
                break
        if chosen is None:  # keep the anchor pose; the report flags it
            label, pos, quat = cands[0][:3]
            d.qpos[qa: qa + 3], d.qpos[qa + 3: qa + 7] = pos, quat / np.linalg.norm(quat)
            mujoco.mj_forward(m, d)
            chosen = ("anchor(conflict)", float("nan"), pos)
            report["ok"] = False
        moved = float(np.linalg.norm(np.asarray(chosen[2])[:2] - base[0][1][:2]))
        report["staged"].append({"name": p, "placement": chosen[0], "pen_mm": round(1000 * chosen[1], 2),
                                 "dyaw_deg": round(float(np.degrees(dyaw)), 2), "off_demo_pose_m": round(moved, 3)})
    # settle with the arms held, recording the last 0.5 s
    n = max(int(settle_steps), SETTLE_MIN_STEPS)
    last = int(round(0.5 / m.opt.timestep))
    frees = [k for k in lib if free_adr(k)[0] is not None]
    bids = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k) for k in frees}
    _hold_arms(env)
    mark = None
    for k in range(n):
        if k == n - last:
            mark = {x: d.xipos[bids[x]].copy() for x in frees}
        mujoco.mj_step(m, d)
    if mark is not None:
        mv = {x: float(np.linalg.norm(d.xipos[bids[x]] - mark[x])) for x in frees}
        worst = max(mv, key=mv.get) if mv else None
        report["moving_last_0p5s_mm"] = round(1000 * mv[worst], 2) if worst else 0.0
        report["moving_object"] = worst
    report["settle_steps"] = n
    report["joints_after_settle"] = {j: round(float(d.qpos[int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])]), 4)
                                     for j in rec.get("joints", {})}
    d.qvel[:] = 0.0
    d.time = 0.0
    mujoco.mj_forward(m, d)
    return report


class Real2GymMidtaskRandomizer(Real2GymRandomizer):
    """Real2GymRandomizer + the mid-task staging of ``midtask_target`` (module docstring). Extra positional /
    keyword arguments go to Real2GymRandomizer unchanged (its signature differs between branches)."""

    def __init__(self, task: str, *args, midtask_target: str, **kwargs) -> None:
        super().__init__(task, *args, **kwargs)
        self._mt_args, self._mt_kwargs = args, kwargs
        self.midtask_target = midtask_target
        self.midtask = midtask_record(task, midtask_target)
        if self.midtask is None:
            raise KeyError(f"no mid-task start for {task}/{midtask_target} in real2gym_midtask.json")
        self.midtask_report: dict[str, Any] | None = None

    def clone(self) -> "Real2GymMidtaskRandomizer":
        return type(self)(self.task, *self._mt_args, midtask_target=self.midtask_target, **self._mt_kwargs)

    def _reload_with_joints(self, xml: str, prompt: str, blocks) -> None:
        super()._reload_with_joints(xml, prompt, blocks)
        env = self._env_ref
        if env is not None:
            self.midtask_report = stage_midtask(env, self.task, self.midtask_target, self.midtask, self.settle_steps)

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any | None = None):
        self.midtask_report = None
        state = super().randomize(model, data, seed=seed, request=request)
        md = dict(state.metadata or {})
        md["r2g_midtask"] = {"target": self.midtask_target, "frame": self.midtask["frame"],
                             "staged": [s["name"] for s in self.midtask["staged"]], "joints": dict(self.midtask["joints"]),
                             "report": self.midtask_report}
        self._respec(md, seed)
        state.metadata = md
        return state

    def _respec(self, md: dict, seed) -> None:
        """Subtask spec envs (branch fix-align, real2gym_spec.py): the phrase table rules the target out for the reason
        the staging removes (it starts under / behind something); resolve the spec again with that flag lifted."""
        spec = md.get("r2g_spec")
        env = self._env_ref
        if not isinstance(spec, dict) or spec.get("valid", True) or env is None:
            return
        try:
            from ...real2gym_spec import load_phrases, resolve_spec
        except ImportError:
            return
        phrases = dict(load_phrases())
        t = copy.deepcopy(phrases.get(self.task) or {})
        e = (t.get("objects") or {}).get(self.midtask_target)
        if not isinstance(e, dict):
            return
        e.update(pick_ok=True, obj_ok=True, pick_reason="", obj_reason="")
        phrases[self.task] = t
        lib = load_library(self.task)["objects"]
        kw = dict(members=[self.midtask_target], group=getattr(self, "group", None), seed=seed,
                  names=[o["name"] for o in lib], extents={o["name"]: o["extent"] for o in lib}, phrases=phrases)
        if getattr(self, "variant", None) == "obj" and hasattr(self, "_goals_met"):
            kw["goal_met"] = self._goals_met(env)
        spec2 = resolve_spec(env.model, env.data, self.task, self.variant, self.midtask_target, **kw)
        spec2["midtask"] = True
        md["r2g_spec"] = spec2
        md["prompt"] = env.prompt = spec2["instruction"]


def midtask_randomizers() -> dict[str, Real2GymMidtaskRandomizer]:
    """One randomizer per mid-task env of real2gym_catalog.variants()."""
    import inspect

    from ...real2gym_catalog import variants
    params = inspect.signature(Real2GymRandomizer.__init__).parameters
    kw = {"include_legacy": True} if "include_legacy" in inspect.signature(variants).parameters else {}
    out = {}
    for name, v in variants(**kw).items():
        if not v.get("midtask"):
            continue
        args = (v["variant"], v["object"], v.get("members"), v.get("group")) if "variant" in params else ()
        out[name] = Real2GymMidtaskRandomizer(v["task"], *args, midtask_target=v["midtask"]["target"])
    return out


__all__ = ["Real2GymMidtaskRandomizer", "midtask_randomizers", "stage_midtask"]
