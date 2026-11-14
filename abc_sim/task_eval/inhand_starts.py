"""Start states for the inhand_transfer subtasks (see ``inhand_stages``), plus the grasp planner and physical motion
helpers the scripted solvers use (phase 3/4 of subtask_rewards/inhand).

A start is ``reset(seed)`` + settle (the item drops from z = 0.82 at reset) + a scripted setup that makes the stage's
precondition true:

    reach    the settled reset scene
    pick     A (spawn-side arm) open around the resting item (synthesized top-down grasp, nothing touched)
    present  A holds the item 5-15 cm above its rest spot (spawn spot pushed out to |y| >= 0.14 so it is not already
             in the middle region); kit ``hold_object``
    handoff  the item moved (weightless) to the middle region, yaw of its long axis from YAWS, A holds it with the
             first grasp family for which B also has a reachable grasp (phase-0 probe ``ih_handoff.py``)
    carry    chained: a handoff start + the scripted handoff solver (B approaches, closes, A opens and backs off);
             fallback (grasp source 3): B teleported onto the same planned grasp, then the same physical release
    place    B holds the item 3-10 cm above a random far-zone point (synthesized; A at home)

Skipped seeds (approved decisions 4/5): spawn overlap (item > 5 mm inside the table/arm at reset; they launch), and
for handoff/carry/place the cream-cheese sticks and cheese graters (no handoff found). Start sets take the first 100
eligible seeds from SEED_BASE (subtask_rewards/inhand/STATUS.md).

Gripper facts (measured in phase 0): link_6 z points to the fingers, grasp site 0.1347 m along it, the fingers close
along link_6 x.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from abc_sim.task_eval.inhand_stages import LABELS, STAGES, R, StageConfig, classify
from abc_sim.task_eval.stagekit import (ARMS, GRIP_MAX, STATE_SPEC, T, actuator_limits, arm_ids, axis_angle_quat,
                                        hold_object, ik_link6, inv, qmul, quat_of, set_object_pose)

SEED_BASE = 1_100_000_000
EXCLUDED_HANDOFF = ("cream_cheese_stick", "cheese_grater")  # decision 4
SPAWN_OVERLAP_M = 0.005  # decision 5
SITE_Z = 0.1347
# Pinch centre in link_6: the finger pads span link_6 y -0.047..-0.017 and z 0.07..0.15 on both arms (measured,
# STATUS phase 2), so the ``<arm>_grasp_site`` at (0, 0, 0.1347) is 3 cm off the pads. Grasps put this point on
# the item; a weightless close (hold_object) hides the offset by pulling the item into the fingers, a held one not.
PAD = np.array([0.0, -0.031, 0.125])
R_DOWN = np.diag([1.0, -1.0, -1.0])  # link_6 z -> world -z
YAWS = np.deg2rad([90, 60, 120, 30, 150, 0])
MIDS = [(0.55, 0.90), (0.50, 0.85), (0.60, 0.85), (0.50, 0.95)]
TILTS = np.deg2rad([0, 25, -25, 45, -45, 65, -65])
# Handoff grasp families, tried in order. mid_* (repair round 1 of handoff G3): both grasps of a lying item 6 / 9 cm
# either side of its CoM -- end grasps of long items creep 2-18 mm per chunk in A's hand (round 0: 26/100 timeouts).
MODES = ("mid_topdown6", "mid_topdown9", "ends_topdown", "top_side", "mid_side6", "ends_side", "side_side")
MAX_CREEP_M = 0.002  # held starts: sink at most this in one extra chunk (put_relative used 1 mm; end grasps of
# long items creep 1.1-1.5 mm and still hold for 10 chunks -- the G3 idle check is the gate)
BUILD_BUDGET_S = 9.0


# ---------------------------------------------------------------------------------------------------------- actions

def current_action(env) -> np.ndarray:
    """14-D action that keeps every actuator at its current target."""
    act = np.zeros(14, np.float32)
    for i, ci in enumerate(env._ctrl_indices):
        act[i] = env.data.ctrl[ci] / GRIP_MAX if i in env._gripper_set else env.data.ctrl[ci]
    if not np.any(act[:6]) and not np.any(act[7:13]):  # nothing commanded yet: hold the joints where they are
        for a, arm in enumerate(ARMS):
            ids = arm_ids(env.model, arm)
            act[a * 7:a * 7 + 6] = env.data.qpos[ids.qadr]
            act[a * 7 + 6] = env.data.qpos[ids.finger_qadr[0]] / GRIP_MAX
    return act


def action_with(env, q: dict | None = None, grip: dict | None = None) -> np.ndarray:
    act = current_action(env)
    for a, v in (q or {}).items():
        if v is not None:
            act[a * 7:a * 7 + 6] = v
    for a, g in (grip or {}).items():
        act[a * 7 + 6] = g
    return act


def run(env, act, n: int) -> None:
    for _ in range(int(n)):
        env.step(act, render_obs=False)


def link6_T(env, arm: int) -> np.ndarray:
    ids = arm_ids(env.model, ARMS[arm])
    return T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])


def move_link(env, arm: int, target: np.ndarray, *, actions: int = 64, grip: float | None = None,
              n_way: int = 8, tol=(0.008, 0.2), extra: dict | None = None) -> tuple[bool, str]:
    """Physical Cartesian move of ``arm``'s link_6 to ``target`` (IK waypoints planned from the current joints,
    executed as joint targets over ``actions`` env steps; other actuators keep their targets). ``extra``:
    {other_arm: target link_6 pose} moved along simultaneously. False if a waypoint is unreachable."""
    model, data = env.model, env.data
    plans = {arm: target, **(extra or {})}
    paths = {}
    planning = mujoco.MjData(model)
    planning.qpos[:] = data.qpos
    for a, goal in plans.items():
        ids = arm_ids(model, ARMS[a])
        start = link6_T(env, a)
        qa, qb = quat_of(start[:3, :3]), quat_of(goal[:3, :3])
        if qa @ qb < 0:
            qb = -qb
        lim = actuator_limits(model, ARMS[a])
        way = []
        for f in np.linspace(0, 1, n_way + 1)[1:]:
            pose = np.eye(4)
            qq = (1 - f) * qa + f * qb
            pose[:3, :3] = T(np.zeros(3), qq / np.linalg.norm(qq))[:3, :3]
            pose[:3, 3] = (1 - f) * start[:3, 3] + f * goal[:3, 3]
            q, pe, re = ik_link6(model, planning, ARMS[a], pose, limits=lim, seeds=3)
            if q is None or pe > tol[0] or re > tol[1]:
                return False, f"{ARMS[a]} IK path unreachable (position {pe:.4f}, rotation {re:.3f})"
            way.append(q)
            planning.qpos[ids.qadr] = q
        paths[a] = way
    per = max(1, actions // n_way)
    for i in range(n_way):
        act = action_with(env, {a: paths[a][i] for a in paths}, {arm: grip} if grip is not None else None)
        run(env, act, per)
    return True, ""


def move_joints(env, arm: int, q_target, *, actions: int = 64, grip: float | None = None) -> None:
    """Joint-space interpolation of ``arm``'s targets from its current targets to ``q_target``."""
    q0 = current_action(env)[arm * 7:arm * 7 + 6].astype(np.float64)
    for i in range(int(actions)):
        f = (i + 1) / actions
        env.step(action_with(env, {arm: (1 - f) * q0 + f * np.asarray(q_target)},
                             {arm: grip} if grip is not None else None), render_obs=False)


# ---------------------------------------------------------------------------------------------------------- grasps

def Rz(yaw: float) -> np.ndarray:
    return np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])


def principal(points: np.ndarray):
    """(bbox centre, unit long axis, (t_min, t_max) along it) of world surface points."""
    c = 0.5 * (points.min(0) + points.max(0))
    _, U = np.linalg.eigh(np.cov((points - c).T))
    u = U[:, -1]
    t = (points - c) @ u
    return c, u, (float(t.min()), float(t.max()))


def section(points, c, u, t, band=0.012):
    """Centre of the item's cross-section at axis coordinate t."""
    tt = (points - c) @ u
    s = points[np.abs(tt - t) < band]
    if len(s) < 3:
        s = points[np.argsort(np.abs(tt - t))[:20]]
    return 0.5 * (s.min(0) + s.max(0))


def topdown_T6(point, axis_yaw, flip, below=0.0):
    """link_6 straight down, fingers closing across the horizontal direction ``axis_yaw``, pinch centre ``below``
    the point."""
    T6 = np.eye(4)
    T6[:3, :3] = Rz(axis_yaw + np.pi / 2 + (np.pi if flip else 0.0)) @ R_DOWN
    T6[:3, 3] = np.asarray(point) - np.array([0.0, 0.0, below]) - T6[:3, :3] @ PAD
    return T6


def side_T6(point, approach, roll):
    """link_6 z horizontal along ``approach``, rolled ``roll`` about it; pinch centre at the point."""
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cr, sr = np.cos(roll), np.sin(roll)
    T6 = np.eye(4)
    T6[:3, :3] = np.stack([x, y, z], axis=1) @ np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]])
    T6[:3, 3] = np.asarray(point) - T6[:3, :3] @ PAD
    return T6


def tilt_T6(T6, beta, axis_col=0):
    """Rotate a grasp pose by ``beta`` about link_6's own axis column (0 = closing axis), keeping the pinch centre
    fixed."""
    centre = T6[:3, 3] + T6[:3, :3] @ PAD
    Rr = T(np.zeros(3), axis_angle_quat(T6[:3, axis_col], beta))[:3, :3]
    out = np.eye(4)
    out[:3, :3] = Rr @ T6[:3, :3]
    out[:3, 3] = centre - out[:3, :3] @ PAD
    return out


def reachable(env, arm, T6, tilts=TILTS, tol=(2.5e-3, 0.04)):
    """First tilt of T6 (about its closing axis, then link_6 y) the arm's IK reaches -> (T6, q) or (None, None)."""
    lim = actuator_limits(env.model, ARMS[arm])
    for col, b in [(c, b) for c in (0, 1) for b in tilts]:
        if col == 1 and b == 0:
            continue
        Tt = tilt_T6(T6, b, col)
        q, pe, re = ik_link6(env.model, env.data, ARMS[arm], Tt, limits=lim, seeds=3)
        if q is not None and pe < tol[0] and re < tol[1]:
            return Tt, q
    return None, None


def item_geometry(env, scene, giver: int):
    """Long axis oriented towards the giver's side (lying items), extents, upright flag, world points."""
    pts = scene.world_points(env.data)
    c, u, (t0, t1) = principal(pts)
    upright = abs(u[2]) > 0.7
    sA = 1.0 if giver == 0 else -1.0
    if not upright and u[1] * sA < 0:
        u, t0, t1 = -u, -t1, -t0
    return dict(points=pts, c=c, u=u, t0=t0, t1=t1, L=t1 - t0, upright=upright, lo=pts.min(0), hi=pts.max(0))


def handoff_plan(env, scene, mode, giver, flipA, flipB):
    """(T6 giver, T6 receiver) for a grasp family (probe ``ih_handoff.py``), or None if the family doesn't apply."""
    g = item_geometry(env, scene, giver)
    sA = 1.0 if giver == 0 else -1.0
    c, u, t0, t1, L, v = g["c"], g["u"], g["t0"], g["t1"], g["L"], g["points"]
    yaw = np.arctan2(u[1], u[0]) if not g["upright"] else 0.0
    H = g["hi"][2] - g["lo"][2]
    if mode.startswith("mid_"):
        d = float(mode[-1]) / 100.0
        if g["upright"] or L < 2 * d + 0.04:
            return None
        t_com = float((np.asarray(env.data.subtree_com[scene.bodies[0]]) - c) @ u)
        tA, tB = min(t_com + d, t1 - 0.02), max(t_com - d, t0 + 0.02)
        mA, mB = section(v, c, u, tA), section(v, c, u, tB)
        TB = (topdown_T6(mB, yaw, flipB) if "topdown" in mode
              else side_T6(mB, np.array([0, sA, 0]), np.pi / 2 + np.pi * flipB))
        return topdown_T6(mA, yaw, flipA), TB
    if mode == "ends_topdown":
        if g["upright"] or L < 0.09:
            return None
        # B as close to the CoM as A's fingers allow: held alone at an end, long items creep 7-10 mm per chunk
        # (rolling pins, SEED_BASE + 8/11, first build round).
        t_com = float((np.asarray(env.data.subtree_com[scene.bodies[0]]) - c) @ u)
        tB = max(t0 + 0.15 * L, min(t_com - 0.06, t1 - 0.15 * L - 0.12))
        mA, mB = section(v, c, u, t1 - 0.15 * L), section(v, c, u, tB)
        # B closes on an item A holds rigidly: its site goes to the section centre (a weightless close, as in
        # hold_object, pulls the item into the fingers; a held one does not).
        return topdown_T6(mA, yaw, flipA), topdown_T6(mB, yaw, flipB, below=0.0)
    if mode in ("top_side", "side_side"):
        if H < 0.06:
            return None
        ptA = np.array([c[0], c[1], g["hi"][2] - min(0.025, 0.3 * H)])
        ptB = np.array([c[0], c[1], g["lo"][2] + min(0.03, 0.3 * H)])
        # Side grasps close horizontally; roll 0 / pi puts the pads (3 cm off link_6's axis) below / above it.
        TA = (topdown_T6(ptA, yaw, flipA, below=0.0) if mode == "top_side"
              else side_T6(ptA, np.array([0, -sA, 0]), np.pi * flipA))
        return TA, side_T6(ptB, np.array([0, sA, 0]), np.pi * flipB)
    if mode == "ends_side":
        if g["upright"] or L < 0.04:
            return None
        mA, mB = section(v, c, u, t1 - 0.15 * L), section(v, c, u, t0 + 0.12 * L)
        return topdown_T6(mA, yaw, flipA), side_T6(mB, np.array([0, sA, 0]), np.pi / 2 + np.pi * flipB)
    raise ValueError(mode)


def single_plans(env, scene, arm, *, below=0.0, side_from=None):
    """Grasps for one arm holding the item alone: top-down across the long axis at the CoM section (lying items),
    top-down on the upper part, side-on from ``side_from`` (+1/-1 along y) on the middle. Yields (name, T6)."""
    g = item_geometry(env, scene, arm)
    c, u, v = g["c"], g["u"], g["points"]
    com = np.asarray(env.data.subtree_com[scene.bodies[0]])
    yaw = np.arctan2(u[1], u[0])
    H = g["hi"][2] - g["lo"][2]
    s = (1.0 if arm == 0 else -1.0) if side_from is None else side_from
    out = []
    if not g["upright"]:
        t_com = float((com - c) @ u)
        for dt in (0.0, 0.03, -0.03):
            m = section(v, c, u, t_com + dt)
            for flip in (0, 1):
                out.append((f"topdown_mid{dt:+.2f}_{flip}", topdown_T6([m[0], m[1], m[2]], yaw, flip, below=below)))
    if H >= 0.05:
        top = np.array([c[0], c[1], g["hi"][2] - min(0.025, 0.3 * H)])
        for yy in (yaw, yaw + np.pi / 2):
            for flip in (0, 1):
                out.append((f"topdown_top_{flip}", topdown_T6(top, yy, flip, below=0.0)))
        for dz in (0.1, -0.1, 0.25):
            mid = np.array([c[0], c[1], c[2] + dz * H])
            for turn in (0.0, -0.6, 0.6):
                app = Rz(turn * s) @ np.array([0.0, -s, 0.0])
                for roll in (0.0, np.pi):
                    out.append((f"side{dz:+.2f}_{turn:+.1f}_{roll:.0f}", side_T6(mid, app, roll)))
    return out


def grasp_of(env, scene, arm, T6, opening=GRIP_MAX - 0.01):
    return {"arm": int(arm), "T_obj_link6": inv(scene.object_T(env.data, 0)) @ T6,
            "fingers": np.array([opening, opening])}


def arm_touches(env, scene, arm: int, *, ignore_item=False) -> list[str]:
    """Bodies (other than its own) touching ``arm``'s subtree."""
    m, d = env.model, env.data
    out = []
    for c in d.contact[:d.ncon]:
        a1, a2 = scene.arm_of[c.geom1], scene.arm_of[c.geom2]
        for x, y, gy in ((a1, a2, c.geom2), (a2, a1, c.geom1)):
            if x == arm and y != arm:
                if ignore_item and scene.owner[gy][0] == 0:
                    continue
                out.append(m.body(int(m.geom_bodyid[gy])).name or "world")
    return out


def teleport_arm(env, arm, q, opening=GRIP_MAX):
    ids = arm_ids(env.model, ARMS[arm])
    env.data.qpos[ids.qadr], env.data.qvel[ids.dadr] = q, 0.0
    env.data.qpos[ids.finger_qadr[0]], env.data.qpos[ids.finger_qadr[1]] = opening, -opening
    env.data.qvel[list(ids.finger_dadr)] = 0.0
    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------------------------------------- scene

def spawn_penetration(env, scene) -> float:
    pen = 0.0
    for c in env.data.contact[:env.data.ncon]:
        if scene.owner[c.geom1][0] == 0 or scene.owner[c.geom2][0] == 0:
            pen = min(pen, float(c.dist))
    return -pen


def settle(env, scene, max_actions=400, still=20):
    """Idle until the item has moved < 5 mm/s for ``still`` actions (linear only: some resting meshes keep a
    ~0.1 rad/s contact chatter with a static position, e.g. seed SEED_BASE + 7)."""
    act = env.get_state().astype(np.float32)
    d = int(scene.dof_adr[0])
    n = 0
    for i in range(max_actions):
        env.step(act, render_obs=False)
        n = n + 1 if np.linalg.norm(env.data.qvel[d:d + 3]) < 0.005 else 0
        if n >= still:
            return i + 1
    return -1


def reset_scene(env, seed: int, stage: str):
    """reset(seed) + settle; (ok, why, info). Skips the approved exclusions."""
    env.forget_arm_state()
    env.reset(seed=seed, randomize=True)
    md = dict(env._last_randomization.metadata)
    ev = env._task_evaluator
    scene = ev.scene
    info = dict(category=md["category"], variant=md["variant"], side=md["side"])
    pen = spawn_penetration(env, scene)
    info["spawn_pen_mm"] = round(1000 * pen, 1)
    if pen > SPAWN_OVERLAP_M:
        return False, "spawn overlap (decision 5)", info
    if stage in ("handoff", "carry", "place") and md["category"] in EXCLUDED_HANDOFF:
        return False, f"{md['category']} excluded from {stage} (decision 4)", info
    n = settle(env, scene)
    info["settle_actions"] = n
    if n < 0:
        return False, "item does not settle", info
    ev.begin(env.model, env.data, 0)  # rest = the settled spot
    if ev.tracker.obs["stock_done"][0] or ev.tracker.obs["ys"][0] <= -0.10:
        return False, "already on the far side after settling", info
    return True, "", info


def snapshot(env):
    return (env.data.qpos.copy(), env.data.qvel.copy(), env.data.act.copy(), env.data.ctrl.copy(),
            float(env.data.time))


def restore(env, snap):
    env.data.qpos[:], env.data.qvel[:], env.data.act[:], env.data.ctrl[:], env.data.time = snap
    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------------------------------------------- holds

def two_arm_hold_check(env, scene, arm, steps=32, settle_steps=16) -> tuple[bool, float]:
    """Hold ``settle_steps`` (the item settles / pivots into the pinch), then the CoM may sink at most MAX_CREEP_M
    over one more chunk."""
    run(env, current_action(env), settle_steps)
    b = scene.bodies[0]
    z0 = float(env.data.subtree_com[b][2])
    run(env, current_action(env), steps)
    creep = z0 - float(env.data.subtree_com[b][2])
    return creep <= MAX_CREEP_M, creep


def hold_at(env, scene, arm, pos_com, plans, rng, *, quat=None, deadline=None):
    """Move the item (weightless) so its CoM is at pos_com (orientation ``quat`` or current), then let ``arm`` hold it
    with the first of ``plans`` (callable -> [(name, T6)]) that is reachable and passes the kit hold checks and a
    one-chunk creep check. -> (ok, info)."""
    a = int(scene.qpos_adr[0])
    q0 = env.data.qpos[a + 3:a + 7].copy() if quat is None else np.asarray(quat)
    off = env.data.qpos[a:a + 3] - env.data.subtree_com[scene.bodies[0]]
    Rq = T(np.zeros(3), q0)[:3, :3]
    R_now = T(np.zeros(3), env.data.qpos[a + 3:a + 7])[:3, :3]
    origin = np.asarray(pos_com) + Rq @ (R_now.T @ off)
    set_object_pose(env, scene, 0, origin, q0)
    snap = snapshot(env)
    whys = []
    for name, T6 in plans():  # planned on the moved item
        if deadline and time.monotonic() > deadline:
            whys.append("time budget")
            break
        restore(env, snap)
        Tt, q = reachable(env, arm, T6)
        if Tt is None:
            whys.append(f"{name}: no IK")
            continue
        teleport_arm(env, arm, q)  # hold_object's IK then starts on this branch (another can collide)
        ok, info = hold_object(env, scene, 0, grasp_of(env, scene, arm, Tt), rng=rng)
        if not ok:
            whys.append(f"{name}: {info.get('why')}")
            continue
        good, creep = two_arm_hold_check(env, scene, arm)
        if not good:
            whys.append(f"{name}: creeps {1000 * creep:.1f} mm")
            continue
        return True, dict(grasp=name, creep_mm=round(1000 * creep, 2), drift=info.get("drift"))
    return False, dict(why="; ".join(whys[-4:]) or "no grasp")


# ---------------------------------------------------------------------------------------------------------- handoff

def handoff_setup(env, scene, giver, deadline, rng):
    """Item to the middle at a yaw/mid from YAWS x MIDS, A holds it with a grasp family for which B has a reachable
    grasp too. -> (ok, info incl. mode/flipB for the solver)."""
    a = int(scene.qpos_adr[0])
    snap = snapshot(env)
    whys = []
    combos = [(mid, y, mode, fa, fb) for mid in MIDS for y in YAWS for mode in MODES for fa in (0, 1) for fb in (0, 1)]
    na = set()  # (mid, yaw, mode) whose family does not apply to this item
    for mid, yaw_axis, mode, fa, fb in combos:
        if (mid, yaw_axis, mode) in na:
            continue
        if time.monotonic() > deadline:
            return False, dict(why="time budget; " + "; ".join(whys[-3:]))
        restore(env, snap)
        g = item_geometry(env, scene, giver)
        sA = 1.0 if giver == 0 else -1.0
        phi = 0.0 if g["upright"] else yaw_axis - np.arctan2(g["u"][1], g["u"][0]) + (0.0 if sA > 0 else np.pi)
        Rm = Rz(phi)
        p, q = env.data.qpos[a:a + 3].copy(), env.data.qpos[a + 3:a + 7].copy()
        set_object_pose(env, scene, 0, np.array([mid[0], 0.0, mid[1]]) + Rm @ (p - g["c"]), qmul(quat_of(Rm), q))
        P = handoff_plan(env, scene, mode, giver, fa, fb)
        if P is None:
            na.add((mid, yaw_axis, mode))
            continue
        TA, qA = reachable(env, giver, P[0])
        if TA is None:
            whys.append(f"{mode}: A no IK")
            continue
        got = check_receiver(env, scene, 1 - giver, P[1])
        if got is None:
            whys.append(f"{mode}: B no IK / open gripper collides")
            continue
        TB = got[0]
        T_objB = inv(scene.object_T(env.data, 0)) @ TB
        teleport_arm(env, giver, qA)
        ok, info = hold_object(env, scene, 0, grasp_of(env, scene, giver, TA), rng=rng)
        if not ok:
            whys.append(f"{mode}: A {info.get('why')}")
            continue
        good, creep = two_arm_hold_check(env, scene, giver)
        if not good:
            whys.append(f"{mode}: A creeps {1000 * creep:.1f} mm")
            continue
        if check_receiver(env, scene, 1 - giver, scene.object_T(env.data, 0) @ T_objB) is None:
            whys.append(f"{mode}: B's grasp collides with A's hold")  # re-checked with A's gripper in place
            continue
        return True, dict(mode=mode, flipA=fa, flipB=fb, yaw=round(float(np.rad2deg(yaw_axis))), mid=mid,
                          creep_mm=round(1000 * creep, 2), T_objB=T_objB.round(6).tolist())
    return False, dict(why="; ".join(whys[-4:]) or "no plan")


def receiver_grasps(env, scene, giver, hint=None, limit=4):
    """Up to ``limit`` reachable grasps of B for the current item pose whose open gripper touches nothing: the hinted
    one (object-frame link_6 pose) first, then top-down sections on B's half nearest the CoM (lying items), then the
    handoff families. -> [(T6, q)]."""
    B = 1 - giver
    cands = []
    if hint is not None:
        cands.append(scene.object_T(env.data, 0) @ np.asarray(hint))
    g = item_geometry(env, scene, giver)
    if not g["upright"] and g["L"] >= 0.09:
        c, u, v = g["c"], g["u"], g["points"]
        t_com = float((np.asarray(env.data.subtree_com[scene.bodies[0]]) - c) @ u)
        yaw = np.arctan2(u[1], u[0])
        for d in (0.05, 0.07, 0.09, 0.03):
            t = max(g["t0"] + 0.1 * g["L"], t_com - d)
            for flip in (0, 1):
                cands.append(topdown_T6(section(v, c, u, t), yaw, flip, below=0.0))
    for mode in MODES:
        for fa in (0, 1):
            for fb in (0, 1):
                P = handoff_plan(env, scene, mode, giver, fa, fb)
                if P is not None:
                    cands.append(P[1])
    out = []
    for T6 in cands:
        got = check_receiver(env, scene, B, T6)
        if got is not None and not any(np.allclose(got[0], o[0], atol=1e-3) for o in out):
            out.append(got)
            if len(out) >= limit:
                break
    Tinv = inv(scene.object_T(env.data, 0))
    return [Tinv @ Tt for Tt, _ in out]  # object frame: re-anchored at each attempt (the item shifts)


def check_receiver(env, scene, B, T6):
    """(T6 at a reachable tilt, q) if B's IK reaches it and the open gripper there touches nothing, else None."""
    Tt, q = reachable(env, B, T6)
    if Tt is None:
        return None
    snap = snapshot(env)
    teleport_arm(env, B, q)
    hit = arm_touches(env, scene, B)
    restore(env, snap)
    return None if hit else (Tt, q)


def approach_pose(env, arm, T6, q):
    """Joint target of a pre-grasp 8/5/3 cm back along link_6 z from T6, on q's IK branch (None if unreachable)."""
    ids = arm_ids(env.model, ARMS[arm])
    planning = mujoco.MjData(env.model)
    planning.qpos[:] = env.data.qpos
    planning.qpos[ids.qadr] = q
    for back in (0.08, 0.05, 0.03):
        pre = T6.copy()
        pre[:3, 3] -= T6[:3, 2] * back
        qpre, pe, re = ik_link6(env.model, planning, ARMS[arm], pre, limits=actuator_limits(env.model, ARMS[arm]),
                                seeds=2)
        if qpre is not None and pe < 0.005 and re < 0.1:
            return qpre
    return None


def monitored_release(env, scene, giver, sag_m=0.02, speed_mps=0.03) -> bool:
    """A opens in steps while B holds; if the CoM sags > sag_m or B loses a finger, or the item still moves at the
    end (a slow slip), A closes again (False). The item may pivot a little in B's pinch; that is not a slip."""
    B = 1 - giver
    b, d = scene.bodies[0], int(scene.dof_adr[0])
    z0 = float(env.data.subtree_com[b][2])
    for grip, n in ((0.35, 4), (0.7, 4), (1.0, 4), (1.0, 4), (1.0, 6), (1.0, 6)):  # last 12: catch slow slips
        run(env, action_with(env, None, {giver: grip}), n)
        f = scene.contact_obs(env.data)["hits"].reshape(2, 2).sum(1)
        if z0 - float(env.data.subtree_com[b][2]) > sag_m or f[B] < 2:
            run(env, action_with(env, None, {giver: 0.0}), 16)
            return False
    if np.linalg.norm(env.data.qvel[d:d + 3]) > speed_mps:
        run(env, action_with(env, None, {giver: 0.0}), 16)
        return False
    return True


def handoff_solver(env, scene, giver, *, hint=None, teleport=False, tries=3) -> tuple[bool, str]:
    """B moves (joint space) to a pre-grasp, then straight onto its grasp, and closes; only once both B fingers are
    on the item (else B opens, backs off and tries its next grasp) A opens, lifts 10 cm and backs off 6 cm.
    ``teleport`` (start-building fallback only) puts B at the grasp pose instead of approaching."""
    B = 1 - giver
    rest = None
    whys = []
    for T_obj in receiver_grasps(env, scene, giver, hint, limit=tries):
        got = check_receiver(env, scene, B, scene.object_T(env.data, 0) @ T_obj)
        if got is None:
            whys.append("grasp no longer reachable")
            continue
        TB, qB = got
        if teleport:
            teleport_arm(env, B, qB)
            run(env, action_with(env, {B: qB}, {B: 1.0}), 4)
            qpre = None
        else:
            qpre = approach_pose(env, B, TB, qB)
            if qpre is None:
                whys.append("pre-grasp unreachable")
                continue
            move_joints(env, B, qpre, actions=96, grip=1.0)
            ok, why = move_link(env, B, TB, actions=48, grip=1.0, n_way=4)
            if not ok:
                whys.append("grasp: " + why)
                move_joints(env, B, qpre, actions=32, grip=1.0)
                continue
        run(env, action_with(env, None, {B: 0.0}), 24)  # close
        o = scene.contact_obs(env.data)["hits"].reshape(2, 2).sum(1)
        if o[B] == 2 and o[giver] == 2 and monitored_release(env, scene, giver):
            break
        whys.append(f"receiver closed with {o[B]} fingers" if o[B] < 2 else "receiver could not hold it alone")
        run(env, action_with(env, None, {B: 1.0}), 12)
        if qpre is not None:
            move_joints(env, B, qpre, actions=32, grip=1.0)
    else:
        return False, "; ".join(whys) or "no reachable receiver grasp"
    retreat(env, giver)
    run(env, current_action(env), 16)
    return True, ""


def retreat(env, arm) -> None:
    """Move the open ``arm`` away: up 10 cm and 6 cm back to its side, else up 8 cm, else (joint space) home."""
    s = 1.0 if arm == 0 else -1.0
    start = link6_T(env, arm)
    for d in (np.array([0.0, 0.06 * s, 0.10]), np.array([0.0, 0.0, 0.08])):
        up = start.copy()
        up[:3, 3] += d
        ok, _ = move_link(env, arm, up, actions=48, grip=1.0, n_way=6, tol=(0.01, 0.3))
        if ok:
            return
    move_joints(env, arm, env.get_init_q()[arm * 7:arm * 7 + 6], actions=64, grip=1.0)


# ---------------------------------------------------------------------------------------------------------- build

def build_start(env, seed: int, stage: str, *, rng: np.random.Generator | None = None,
                cfg: StageConfig | None = None, budget_s: float = BUILD_BUDGET_S):
    """Make a ``stage`` start for ``seed`` on an env made for an ``inhand_transfer_item_to_other_side__<stage>``
    task. -> (start or None, log)."""
    cfg = cfg or StageConfig()
    rng = rng or np.random.default_rng(seed)
    t0 = time.monotonic()
    deadline = t0 + budget_s
    si = STAGES.index(stage)
    ok, why, info = reset_scene(env, seed, stage)
    info = dict(seed=int(seed), **info)
    if not ok:
        return None, [{**info, "why": why, "skipped": True}]
    ev = env._task_evaluator
    scene = ev.scene
    A = scene.giver
    B = 1 - A
    rest = ev.tracker.rest_height.copy()
    arm = A
    base = snapshot(env)
    if stage == "pick":
        ok, gi = pick_setup(env, scene, A, rng, deadline)
    elif stage == "present":
        for attempt in range(3):  # a new random height per attempt
            restore(env, base)
            com = rest[0].copy()
            com[1] = scene.sign * max(abs(com[1]), 0.14)
            com[2] += rng.uniform(0.05, 0.15)
            ok, gi = hold_at(env, scene, A, com, lambda: single_plans(env, scene, A), rng, deadline=deadline)
            gi["attempts"] = attempt + 1
            if ok or time.monotonic() > deadline:
                break
    elif stage == "handoff":
        ok, gi = handoff_setup(env, scene, A, deadline, rng)
    elif stage == "carry":
        arm = B
        # Source 1 (chain): a handoff start + the scripted handoff solver, at most 5 s. Any rejection of its result
        # (B not holding alone in the air, creep, stage test) falls back to source 3 with the rest of the budget
        # (carry G3 repair round 3).
        ok, gi = handoff_setup(env, scene, A, min(deadline, t0 + 5.0), rng)
        why = gi.get("why", "")
        if ok:
            hint = np.asarray(gi.pop("T_objB"))
            ok, why = handoff_solver(env, scene, A, hint=hint)
            if ok:
                o = scene.observe(env.data, rest)
                ok = bool(o["f_recv"][0] == 2 and o["f_give"][0] == 0 and not o["env_touch"][0])
                why = "" if ok else "handoff solver ended without B holding it alone in the air"
            if ok:
                ok, creep = two_arm_hold_check(env, scene, B)
                gi["creep_mm"] = round(1000 * creep, 2)
                why = "" if ok else f"creeps {1000 * creep:.1f} mm"
            if ok:
                o = scene.observe(env.data, rest)
                lab = classify(o, classify(o, classify(o, np.full(1, R), cfg), cfg), cfg)
                ok = int(lab[0]) == si
                why = "" if ok else f"chain ends in {LABELS[int(lab[0])]}"
        gi["source"] = "chained handoff solver"
        if not ok:
            gi["chain_failure"] = why
            restore(env, base)
            mid = np.array([0.55, -scene.sign * 0.03, rest[0, 2] + 0.12])
            ok, gb = hold_at(env, scene, B, mid, lambda: single_plans(env, scene, B), rng,
                             deadline=max(deadline, time.monotonic() + 2.0))
            gi.update(source="sampled receiver grasp in the middle (grasp source 3)", grasp=gb.get("grasp"),
                      creep_mm=gb.get("creep_mm"))
            gi.pop("why", None)
            if not ok:
                gi["why"] = gb.get("why", "")
    elif stage == "place":
        arm = B
        sB = -scene.sign
        for attempt in range(4):  # a new random far-zone point per attempt
            restore(env, base)
            com = np.array([rng.uniform(0.47, 0.68), sB * rng.uniform(0.17, 0.32),
                            rest[0, 2] + rng.uniform(0.04, 0.10)])
            ok, gi = hold_at(env, scene, B, com, lambda: single_plans(env, scene, B), rng,
                             deadline=deadline)
            gi.update(far_xy=com[:2].round(3).tolist(), attempts=attempt + 1)
            if ok or time.monotonic() > deadline:
                break
    else:
        ok, gi = True, {}
    info.update(gi)
    info["build_s"] = round(time.monotonic() - t0, 2)
    if not ok:
        return None, [{**info, "why": info.get("why", "setup failed")}]
    obs = scene.observe(env.data, rest)
    labels = classify(obs, np.full(1, R), cfg)
    labels = classify(obs, labels, cfg)
    labels = classify(obs, labels, cfg)
    info.update(label=LABELS[int(labels[0])], lift=round(float(obs["lift"][0]), 3),
                gap_give=round(float(obs["gap_give"][0]), 3))
    if int(labels[0]) != si:
        return None, [{**info, "why": f"stage test says {info['label']}, not {stage}"}]
    state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
    return dict(seed=int(seed), stage=stage, target=0, arm=int(arm) if si > R else -1, state=state,
                rest_height=rest, labels=np.asarray(labels), info=info), [{**info, "ok": True}]


def pick_setup(env, scene, A, rng, deadline):
    """A open around the resting item: top-down grasps at the CoM section / upper part, raised until the open
    fingers touch nothing; then 10 open-gripper actions must not move the item."""
    snap = snapshot(env)
    a = int(scene.qpos_adr[0])
    p0 = env.data.qpos[a:a + 3].copy()
    whys = []
    for name, T6 in single_plans(env, scene, A):
        if name.startswith("side"):
            continue
        for lift in (0.0, 0.01, 0.02, 0.03):  # pads reach 2.8 cm below the pinch centre
            if time.monotonic() > deadline:
                return False, dict(why="time budget; " + "; ".join(whys[-3:]))
            restore(env, snap)
            T6l = T6.copy()
            T6l[2, 3] += lift
            Tt, q = reachable(env, A, T6l)
            if Tt is None:
                whys.append(f"{name}: no IK")
                break
            teleport_arm(env, A, q)
            hit = arm_touches(env, scene, A)
            if hit:
                whys.append(f"{name}+{lift}: open fingers touch {sorted(set(hit))[:2]}")
                continue
            run(env, action_with(env, {A: q}, {A: 1.0}), 10)
            moved = float(np.linalg.norm(env.data.qpos[a:a + 3] - p0))
            if moved > 0.005 or arm_touches(env, scene, A):
                whys.append(f"{name}+{lift}: item disturbed {moved:.3f}")
                continue
            return True, dict(grasp=name, raise_m=lift)
    return False, dict(why="; ".join(whys[-4:]) or "no grasp")
