"""Handoff geometry and the scripted handoff for put_relative cross-table episodes (see ``put_relative_stages``).

A cross-table episode has its goal zone on the other side of the table from the mover (the zone point, 14 cm along
the prompted direction from the reference, has the other y sign). The giver G is the arm on the mover's side (the arm
the pick starts use), the receiver V the arm on the zone's side.

``MoverView`` is an ``ObjectScene`` over the mover alone (plus a geom -> arm map), so the inhand helpers in
``inhand_starts`` (IK reachability with tilts, open-gripper collision checks, monitored release, retreat, joint/
Cartesian moves) work on it unchanged. Grasps put the **pinch centre** ``PAD`` (between the finger pads, link_6
(0, -0.031, 0.125)) on the object: the ``<arm>_grasp_site`` is 3 cm beside the pads (inhand finding).

put_relative objects collide as one box each (visual meshes are larger); ``box_gap`` is the exact distance from a
point to that box. The scripted handoff (``plan_handoff`` + ``run_handoff``):
  1. G lifts the held mover to a handoff height, then carries it to a handoff point in the middle (x 0.50-0.60,
     y 0) with one of a few yaw turns about its CoM;
  2. V moves (joint space) to a pre-grasp, then straight onto a grasp on the mover's free part (top-down on the
     V end, or side-on from V's side), closes; only once both V fingers are on it does G open (monitored release);
  3. G backs off (up 10 cm and 6 cm towards its side).
The plan (handoff pose + V grasp in the mover's frame) is found kinematically first (teleports on a snapshot).
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from abc_sim.task_eval.inhand_starts import (PAD, action_with, approach_pose, arm_touches, current_action, tilt_T6,
                                             link6_T, monitored_release, move_joints, move_link, restore, retreat,
                                             run, side_T6, snapshot, teleport_arm, topdown_T6)
from abc_sim.task_eval.stagekit import ARMS, GRIP_MAX, T, actuator_limits, arm_ids, axis_angle_quat, ik_link6, inv, quat_of
from abc_sim.task_eval.stagekit.scene import ObjectScene, geoms_of, subtree

# Handoff poses, tried in order: CoM point (x, height above the table top; y = 0), a yaw turn of the mover about
# world z, then a tilt about G's closing axis (G presents the mover with its wrist leaning back towards its own side;
# 90 deg = G's fingers pointing at V). Keeping link_6 straight down in the middle is often beyond the wrist (probe
# handoff_probe try3: 17/24 straight-down poses unreachable for a tomato).
# Points: (x, y towards G's side, height). Points whose xy lies in the goal zone are skipped (``avoid``): G holding the
# mover there is "carried it to the goal" (zones that straddle the middle: directions along y).
HANDOFF_POINTS = ((0.55, 0.0, 0.15), (0.50, 0.0, 0.13), (0.60, 0.0, 0.13), (0.45, 0.0, 0.13), (0.65, 0.0, 0.13),
                  (0.55, 0.08, 0.15), (0.55, 0.15, 0.15))
HANDOFF_YAWS = np.deg2rad([0, 90, -90, 180])
HANDOFF_TILTS = np.deg2rad([0, 45, -45, 90, -90])
RECV_TILTS = np.deg2rad([0, 25, -25, 45, -45])
TABLE_Z = 0.75


def zone_avoider(scene, rest: np.ndarray, margin: float = 0.03):
    """xy -> True when it lies within ``margin`` of the goal zone (the stock cone from the reference's rest spot)."""
    cfg = scene.cfg
    axis = np.asarray(scene.goal["axis"], dtype=np.float64)
    ref = np.asarray(rest)[scene.reference, :2]

    def inside(xy) -> bool:
        rel = np.asarray(xy, dtype=np.float64)[:2] - ref
        along, perp = float(rel @ axis), abs(float(rel @ np.array([-axis[1], axis[0]])))
        gap = np.hypot(along - np.clip(along, cfg.along_min_m, cfg.along_max_m), perp - min(perp, cfg.perp_max_m))
        return bool(gap <= margin)
    return inside


def zone_point(rest: np.ndarray, reference: int, axis) -> np.ndarray:
    """The randomizer's goal point (14 cm along the direction from the reference's rest spot)."""
    return np.asarray(rest)[reference, :2] + np.asarray(axis, dtype=np.float64) * 0.14


def sides(rest: np.ndarray, mover: int, reference: int, axis) -> tuple[bool, int, int]:
    """(cross-table, giver arm, receiver arm): giver = the arm on the mover's side (0 = left for y > 0)."""
    rest = np.asarray(rest)
    giver = 0 if rest[mover, 1] > 0 else 1
    cross = bool(np.sign(zone_point(rest, reference, axis)[1]) != np.sign(rest[mover, 1]))
    return cross, giver, 1 - giver


def collision_geoms(model: mujoco.MjModel, body: int) -> np.ndarray:
    return np.asarray([g for g in geoms_of(model, subtree(model, body))
                       if model.geom_contype[g] or model.geom_conaffinity[g]], dtype=np.int64)


def box_gap(model: mujoco.MjModel, data: mujoco.MjData, geoms, points: np.ndarray) -> np.ndarray:
    """Distance from each point (n, 3) to the nearest collision geom of an object (0 inside). Boxes exactly; other
    geom types by their bounding sphere (put_relative objects are all boxes)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    best = np.full(len(points), np.inf)
    for g in geoms:
        p = (points - data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            d = np.linalg.norm(np.maximum(np.abs(p) - model.geom_size[g], 0.0), axis=1)
        else:
            d = np.maximum(np.linalg.norm(p, axis=1) - float(model.geom_rbound[g]), 0.0)
        best = np.minimum(best, d)
    return best


class MoverView(ObjectScene):
    """The mover alone, as object 0, with ``arm_of`` (geom -> 0 left arm, 1 right arm, -1 other)."""

    def __init__(self, model: mujoco.MjModel, name: str, joint: int) -> None:
        super().__init__(model, [(name, joint)])
        self.arm_of = arm_map(model)
        self.geoms = collision_geoms(model, self.bodies[0])


def arm_map(model: mujoco.MjModel) -> np.ndarray:
    arm_of = np.full(model.ngeom, -1, dtype=np.int64)
    for a, root in enumerate(("left_arm", "right_arm")):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
        if b < 0:
            raise ValueError(f"{root} not found")
        arm_of[geoms_of(model, subtree(model, b))] = a
    return arm_of


def pad_centre(data: mujoco.MjData, link6: int) -> np.ndarray:
    return data.xpos[link6] + data.xmat[link6].reshape(3, 3) @ PAD


def _box(model, data, view):
    g = int(view.geoms[0])
    return np.array(data.geom_xpos[g]), data.geom_xmat[g].reshape(3, 3).copy(), np.array(model.geom_size[g])


def receiver_candidates(env, view: MoverView, V: int) -> list[tuple[str, np.ndarray]]:
    """Grasps of V on the mover's V end (pinch centre on the object): top-down closing across / along the V-facing
    axis, and side-on from V's side closing horizontally or vertically. -> [(name, T6)]."""
    c, Rb, h = _box(env.model, env.data, view)
    sV = 1.0 if V == 0 else -1.0
    toward_v = np.array([0.0, sV, 0.0])
    vert = int(np.argmax(np.abs(Rb[2, :])))
    horiz = [i for i in range(3) if i != vert]
    i_n = max(horiz, key=lambda i: abs(Rb[:, i] @ toward_v))
    i_t = [i for i in horiz if i != i_n][0]
    n = Rb[:, i_n] * np.sign(Rb[:, i_n] @ toward_v)
    up = Rb[:, vert] * np.sign(Rb[2, vert])
    hn, ht, hv = h[i_n], h[i_t], h[vert]
    yaw_n = float(np.arctan2(n[1], n[0]))
    yaw_t = float(np.arctan2(Rb[1, i_t], Rb[0, i_t]))
    out = []
    # Side-on from V's side first: closing vertically (pads on the top and bottom faces, between G's fingers), then
    # horizontally on the lower part; top-down on the V end last (G's gripper is usually above it).
    for d_in in (0.015, 0.025):
        if d_in > 2 * hn - 0.005:
            continue
        p = c + n * (hn - d_in)
        for roll in (np.pi / 2, -np.pi / 2):
            out.append((f"side_vert_{d_in}_{roll:.2f}", side_T6(p, -n, roll)))
    for d_in in (0.015,):
        if d_in > 2 * hn - 0.005:
            continue
        for fz in (-0.5, -0.25):
            p = c + n * (hn - d_in) + up * (fz * hv)
            for roll in (0.0, np.pi):
                out.append((f"side_horiz_{d_in}_{fz}_{roll:.2f}", side_T6(p, -n, roll)))
    for d_in in (0.02,):
        if d_in > 2 * hn - 0.01:
            continue
        for dz in (0.015,):
            p = c + n * (hn - d_in) + up * (hv - min(dz, hv))
            for yaw, tag in ((yaw_n, "across"), (yaw_t, "along")):
                for flip in (0, 1):
                    out.append((f"top_{tag}_{d_in}_{dz}_{flip}", topdown_T6(p, yaw, flip)))
    return out


def closes_clear(env, view: MoverView, V: int, q) -> bool:
    """V at joints q with its fingers closed onto the box (opening = the box's half extent along V's closing axis
    + 3 mm) touches nothing but the mover (catches fingers that would land on G's)."""
    c, Rb, h = _box(env.model, env.data, view)
    snap = snapshot(env)
    teleport_arm(env, V, q)
    l6 = arm_ids(env.model, ARMS[V]).link6
    x = env.data.xmat[l6].reshape(3, 3)[:, 0]
    half = float(np.sum(np.abs(x @ Rb) * h))
    teleport_arm(env, V, q, opening=min(GRIP_MAX, half + 0.003))
    hit = arm_touches(env, view, V, ignore_item=True)
    restore(env, snap)
    return not hit


def giver_grasps(env, view: MoverView, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Top-down grasps of the mover with the pinch centre on its vertical axis at max(centre, top - 2.5 cm) (tall
    objects keep their lower part free for the receiver), closing across one of the box's horizontal faces; the four
    (axis, flip) choices in a random order."""
    c, Rb, h = _box(env.model, env.data, view)
    vert = int(np.argmax(np.abs(Rb[2, :])))
    horiz = [i for i in range(3) if i != vert]
    top = c[2] + h[vert]
    p = np.array([c[0], c[1], max(c[2], top - 0.025)])
    out = []
    for i in horiz:
        yaw = float(np.arctan2(Rb[1, i], Rb[0, i]))  # fingers close across axis i = along the other horizontal axis
        for flip in (0, 1):
            out.append((f"giver_top_{i}_{flip}", topdown_T6(p, yaw, flip)))
    order = rng.permutation(len(out))
    return [out[j] for j in order]


def handoff_pose(env, view: MoverView, G: int, point, yaw: float, tilt: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(G's link_6 target, mover pose 4x4): the held mover turned ``yaw`` about world z, then ``tilt`` about G's
    closing axis (both through its CoM), with its CoM at ``point``. G's grasp of it is unchanged."""
    com = np.asarray(env.data.subtree_com[view.bodies[0]], dtype=np.float64)
    T6 = link6_T(env, G)
    Rz = T(np.zeros(3), axis_angle_quat(np.array([0.0, 0.0, 1.0]), yaw))[:3, :3]
    Rt = T(np.zeros(3), axis_angle_quat(Rz @ T6[:3, 0], tilt))[:3, :3]
    M = np.eye(4)
    M[:3, :3] = Rt @ Rz
    M[:3, 3] = np.asarray(point) - M[:3, :3] @ com
    return M @ T6, M @ view.object_T(env.data, 0)


def reach_fast(env, arm: int, T6, tilts=RECV_TILTS, tol=(2.5e-3, 0.04)):
    """``inhand_starts.reachable`` with a cheap first pass: per tilt one short IK run (80 iterations from the current
    joints); only poses it gets within 3 cm of are refined with the full IK (3 seeds). Same acceptance tolerance."""
    lim = actuator_limits(env.model, ARMS[arm])
    for col, b in [(c, b) for c in (0, 1) for b in tilts]:
        if col == 1 and b == 0:
            continue
        Tt = tilt_T6(T6, b, col)
        q, pe, re = ik_link6(env.model, env.data, ARMS[arm], Tt, limits=lim, seeds=1, iters=80)
        if q is None or pe > 0.03:
            continue
        if not (pe < tol[0] and re < tol[1]):
            q, pe, re = ik_link6(env.model, env.data, ARMS[arm], Tt, limits=lim, seeds=3)
        if q is not None and pe < tol[0] and re < tol[1]:
            return Tt, q
    return None, None


def check_recv(env, view: MoverView, V: int, T6) -> tuple[np.ndarray, np.ndarray] | None:
    """``inhand_starts.check_receiver`` with fewer tilts (RECV_TILTS) and ``reach_fast``: (T6 at a reachable tilt,
    q) if V's open gripper there touches nothing, else None."""
    Tt, q = reach_fast(env, V, T6)
    if Tt is None:
        return None
    snap = snapshot(env)
    teleport_arm(env, V, q)
    hit = arm_touches(env, view, V)
    restore(env, snap)
    return None if hit else (Tt, q)


def cart_path(env, arm: int, goals, *, n_way: int = 8, tol=(0.008, 0.2), q0=None) -> tuple[list | None, str]:
    """Joint waypoints for straight link_6 segments through the poses ``goals`` (sequential IK seeded from the
    previous waypoint, like ``inhand_starts.move_link``), starting from the current pose / joints (or ``q0``)."""
    model = env.model
    ids_q = arm_ids(model, ARMS[arm]).qadr
    planning = mujoco.MjData(model)
    planning.qpos[:] = env.data.qpos
    if q0 is not None:
        planning.qpos[ids_q] = q0
    mujoco.mj_kinematics(model, planning)
    l6 = arm_ids(model, ARMS[arm]).link6
    start = T(planning.xpos[l6], planning.xquat[l6])
    lim = actuator_limits(model, ARMS[arm])
    way = []
    for goal in goals:
        qa, qb = quat_of(start[:3, :3]), quat_of(goal[:3, :3])
        if qa @ qb < 0:
            qb = -qb
        for f in np.linspace(0, 1, n_way + 1)[1:]:
            pose = np.eye(4)
            qq = (1 - f) * qa + f * qb
            pose[:3, :3] = T(np.zeros(3), qq / np.linalg.norm(qq))[:3, :3]
            pose[:3, 3] = (1 - f) * start[:3, 3] + f * goal[:3, 3]
            q, pe, re = ik_link6(model, planning, ARMS[arm], pose, limits=lim, seeds=3)
            if q is None or pe > tol[0] or re > tol[1]:
                return None, f"{ARMS[arm]} IK path unreachable (position {pe:.4f}, rotation {re:.3f})"
            way.append(q)
            planning.qpos[ids_q] = q
        start = goal
    return way, ""


def follow(env, arm: int, way, *, actions: int, grip: float | None = None) -> None:
    """Joint targets linearly interpolated from the current targets through the waypoints over ``actions`` steps."""
    pts = [current_action(env)[arm * 7:arm * 7 + 6].astype(np.float64)] + [np.asarray(q) for q in way]
    n = len(pts) - 1
    for i in range(int(actions)):
        s = (i + 1) / actions * n
        j = min(int(s), n - 1)
        q = pts[j] + (s - j) * (pts[j + 1] - pts[j])
        env.step(action_with(env, {arm: q}, {arm: grip} if grip is not None else None), render_obs=False)


def giver_goals(env, G: int, T6_goal: np.ndarray) -> list:
    """Straight up to the goal height (if higher), then over to the goal pose."""
    start = link6_T(env, G)
    goals = []
    if T6_goal[2, 3] > start[2, 3] + 0.01:
        up = start.copy()
        up[2, 3] = T6_goal[2, 3]
        goals.append(up)
    return goals + [T6_goal]


def receiver_options(env, view: MoverView, V: int, limit: int = 3) -> list[tuple[str, np.ndarray]]:
    """Up to ``limit`` V grasps for the current state, in the mover's frame: reachable, open gripper touching
    nothing, closed fingers clear of everything but the mover, and a reachable pre-grasp."""
    out = []
    Tinv = inv(view.object_T(env.data, 0))
    for name, TV in receiver_candidates(env, view, V):
        got = check_recv(env, view, V, TV)
        if got is None or not closes_clear(env, view, V, got[1]) or approach_pose(env, V, got[0], got[1]) is None:
            continue
        out.append((name, Tinv @ got[0]))
        if len(out) >= limit:
            break
    return out


def plan_handoff(env, view: MoverView, G: int, *, deadline: float | None = None, max_plans: int = 1,
                 avoid=None) -> list[dict]:
    """Kinematic plans: a handoff point/yaw G can carry the mover to (joint path planned from G's current joints, so
    the arm's configuration there is the one the move ends in), and a V grasp (mover frame) that V reaches with its
    open gripper touching nothing (G and the mover in place). ``avoid(xy) -> bool`` skips handoff points (the goal
    zone). Leaves the state unchanged."""
    model, data = env.model, env.data
    V = 1 - G
    snap = snapshot(env)
    a = int(view.qpos_adr[0])
    opening = float(data.qpos[_finger(model, G)])
    lim = actuator_limits(model, ARMS[G])
    plans = []
    try:
        sG = 1.0 if G == 0 else -1.0
        for x, y, dz in HANDOFF_POINTS:
            if avoid is not None and avoid(np.array([x, sG * y])):
                continue
            for yaw in HANDOFF_YAWS:
                for tilt in HANDOFF_TILTS:
                    if deadline is not None and time.process_time() > deadline:
                        return plans
                    restore(env, snap)
                    com_now = np.asarray(data.subtree_com[view.bodies[0]])
                    point = np.array([x, sG * y, TABLE_Z + dz + (com_now[2] - data.qpos[a + 2])])
                    T6, Tobj = handoff_pose(env, view, G, point, yaw, tilt)
                    q, pe, re = ik_link6(model, data, ARMS[G], T6, limits=lim, seeds=2)
                    if q is None or pe > 4e-3 or re > 0.06:  # cheap test before planning the path
                        continue
                    way, _ = cart_path(env, G, giver_goals(env, G, T6), tol=(0.01, 0.35))
                    if way is None:
                        continue
                    teleport_arm(env, G, way[-1], opening=opening)
                    data.qpos[a:a + 3], data.qpos[a + 3:a + 7] = Tobj[:3, 3], quat_of(Tobj[:3, :3])
                    mujoco.mj_forward(model, data)
                    if arm_touches(env, view, G, ignore_item=True):
                        continue
                    grasps = receiver_options(env, view, V, limit=2)
                    if grasps:
                        plans.append(dict(point=point.round(4).tolist(), yaw=float(yaw), tilt=float(tilt),
                                          T6_giver=T6, way_giver=way, grasps=grasps))
                    if len(plans) >= max_plans:
                        return plans
    finally:
        restore(env, snap)
    return plans


def _finger(model, arm: int) -> int:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{ARMS[arm]}_left_finger")
    return int(model.jnt_qposadr[j])


def giver_to(env, G: int, way) -> None:
    """G (closed) carries the mover along its planned joint path (~4 cm per 16 actions)."""
    follow(env, G, way, actions=max(48, 16 * len(way) // 2), grip=0.0)


def receiver_grasp(env, view: MoverView, V: int, grasps) -> tuple[bool, str]:
    """V approaches (joint space to a pre-grasp, then straight on) the first workable of ``grasps`` (mover frame)
    and closes; re-plans on the current pose if none of them works any more (the mover settled differently in G's
    hand)."""
    whys = []
    for replan in (False, True):
        if replan:
            grasps = receiver_options(env, view, V)
        for name, T_obj in grasps:
            got = check_recv(env, view, V, view.object_T(env.data, 0) @ np.asarray(T_obj))
            if got is None:
                whys.append(f"{name}: no longer reachable / collides")
                continue
            TV, qV = got
            qpre = approach_pose(env, V, TV, qV)
            if qpre is None:
                whys.append(f"{name}: pre-grasp unreachable")
                continue
            move_joints(env, V, qpre, actions=96, grip=1.0)
            ok, why = move_link(env, V, TV, actions=48, grip=1.0, n_way=4)
            if not ok:
                move_joints(env, V, qpre, actions=32, grip=1.0)
                whys.append(f"{name}: {why}")
                continue
            run(env, action_with(env, None, {V: 0.0}), 24)
            f = view.contact_obs(env.data)["hits"].reshape(2, 2).sum(1)
            if f[V] == 2:
                return True, name
            whys.append(f"{name}: closed with {int(f[V])} fingers")
            run(env, action_with(env, None, {V: 1.0}), 12)
            move_joints(env, V, qpre, actions=32, grip=1.0)
            if f[1 - V] < 2:
                return False, "; ".join(whys)
    return False, "; ".join(whys[-4:]) or "no receiver grasp"


def run_handoff(env, view: MoverView, G: int, plan: dict) -> tuple[bool, str]:
    """Execute a plan physically: G to the handoff pose, V grasps, G opens (monitored) and backs off."""
    V = 1 - G
    giver_to(env, G, plan["way_giver"])
    run(env, current_action(env), 16)
    f = view.contact_obs(env.data)["hits"].reshape(2, 2).sum(1)
    if f[G] < 2:
        return False, "giver lost the mover on the way"
    ok, why = receiver_grasp(env, view, V, plan["grasps"])
    if not ok:
        return False, why
    if not monitored_release(env, view, G):
        return False, "receiver could not hold it alone"
    retreat(env, G)
    run(env, current_action(env), 16)
    return True, ""


def handoff_solver(env, view: MoverView, G: int, *, deadline: float | None = None, tries: int = 2,
                   avoid=None) -> tuple[bool, str]:
    """Plan + execute, re-planning from the current state after a failed attempt (at most ``tries``)."""
    whys = []
    for _ in range(tries):
        plans = plan_handoff(env, view, G, deadline=deadline, avoid=avoid)
        if not plans:
            whys.append("no handoff plan")
            break
        ok, why = run_handoff(env, view, G, plans[0])
        if ok:
            return True, ""
        whys.append(why)
        f = view.contact_obs(env.data)["hits"].reshape(2, 2).sum(1)
        if f[G] < 2:
            break
    return False, "; ".join(whys)


__all__ = ["MoverView", "PAD", "arm_map", "box_gap", "collision_geoms", "handoff_solver", "pad_centre",
           "plan_handoff", "receiver_candidates", "run_handoff", "sides", "zone_point"]
