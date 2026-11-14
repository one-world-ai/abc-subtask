"""Dish-rack geometry for the subtasks: plate / rack collision point clouds, the kinematic slot map and top-edge grasps.

Everything is measured from the collision geometry of the current scene (21 plate models, 12 rack models, random
scales), never from a per-model table.

Jaw geometry (``subtask_rewards/dishrack/probes/dr_fingers.py``): the jaws close along link_6 x, open 9.46 cm, are
2 cm thick, and their pads span link_6 y -0.052..-0.033 and z 0.121..0.1465. The pinch centre is therefore
``PINCH = (0, -0.043, 0.134)`` in link_6, 4.3 cm off the link_6 axis that ``stagekit``'s grasp-site constant assumes.
That offset is handled here, for this task only; ``stagekit`` is unchanged.
"""
from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.dishrack import _geom_points_world

PINCH = np.array([0.0, -0.043, 0.134])
JAW_OPEN_M = 0.0946
GRIP_HALF_MAX = 0.0475   # finger joint travel = half the jaw opening
PAD_LEN_M = 0.025
TABLE_Z = 0.75


def local_points(model, data, geoms, body) -> np.ndarray:
    """Collision points of ``geoms`` in ``body``'s frame (constant for a rigid subtree)."""
    pts = np.concatenate([_geom_points_world(model, data, g) for g in geoms])
    return (pts - data.xpos[body]) @ data.xmat[body].reshape(3, 3)


def frame(data, body):
    return np.asarray(data.xpos[body], np.float64).copy(), np.asarray(data.xmat[body], np.float64).reshape(3, 3).copy()


def quat_of_R(R) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(9))
    return q


def upright_rotation(rack_R: np.ndarray, axis: int, yaw: float = 0.0) -> np.ndarray:
    """Plate rotation with its flat normal (local z) along rack axis ``axis`` (+ ``yaw`` about world z) and local x up."""
    ez = rack_R[:, axis].copy()
    ez[2] = 0.0
    ez /= np.linalg.norm(ez)
    c, s = np.cos(yaw), np.sin(yaw)
    ez = np.array([c * ez[0] - s * ez[1], s * ez[0] + c * ez[1], 0.0])
    ex = np.array([0.0, 0.0, 1.0])
    return np.stack([ex, np.cross(ez, ex), ez], 1)


def compute_slot_map(model, data, rack_body, rack_geoms, plate_body, plate_qadr, plate_geoms, *, step=0.005,
                     sink_m=0.03, fracs=(-0.25, 0.0, 0.25)) -> dict:
    """Where an upright plate (normal along rack x or y) can be lowered into the rack, measured kinematically.

    On 3 lines across the rack footprint per axis (centre line, +-25 % of the width), at 5 mm steps t along the
    axis, the upright plate (local x up, bbox centre on the line) is lowered in 5 mm steps until it would penetrate a
    rack collision geom by more than 1 mm (the ``dr_slots.py`` method, without the release). ``entry`` = the top of
    the rack parts within 2 cm (along the axis) of the plate's plane and under the plate; a position is a slot
    position if the plate sinks >= ``sink_m`` below that entry. Slots = runs of slot positions: centre ``t``,
    lateral line ``u``, deepest seat height ``seat`` of the lowest point (and at the run centre, ``seat_mid``) and the
    run's highest ``entry``. Heights are rack-frame z (rack base = 0)."""
    rp, Rr = frame(data, rack_body)
    rack_pts = local_points(model, data, rack_geoms, rack_body)
    lo, hi = rack_pts[:, :2].min(0), rack_pts[:, :2].max(0)
    plate_pts = local_points(model, data, plate_geoms, plate_body)
    pc = (plate_pts.min(0) + plate_pts.max(0)) / 2
    half_up = float(plate_pts[:, 0].max() - pc[0])        # upright: local x is world up
    half_low = float(pc[0] - plate_pts[:, 0].min())
    half_side = float(np.abs(plate_pts[:, 1] - pc[1]).max())
    d2 = mujoco.MjData(model)
    d2.qpos[:] = data.qpos
    d2.qvel[:] = 0
    pset, rset = set(int(g) for g in plate_geoms), set(int(g) for g in rack_geoms)
    slots, lines = [], {}
    for axis in (0, 1):
        other = 1 - axis
        Rp = upright_rotation(Rr, axis)
        # rack frame rotation of the plate: used to place the bbox centre
        qp = quat_of_R(Rp)
        for frac in fracs:
            u = (lo[other] + hi[other]) / 2 + frac * (hi[other] - lo[other])
            ts = np.arange(lo[axis] + 0.01, hi[axis] - 0.01 + 1e-9, step)
            zlow = np.full(len(ts), np.nan)
            entry = np.full(len(ts), np.nan)
            for i, t in enumerate(ts):
                col = rack_pts[(np.abs(rack_pts[:, axis] - t) <= 0.02) & (np.abs(rack_pts[:, other] - u) <= half_side)]
                if not len(col):
                    continue
                entry[i] = col[:, 2].max()
                best = None
                c = np.zeros(3)
                c[axis], c[other] = t, u
                for z in np.arange(entry[i] + 0.01, 0.002, -step):
                    c[2] = z + half_low
                    d2.qpos[plate_qadr:plate_qadr + 3] = rp + Rr @ c - Rp @ pc
                    d2.qpos[plate_qadr + 3:plate_qadr + 7] = qp
                    mujoco.mj_kinematics(model, d2)
                    mujoco.mj_collision(model, d2)
                    hit = False
                    for con in d2.contact[:d2.ncon]:
                        g1, g2 = int(con.geom1), int(con.geom2)
                        if con.dist < -0.001 and ((g1 in pset and g2 in rset) or (g2 in pset and g1 in rset)):
                            hit = True
                            break
                    if hit:
                        break
                    best = z
                if best is not None:
                    zlow[i] = best
            lines[(axis, round(float(frac), 2))] = dict(u=float(u), t=ts, zlow=zlow, entry=entry)
            ok = np.isfinite(zlow) & (zlow <= entry - sink_m)
            i = 0
            while i < len(ts):
                if not ok[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(ts) and ok[j + 1]:
                    j += 1
                mid = (i + j) // 2
                slots.append(dict(axis=axis, line=(axis, round(float(frac), 2)), u=float(u),
                                  t=float((ts[i] + ts[j]) / 2), t_lo=float(ts[i]),
                                  t_hi=float(ts[j]), seat=float(np.nanmin(zlow[i:j + 1])),
                                  seat_mid=float(zlow[mid]), entry=float(np.nanmax(entry[i:j + 1]))))
                i = j + 1
    return dict(slots=slots, lines=lines, footprint=(lo, hi), rack_top=float(rack_pts[:, 2].max()),
                plate_centre=pc, half_up=half_up, half_low=half_low, half_side=half_side)


def edge_grasp(plate_pts: np.ndarray, theta: float, *, beta: float = 0.0, below_rim_m=0.015, side=1
               ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pinch at the plate's rim in direction ``theta`` (rad, in the plate plane, 0 = local +x), the pinch centre
    ``below_rim_m`` inside the rim, the jaws closing across the plate (link_6 x along the plate normal times
    ``side``) and the approach tilted by ``beta`` (rad) from radially inward, within the plate plane.

    With the plate held local x up and theta = 0 the pinch is at the plate's top, where the plate hangs in
    equilibrium (no gravity torque about the jaw axis); ``beta`` then tilts the wrist (IK needs 30-75 deg,
    ``pilot_reach2.py``). Returns (T_plate_link6, pinch point in the plate frame, info)."""
    pc = (plate_pts.min(0) + plate_pts.max(0)) / 2
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    lat = np.array([-np.sin(theta), np.cos(theta), 0.0])
    rel = plate_pts - pc
    along, across = rel @ d, rel @ lat
    band = np.abs(across) <= 0.01
    r = float(along[band].max())
    pinch_xy = d * (r - below_rim_m)
    near = np.linalg.norm(rel[:, :2] - pinch_xy[:2], axis=1) <= PAD_LEN_M / 2 + 0.002
    if not near.any():
        near = band & (along >= r - PAD_LEN_M)
    zlo, zhi = float(rel[near, 2].min()), float(rel[near, 2].max())
    pinch = pc + pinch_xy + np.array([0, 0, (zlo + zhi) / 2])
    x6 = np.array([0, 0, float(side)])
    z6 = -(np.cos(beta) * d + np.sin(beta) * lat)
    y6 = np.cross(z6, x6)
    R6 = np.stack([x6, y6, z6], 1)
    T6 = np.eye(4)
    T6[:3, :3] = R6
    T6[:3, 3] = pinch - R6 @ PINCH
    return T6, pinch, dict(rim_r=r, thickness=zhi - zlo, fits=bool(zhi - zlo <= JAW_OPEN_M - 0.012))
