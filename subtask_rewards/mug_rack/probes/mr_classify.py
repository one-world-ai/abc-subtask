"""Classify settle-search rest poses of mug_1 on the tree geometrically, independent of the scorer's handle rule.

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_classify.py <settle_seed*.json> ... --out <classes.json>

Per pose (mug_1 at its settled qpos, mj_forward):
  threaded   a tree piece passes through the handle hole: a cell of the measured hole (slice through the cup axis
             and the handle direction, see mr_scene.handle_hole) lies inside a tree piece's convex hull
  in_cup     a tree touch point is inside the cup (radius < wall - 3 mm, above the floor)
  handle     a touch point on the handle (outside wall + 3 mm in the handle sectors), any height
  wall/rim   other touch points (outer wall, rim, bottom)
Class = threaded > in_cup > handle_outside (handle touch, not threaded) > body (wall/rim/bottom only).
Also records the scorer's verdict (reward > 0) and tilt of the cup axis.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, label  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

import abc_sim  # noqa: E402

print(f"abc_sim from {abc_sim.__file__}", flush=True)


def hulls_local(m, d, geoms, root):
    R, p = d.xmat[root].reshape(3, 3), d.xpos[root]
    out = []
    for g in geoms:
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(m.geom_dataid[g])
        v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        out.append((w - p) @ R)
    return out


def hole_cells(env, k=0):
    """Hole cells (mug frame) for mug k, plus the floor height; empty array if the collision handle is closed."""
    m, d, ev = env.model, env.data, env._task_evaluator
    info = ev._mug_handles[k]
    root = int(ev._mug_bodies[k][0])
    locs = hulls_local(m, d, ev._mug_geoms[k], root)
    hulls = [ConvexHull(v) for v in locs]
    zmin = min(v[:, 2].min() for v in locs)
    ang = (-np.pi + (info["handle_sectors"] + 0.5) * np.pi / 12).mean()
    h = np.array([np.cos(ang), np.sin(ang)])
    c = info["centre"]
    tdir = np.array([-h[1], h[0]])
    # The handle is not always centred on the cup axis tangentially (mug_8: ~5 mm off): slice through its middle.
    allv0 = np.concatenate(locs)
    off0 = allv0[:, :2] - c
    on_h = (np.linalg.norm(off0, axis=1) > info["wall"] + 0.003) & np.isin(ev._sector(off0), info["allowed"])
    t0 = float(np.median(off0[on_h] @ tdir)) if on_h.any() else 0.0
    rs = np.arange(0.0, 0.10, 0.001)
    zs = np.arange(zmin - 0.005, zmin + 0.14, 0.001)
    cells = []
    inscribed = 0.0
    for dt in (t0 - 0.003, t0, t0 + 0.003):  # three parallel slices across the handle's thickness
        tvec = tdir * dt
        rr, zz = np.meshgrid(rs, zs, indexing="ij")
        pts = np.stack([c[0] + tvec[0] + rr * h[0], c[1] + tvec[1] + rr * h[1], zz], -1).reshape(-1, 3)
        solid = np.zeros(len(pts), bool)
        for hull in hulls:
            solid |= np.all(pts @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-4, axis=1)
        solid = solid.reshape(rr.shape)
        # close 2-mm gaps between convex pieces (handle-wall joints), else the hole leaks to the border
        closed = binary_closing(np.pad(solid, 3), iterations=2)[3:-3, 3:-3] | solid
        lab, n = label(~closed)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        for i in range(1, n + 1):
            reg = lab == i
            if i not in border and rr[reg].min() > 0.01 and reg.sum() >= 20:  # enclosed, off the axis (tapered walls: no wall-radius test)
                reg = binary_dilation(reg, iterations=2) & ~solid  # undo the closing's shrink
                cells.append(pts.reshape(*rr.shape, 3)[reg])
                inscribed = max(inscribed, 2 * 0.001 * float(distance_transform_edt(reg).max()))
    allv = np.concatenate(locs)
    off = allv[:, :2] - c
    body = allv[~np.isin(ev._sector(off), info["allowed"]) | (np.linalg.norm(off, axis=1) <= info["wall"])]
    hole_cells.last = {"t0": t0, "inscribed_d": inscribed}
    return (np.concatenate(cells) if cells else np.zeros((0, 3))), (zmin, ConvexHull(body))


def classify(env, qpos, cells, aux):
    zmin, cup_hull = aux
    m, d, ev = env.model, env.data, env._task_evaluator
    adr = ev._mug_qpos_addrs[0]
    d.qpos[adr: adr + 7] = qpos
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    root = int(ev._mug_bodies[0][0])
    R, p = d.xmat[root].reshape(3, 3), d.xpos[root]
    info = ev._mug_handles[0]
    threaded = False
    if len(cells):
        world = cells @ R.T + p
        for g in ev._tree_hook_geoms:
            mid = int(m.geom_dataid[g])
            v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
            hull = ConvexHull(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
            if np.any(np.all(world @ hull.equations[:, :3].T + hull.equations[:, 3] <= 0, axis=1)):
                threaded = True
                break
    pts = ev._touch_points(d, ev._mug_geoms[0], ev._tree_hook_geoms)
    kinds = set()
    for w in pts:
        loc = (w - p) @ R
        off = loc[:2] - info["centre"]
        r = float(np.linalg.norm(off))
        depth = -float(np.max(cup_hull.equations[:, :3] @ loc + cup_hull.equations[:, 3]))
        if depth > 0.004 and loc[2] > zmin + 0.005:  # >4 mm inside the cup's outer hull: in the cavity
            kinds.add("in_cup")
        elif r > info["wall"] + 0.003 and int(ev._sector(off[None])[0]) in set(info["allowed"].tolist()):
            kinds.add("handle")
        else:
            kinds.add("body")
    cls = ("threaded" if threaded else "in_cup" if "in_cup" in kinds else
           "handle_outside" if "handle" in kinds else "body")
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    share = handle_load_share(m, d, ev, R, p, info)
    r = env.evaluate_task()
    return cls, sorted(kinds), tilt, float(np.asarray(r.reward).ravel()[0]), share


def handle_load_share(m, d, ev, R, p, info):
    """Fraction of the upward contact force on the mug that acts through handle contacts (mj_forward at rest)."""
    geoms = set(ev._mug_geoms[0].tolist())
    f6 = np.zeros(6)
    up_handle = up_total = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if (a in geoms) == (b in geoms):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]  # force on geom2, world frame
        fz = fw[2] if b in geoms else -fw[2]
        loc = (c.pos - p) @ R
        off = loc[:2] - info["centre"]
        on_handle = (np.linalg.norm(off) > info["wall"] + 0.003) and int(ev._sector(off[None])[0]) in set(
            info["allowed"].tolist())
        up_total += max(fz, 0.0)
        if on_handle:
            up_handle += max(fz, 0.0)
    return round(up_handle / up_total, 3) if up_total > 1e-6 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
    rows, table = [], collections.Counter()
    for f in a.files:
        s = json.load(open(f))
        env.reset(seed=int(s["seed"]))
        variant = env._last_randomization.metadata["mug_variants"][0]
        q0 = env.data.qpos.copy()
        cells, aux = hole_cells(env)
        print(s["seed"], variant, "hole cells", len(cells), hole_cells.last, flush=True)
        for h in s["kept"]:
            env.data.qpos[:] = q0
            cls, kinds, tilt, reward, share = classify(env, np.asarray(h["qpos"]), cells, aux)
            assert abs(reward - h["reward"]) < 1e-6, (f, h["k"], reward, h["reward"])
            rows.append({"seed": s["seed"], "variant": variant, "has_hole": bool(len(cells)), "k": h["k"],
                         "class": cls, "touch": kinds, "handle_load": share, "tilt": round(tilt, 1), "accepted": reward > 0,
                         "qpos": h["qpos"]})
            table[(variant, cls, reward > 0)] += 1
    for key in sorted(table):
        print(key, table[key])
    json.dump({"rows": rows, "table": [[*k, v] for k, v in sorted(table.items())]}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
