"""Mug rack phase 0: reset probe + scene geometry (tree pegs, mug handle holes, spawn layout).

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_scene.py --seeds 40 --out ../simulation/outputs/subtask_rewards/mug_rack/scene.json

Reset probe: every seed must score reward 0 / success False.
Tree: each collision piece above the base -> principal axis, length, thickness, elevation, tip position.
Mugs: every one of the 19 models (request mug_variants=(v,), scales off) -> wall radius, handle hole (SUPERSEDED by
mr_holes.py: this slice at the cup mid-plane misses offset handles such as mug_8) measured by
slicing the collision pieces in the plane through the cup axis and the handle direction (grid point inside a convex
piece = solid; empty cells not connected to the grid border, outside the wall = the hole).
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import label  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

import abc_sim  # noqa: E402

print(f"abc_sim from {abc_sim.__file__}", flush=True)
TASK = "hang_mug_on_mug_rack"


def mesh_world(m, d, g):
    mid = int(m.geom_dataid[g])
    v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], dtype=float)
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tree_pieces(env):
    m, d, ev = env.model, env.data, env._task_evaluator
    axis = ev._axis_xy
    out = []
    for g in ev._tree_hook_geoms:
        v = mesh_world(m, d, int(g))
        c = v.mean(0)
        u, s, vt = np.linalg.svd(v - c)
        ax = vt[0]
        t = (v - c) @ ax
        perp = (v - c) - np.outer(t, ax)
        thick = 2 * float(np.median(np.linalg.norm(perp, axis=1)))
        # orient the axis away from the trunk axis (pegs) / upward (trunk)
        ends = [c + ax * t.min(), c + ax * t.max()]
        r_end = [np.linalg.norm(e[:2] - axis) for e in ends]
        tip = ends[int(np.argmax(r_end))]
        root = ends[int(np.argmin(r_end))]
        direction = tip - root
        length = float(t.max() - t.min())
        is_peg = max(r_end) > 0.03
        out.append({
            "geom": int(g), "kind": "peg" if is_peg else "trunk", "length": round(length, 4),
            "thickness_med": round(thick, 4),
            "thickness_max": round(2 * float(np.linalg.norm(perp, axis=1).max()), 4),
            "tip": np.round(tip, 4).tolist(), "tip_r": round(max(r_end), 4),
            "elev_deg": round(float(np.degrees(np.arctan2(direction[2], np.linalg.norm(direction[:2]) + 1e-9))), 1),
            "azim_world_deg": round(float(np.degrees(np.arctan2(direction[1], direction[0]))), 1),
            "z_range": [round(float(v[:, 2].min()), 4), round(float(v[:, 2].max()), 4)],
        })
    return out


def handle_hole(env, k):
    """Hole of mug k's handle in its own frame, from the collision pieces."""
    m, d, ev = env.model, env.data, env._task_evaluator
    info = ev._mug_handles[k]
    root = int(ev._mug_bodies[k][0])
    R, p = d.xmat[root].reshape(3, 3), d.xpos[root]
    hulls = []
    zmin = np.inf
    for g in ev._mug_geoms[k]:
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        loc = (mesh_world(m, d, int(g)) - p) @ R
        zmin = min(zmin, loc[:, 2].min())
        hulls.append(ConvexHull(loc))
    sect = info["handle_sectors"]
    ang = (-np.pi + (sect + 0.5) * np.pi / 12).mean()
    h = np.array([np.cos(ang), np.sin(ang)])
    c = info["centre"]
    rs = np.arange(0.0, 0.10, 0.001)
    zs = np.arange(zmin - 0.005, zmin + 0.14, 0.001)
    rr, zz = np.meshgrid(rs, zs, indexing="ij")
    pts = np.stack([c[0] + rr * h[0], c[1] + rr * h[1], zz], -1).reshape(-1, 3)
    solid = np.zeros(len(pts), bool)
    for hull in hulls:
        solid |= np.all(pts @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-4, axis=1)
    solid = solid.reshape(rr.shape)
    lab, n = label(~solid)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    hole = np.zeros_like(solid)
    for i in range(1, n + 1):
        if i not in border:
            reg = lab == i
            if rr[reg].min() > info["wall"] - 0.005:
                hole |= reg
    height = float(zz[solid].max() - zz[solid].min()) if solid.any() else 0.0
    outer = float(rr[solid].max())
    res = {"wall_r": round(info["wall"], 4), "handle_outer_r": round(outer, 4), "mug_height": round(height, 4),
           "handle_angle_deg": round(float(np.degrees(ang)), 1), "centre_offset": round(float(np.linalg.norm(c)), 4)}
    if hole.any():
        res.update(hole_r=[round(float(rr[hole].min()), 4), round(float(rr[hole].max()), 4)],
                   hole_z=[round(float(zz[hole].min() - zmin), 4), round(float(zz[hole].max() - zmin), 4)],
                   hole_w=round(float(rr[hole].max() - rr[hole].min()) + 0.001, 4),
                   hole_h=round(float(zz[hole].max() - zz[hole].min()) + 0.001, 4))
        # tangential thickness of the handle around the hole
        hz = 0.5 * (res["hole_z"][0] + res["hole_z"][1]) + zmin
        tvec = np.array([-h[1], h[0], 0.0])
        ts = np.arange(-0.03, 0.03, 0.0005)
        r_out = res["hole_r"][1] + 0.003
        q = np.array([c[0] + r_out * h[0], c[1] + r_out * h[1], hz])[None] + ts[:, None] * tvec
        s = np.zeros(len(q), bool)
        for hull in hulls:
            s |= np.all(q @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-4, axis=1)
        res["handle_thick_t"] = round(float(s.sum() * 0.0005), 4)
    else:
        res["hole"] = None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    resets = []
    for s in range(a.seeds):
        env.reset(seed=s)
        ev = env._task_evaluator
        r = env.evaluate_task()
        m, d = env.model, env.data
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mug_tree")
        mugs = []
        for k, name in enumerate(ev.mug_names):
            adr = ev._mug_qpos_addrs[k]
            q = d.qpos[adr: adr + 7]
            R = d.xmat[int(ev._mug_bodies[k][0])].reshape(3, 3)
            hd = R[:2, 1] * -1  # handle at -y in the mug frame
            mugs.append({"name": name, "pos": np.round(q[:3], 3).tolist(),
                         "dist_tree": round(float(np.linalg.norm(q[:2] - ev._axis_xy)), 3),
                         "handle_dir_deg": round(float(np.degrees(np.arctan2(hd[1], hd[0]))), 1),
                         "variant": env._last_randomization.metadata["mug_variants"][k]})
        resets.append({"seed": s, "reward": float(np.asarray(r.reward).ravel()[0]),
                       "success": bool(np.asarray(r.success).ravel()[0]),
                       "n_mugs": len(ev.mug_names), "tree_pos": np.round(d.xpos[tid], 3).tolist(),
                       "tree_yaw_deg": round(np.degrees(yaw_of(d.xquat[tid])), 1), "mugs": mugs})
    bad = [x["seed"] for x in resets if x["reward"] != 0 or x["success"]]
    print(f"resets: {a.seeds} seeds, nonzero/success: {bad}")
    env.reset(seed=0)
    pieces = tree_pieces(env)
    for p in pieces:
        print("tree", p)
    models = {}
    for v in range(19):
        env.reset(seed=0, options={"randomization": {"mug_variants": [f"mug_{v}"], "mug_count": 1,
                                                    "randomize_scales": False}})
        assert env._last_randomization.metadata["mug_variants"][0] == f"mug_{v}", env._last_randomization.metadata
        models[f"mug_{v}"] = handle_hole(env, 0)
        print(f"mug_{v}", models[f"mug_{v}"], flush=True)
    json.dump({"resets": resets, "reset_bad": bad, "tree": pieces, "mug_models": models}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
