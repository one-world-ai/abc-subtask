"""Phase 0: reset probe + scene geometry. For N seeds: stock reward/success at reset (must be 0/False), rack and
plate variants and scales, rack pose, and per plate the collision-mesh size in the plate frame (diameter, total
height, rim lip height above the table, wall thickness at the rim), measured from mesh vertices.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python subtask_rewards/dishrack/probes/dr_reset_geom.py \
        --seeds 0:40 --out ../simulation/outputs/subtask_rewards/dishrack/reset_geom.json
"""
from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

import abc_sim

print("abc_sim from", abc_sim.__file__, flush=True)


def subtree(m, root):
    out = set()
    for b in range(m.nbody):
        c = b
        while c > 0:
            if c == root:
                out.add(b)
                break
            c = int(m.body_parentid[c])
    return out


def verts_in(m, d, bodies, frame_body, collision_only=True):
    """Collision-mesh vertices of geoms on ``bodies`` expressed in ``frame_body``'s frame."""
    pf, Rf = d.xpos[frame_body], d.xmat[frame_body].reshape(3, 3)
    pts = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in bodies:
            continue
        if collision_only and (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:  # racks collide through boxes (8 corners)
            v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        else:
            continue
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        pts.append((w - pf) @ Rf)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def plate_dims(m, d, b):
    v = verts_in(m, d, subtree(m, b), b)
    lo, hi = v.min(0), v.max(0)
    # the plate is flat in its own xy plane (checked: smallest extent along z)
    ext = hi - lo
    ax = int(np.argmin(ext))
    other = [i for i in range(3) if i != ax]
    r = np.linalg.norm(v[:, other] - (lo[other] + hi[other]) / 2, axis=1)
    R = float(r.max())
    h = v[:, ax] - lo[ax]
    rim = r > R - 0.01  # outer 1 cm ring
    # wall thickness at the rim: vertical extent of the outer 1 cm ring, and its lowest point above the bottom
    return {"diameter": round(2 * R, 4), "height": round(float(ext[ax]), 4), "flat_axis": ax,
            "rim_ring_z": [round(float(h[rim].min()), 4), round(float(h[rim].max()), 4)],
            "foot_r": round(float(r[h < 0.003].max()), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0:40")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    s0, s1 = map(int, a.seeds.split(":"))
    env = abc_sim.make_env(task="load_plates_into_dish_rack", render_cameras=False)
    rows = []
    for seed in range(s0, s1):
        env.forget_arm_state()
        env.reset(seed=seed, randomize=True)
        m, d = env.model, env.data
        r = env.evaluate_task()
        ev = env._task_evaluator
        meta = env._last_randomization.metadata
        rack = ev._rack_body_id
        rv = verts_in(m, d, subtree(m, rack), rack)
        row = {"seed": seed, "reward": float(np.asarray(r.reward).ravel()[0]),
               "success": bool(np.asarray(r.success).ravel()[0]),
               "rack_variant": meta["dish_rack_variant"], "plate_variants": meta["plate_variants"],
               "scales": {k: round(v, 3) for k, v in env._last_randomization.scale_states.items()},
               "rack_pos": np.round(d.xpos[rack], 3).tolist(),
               "rack_yaw_deg": round(float(np.degrees(np.arctan2(d.xmat[rack][3], d.xmat[rack][0]))), 1),
               "rack_extent": np.round(rv.max(0) - rv.min(0), 3).tolist(),
               "rack_lo": np.round(rv.min(0), 3).tolist(),
               "plates": []}
        for i in ev.plate_indices:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate" if i == 0 else f"plate_{i}")
            pd = plate_dims(m, d, b)
            pd["pos"] = np.round(d.xpos[b], 3).tolist()
            pd["normal_z"] = round(float(d.xmat[b].reshape(3, 3)[2, pd["flat_axis"]]), 3)
            pd["mass"] = round(float(sum(m.body_mass[x] for x in subtree(m, b))), 3)
            row["plates"].append(pd)
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("resets nonzero:", sum(r["reward"] != 0 or r["success"] for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
