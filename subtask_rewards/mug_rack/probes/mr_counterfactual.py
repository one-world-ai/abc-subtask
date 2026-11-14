"""Physical "hanging by the handle" test for settled mug poses (independent of the scorer's contact rule).

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_counterfactual.py <classes5k_seed*.json> --out <cf.json>

For every settled pose (rows of mr_classify output) run two 1.5 s simulations from rest:
  handle_only  the mug's non-handle collision pieces stop colliding; the handle pieces still do.
               Mug stays (CoM moves < 3 cm)  ->  the handle alone holds it = hangs by the handle.
  body_only    the handle pieces stop colliding.  Mug falls (CoM moves > 3 cm) -> the handle was needed.
Handle pieces = collision pieces of the mug whose vertices lie (median) > wall + 3 mm out, in the handle sectors.
A mug "hangs by its handle" when handle_only stays. The scorer verdict is compared against it.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402

print(f"abc_sim from {abc_sim.__file__}", flush=True)


def handle_pieces(env, k=0):
    m, d, ev = env.model, env.data, env._task_evaluator
    info = ev._mug_handles[k]
    root = int(ev._mug_bodies[k][0])
    R, p = d.xmat[root].reshape(3, 3), d.xpos[root]
    out = []
    for g in ev._mug_geoms[k]:
        mid = int(m.geom_dataid[g])
        v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
        loc = (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - p) @ R
        off = loc[:, :2] - info["centre"]
        r = np.linalg.norm(off, axis=1)
        if np.median(r) > info["wall"] + 0.003 and np.isin(ev._sector(off), info["allowed"]).mean() > 0.5:
            out.append(int(g))
    return np.array(out, int)


def run(env, q0, ctrl0, qpos, off_geoms, seconds=1.5):
    m, d, ev = env.model, env.data, env._task_evaluator
    ct, ca = m.geom_contype.copy(), m.geom_conaffinity.copy()
    m.geom_contype[off_geoms] = 0
    m.geom_conaffinity[off_geoms] = 0
    try:
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q0
        adr = ev._mug_qpos_addrs[0]
        d.qpos[adr: adr + 7] = qpos
        mujoco.mj_forward(m, d)
        d.ctrl[:] = ctrl0  # arms hold their reset pose (position actuators)
        com0 = d.subtree_com[int(ev._mug_bodies[0][0])].copy()
        for _ in range(int(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return float(np.linalg.norm(d.subtree_com[int(ev._mug_bodies[0][0])] - com0))
    finally:
        m.geom_contype[:] = ct
        m.geom_conaffinity[:] = ca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
    rows, table = [], collections.Counter()
    for f in a.files:
        src = json.load(open(f))["rows"]
        if not src:
            continue
        env.reset(seed=int(src[0]["seed"]))
        q0 = env.data.qpos.copy()
        ctrl0 = env.data.ctrl.copy()
        hp = handle_pieces(env)
        body = np.setdiff1d(env._task_evaluator._mug_geoms[0], hp)
        print(src[0]["seed"], src[0]["variant"], "handle pieces", len(hp), "of", len(env._task_evaluator._mug_geoms[0]),
              flush=True)
        for row in src:
            base = run(env, q0, ctrl0, np.array(row["qpos"]), np.zeros(0, int))
            ho = run(env, q0, ctrl0, np.array(row["qpos"]), body)
            bo = run(env, q0, ctrl0, np.array(row["qpos"]), hp)
            hangs = ho < 0.03
            rows.append({**{k: row[k] for k in ("seed", "variant", "k", "class", "accepted", "handle_load")},
                         "move_full": round(base, 4), "move_handle_only": round(ho, 4),
                         "move_body_only": round(bo, 4), "hangs_by_handle": bool(hangs)})
            table[(row["variant"], row["class"], "accepted" if row["accepted"] else "rejected",
                   "hangs" if hangs else "not")] += 1
    for key in sorted(table):
        print(key, table[key])
    json.dump({"rows": rows, "table": [[*k, v] for k, v in sorted(table.items())]}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
