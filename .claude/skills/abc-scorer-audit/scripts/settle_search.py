"""Drop one free object from random poses near a goal fixture and record where it comes to rest and how it scores.

Finds scorer false positives nobody thought of: e.g. for the mug rack, 57 of 63 mugs that settled touching only the
tree had a peg inside the cup, and the contacts-only scorer accepted them.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        .claude/skills/abc-scorer-audit/scripts/settle_search.py --task hang_mug_on_mug_rack \
        --object-joint mug_1_jnt --fixture-body mug_tree --seed 0 --tries 3000 --out <file.json>

Sample centres: the fixture's collision-mesh vertices farthest from its vertical axis (peg tips, rims), or
--centre x,y,z. A sample is kept if it starts without interpenetration (> 1 mm), comes to rest (speed < 1 cm/s
after --settle-s + 2 s), and touches the fixture subtree and nothing else except objects of the same kind.
Each kept pose stores the object's qpos and the evaluator's reward/success/metrics. Classify and render them next.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402

# PYTHONPATH must point at the abc checkout under test; `python -I` would silently import the venv's installed copy.
print(f"abc_sim from {abc_sim.__file__}", flush=True)


def subtree(model, root: int) -> set[int]:
    out = {root}
    for body in range(model.nbody):
        b = body
        while b > 0:
            if b == root:
                out.add(body)
                break
            b = int(model.body_parentid[b])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", required=True)
    p.add_argument("--object-joint", required=True, help="free joint of the object to drop")
    p.add_argument("--fixture-body", required=True, help="body whose subtree is the goal fixture")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tries", type=int, default=2000)
    p.add_argument("--sigma", type=float, default=0.03, help="position noise around a sample centre (m)")
    p.add_argument("--centre", default=None, help="x,y,z to sample around instead of fixture extremities")
    p.add_argument("--settle-s", type=float, default=1.5)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env = abc_sim.make_env(task=args.task, render_cameras=False)
    env.reset(seed=args.seed)
    m, d = env.model, env.data
    q0 = d.qpos.copy()
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, args.object_joint)
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.fixture_body)
    if jid < 0 or fid < 0:
        raise SystemExit("unknown joint or fixture body name")
    adr = int(m.jnt_qposadr[jid])
    obj_bodies = subtree(m, int(m.jnt_bodyid[jid]))
    fix_bodies = subtree(m, fid)
    obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in obj_bodies}
    fix_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in fix_bodies}

    if args.centre:
        centres = [np.array([float(v) for v in args.centre.split(",")])]
    else:
        mujoco.mj_forward(m, d)
        pts = []
        for g in fix_geoms:
            mid = int(m.geom_dataid[g])
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH) or mid < 0:
                continue
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        pts = np.concatenate(pts)
        axis = d.xpos[fid][:2]
        r = np.linalg.norm(pts[:, :2] - axis, axis=1)
        centres = list(pts[r > np.quantile(r, 0.9)][:: max(1, int((r > np.quantile(r, 0.9)).sum() // 40))])
        centres.append(pts[pts[:, 2].argmax()])
    rng = np.random.default_rng(args.seed)

    def contacts_other() -> tuple[bool, bool]:
        on_fix, other = False, False
        for c in d.contact[: d.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            if (a in obj_geoms) == (b in obj_geoms):
                continue
            o = b if a in obj_geoms else a
            if o in fix_geoms:
                on_fix = True
            else:
                other = True
        return on_fix, other

    kept, tried = [], 0
    for k in range(args.tries):
        centre = centres[rng.integers(len(centres))]
        pos = centre + rng.normal(0, args.sigma, 3)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        d.qpos[:] = q0
        d.qvel[:] = 0
        d.qpos[adr: adr + 3] = pos
        d.qpos[adr + 3: adr + 7] = quat
        mujoco.mj_forward(m, d)
        if any(((int(c.geom1) in obj_geoms) or (int(c.geom2) in obj_geoms)) and c.dist < -0.001
               for c in d.contact[: d.ncon]):
            continue
        tried += 1
        for _ in range(int((args.settle_s + 2.0) / m.opt.timestep)):
            mujoco.mj_step(m, d)
        on_fix, other = contacts_other()
        speed = float(np.linalg.norm(d.qvel[int(m.jnt_dofadr[jid]): int(m.jnt_dofadr[jid]) + 3]))
        if not on_fix or other or speed > 0.01:
            continue
        r = env.evaluate_task()
        kept.append({"k": k, "qpos": d.qpos[adr: adr + 7].tolist(),
                     "reward": float(np.asarray(r.reward).ravel()[0]),
                     "success": bool(np.asarray(r.success).ravel()[0]),
                     "metrics": {key: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for key, v in r.metrics.items() if not callable(v)}})
    with open(args.out, "w") as f:
        json.dump({"task": args.task, "seed": args.seed, "object_joint": args.object_joint,
                   "fixture_body": args.fixture_body, "tried": tried, "kept": kept}, f, default=str)
    accepted = sum(1 for h in kept if h["reward"] > 0)
    print(f"tried {tried} (no initial overlap), resting on the fixture only {len(kept)}, scorer reward > 0: {accepted}")


if __name__ == "__main__":
    main()
