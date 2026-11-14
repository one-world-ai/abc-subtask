"""Feasibility: can a scripted IK motion thread a peg through a held mug's handle and leave it hanging?

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_insert.py --cf ../simulation/outputs/subtask_rewards/mug_rack/settle/cf_seed1.json \
        --classes ../simulation/outputs/subtask_rewards/mug_rack/settle/classes5k_seed1.json --poses 3 --grasps 40 --out <json>

Privileged, best case: the goal pose H is a settled real hang of mug_1 (threaded, handle alone holds it). The peg it
hangs on is found from the scorer's hole points; u = that peg's axis, pointing out of the trunk.
  1. mug teleported to P = H + 1 cm up + d_out * u (in line with the peg, handle hole around the peg axis);
  2. a grasp from the mug-flip grasp library (same mug assets) is placed with the kit's hold_object (IK teleport,
     weightless close, gravity hold, must touch nothing else);
  3. physical Cartesian move of link_6 (8 IK waypoints per segment, mug held, no teleports) P -> H + 1 cm up;
  4. open the gripper, back link_6 off 6 cm along -approach, idle 1 s; score with the stock scorer.
Reports per grasp: IK/hold failures, slip during insertion, collision events, final scorer verdict.
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402
from abc_sim.task_eval.bottles_starts import idle_action, move_link  # noqa: E402
from abc_sim.task_eval.mug_flip_starts import load_grasp_library  # noqa: E402
from abc_sim.task_eval.stagekit import (ARMS, T, ObjectScene, arm_ids, free_joints_matching, hold_object,  # noqa: E402
                                       set_object_pose)

print(f"abc_sim from {abc_sim.__file__}", flush=True)


def threading_peg(env, k=0):
    """(peg geom, outward unit axis, tip) of the tree piece through mug k's handle hole."""
    m, d, ev = env.model, env.data, env._task_evaluator
    root = int(ev._mug_bodies[k][0])
    hole = ev._hole(k) @ d.xmat[root].reshape(3, 3).T + d.xpos[root]
    for g, planes in zip(ev._tree_hook_geoms, ev._tree_hook_planes):
        if planes is not None and ev._inside(hole, planes).any():
            mid = int(m.geom_dataid[g])
            v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            c = w.mean(0)
            ax = np.linalg.svd(w - c)[2][0]
            r = np.linalg.norm((w - c) @ ax)
            if np.linalg.norm((c + ax * 0.02)[:2] - ev._axis_xy) < np.linalg.norm((c - ax * 0.02)[:2] - ev._axis_xy):
                ax = -ax
            t = (w - c) @ ax
            return int(g), ax, c + ax * t.max()
    return None, None, None


def touching(env, scene, k):
    m, d = env.model, env.data
    out = set()
    for c in d.contact[: d.ncon]:
        o1, o2 = scene.owner[int(c.geom1)], scene.owner[int(c.geom2)]
        for a, b, gb in ((o1, o2, int(c.geom2)), (o2, o1, int(c.geom1))):
            if a[0] == 0 and a[1] == k and not (b[0] == 0 and b[1] == k):
                out.add("finger" if b[0] == 1 else m.body(int(m.geom_bodyid[gb])).name)
    return out


def attempt(env, seed, qpos_h, grasp, d_out):
    env.forget_arm_state()
    env.reset(seed=seed)
    m, d, ev = env.model, env.data, env._task_evaluator
    scene = ObjectScene(m, free_joints_matching(m, r"mug_(\d+)_jnt"))
    adr = ev._mug_qpos_addrs[0]
    d.qpos[adr:adr + 7] = qpos_h
    mujoco.mj_forward(m, d)
    g, u, tip = threading_peg(env)
    if g is None:
        return {"why": "goal pose not threaded"}
    H = np.array(qpos_h[:3]) + [0, 0, 0.01]
    P = H + d_out * u
    set_object_pose(env, scene, 0, P, qpos_h[3:])
    if touching(env, scene, 0):
        return {"why": f"pre-insert pose collides: {sorted(touching(env, scene, 0))}"}
    arm = int(grasp["arm"])
    ok, info = hold_object(env, scene, 0, grasp)
    if not ok:
        return {"why": "hold: " + info.get("why", "?")}
    ids = arm_ids(m, ARMS[arm])
    T6 = T(d.xpos[ids.link6], d.xquat[ids.link6])
    obj = scene.object_T(d, 0)
    rel = np.linalg.inv(obj) @ T6
    events = []
    for frac in (0.5, 1.0):
        goal = obj.copy()
        goal[:3, 3] = P + frac * (H - P) + (obj[:3, 3] - P)
        ok, why = move_link(env, arm, goal @ rel, actions=96)
        if not ok:
            return {"why": f"insert IK: {why}", "arm": arm}
        tch = touching(env, scene, 0) - {"finger"}
        if tch:
            events.append(sorted(tch))
        if not scene.contact_obs(d)["grasped"][0]:
            return {"why": "slipped during insertion", "events": events, "arm": arm}
    err = float(np.linalg.norm(d.qpos[adr:adr + 3] - H))
    act = idle_action(env)
    act[arm * 7 + 6] = 1.0
    for _ in range(40):
        env.step(act, render_obs=False)
    back = T(d.xpos[ids.link6], d.xquat[ids.link6])
    back[:3, 3] -= 0.06 * back[:3, 2]
    move_link(env, arm, back, actions=48, grip=1.0)
    act = idle_action(env)
    for _ in range(100):
        env.step(act, render_obs=False)
    r = env.evaluate_task()
    met = r.metrics
    return {"arm": arm, "events": events, "pos_err_at_goal": round(err, 4),
            "hanging": bool(np.asarray(r.reward).ravel()[0] > 0), "threaded": met["mug_threaded_mask"][0][0],
            "touch_after": sorted(touching(env, scene, 0)),
            "mug_drop_m": round(float(H[2] - d.qpos[adr + 2]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--poses", type=int, default=3)
    ap.add_argument("--grasps", type=int, default=40)
    ap.add_argument("--d-out", type=float, default=0.07)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cf = {r["k"]: r for r in json.load(open(a.cf))["rows"]}
    rows = [r for r in json.load(open(a.classes))["rows"]
            if r["class"] == "threaded" and r["accepted"] and cf[r["k"]]["move_handle_only"] < 0.005]
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))[: a.poses]]
    lib = load_grasp_library()
    env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
    results = []
    for row in rows:
        variant = row["variant"]
        cands = [g for g in lib if g["variant"] == variant] or lib
        pick = rng.permutation(len(cands))[: a.grasps]
        for gi in pick:
            res = attempt(env, int(row["seed"]), np.array(row["qpos"]), cands[gi], a.d_out)
            res.update(seed=row["seed"], variant=variant, k=row["k"], grasp=int(gi))
            results.append(res)
            print(json.dumps(res), flush=True)
    summ = {"attempts": len(results), "held": sum("why" not in r or not r["why"].startswith(("hold", "pre")) for r in results),
            "inserted": sum("hanging" in r for r in results), "hanging": sum(r.get("hanging", False) for r in results)}
    print(summ)
    json.dump({"summary": summ, "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
