"""Phase 1 feasibility: can a scripted IK policy pick a plate lying flat on the table by pinching its rim?

One plate per scene. The arm nearer the plate pinches the near rim: jaw pinch centre at the rim wall, tips just above the table, closing axis radial
(one finger over the plate floor, one outside the rim), the gripper leaning outward by ``tilt`` degrees from
straight down. Physical motion only after the pre-grasp teleport (arm only, 6 cm above, open): move down
(bottles_starts.move_link, IK waypoints), close for 40 actions, lift 25 cm. Pass = the whole plate >= 1 cm off the
table with both jaws on it after the lift and 30 more idle actions.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python subtask_rewards/dishrack/probes/dr_rim_grasp.py \
        --seeds 0:12 --out ../simulation/outputs/subtask_rewards/dishrack/rim_grasp.json
"""
from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.bottles_starts import idle_action, move_link
from abc_sim.task_eval.dishrack import _geom_points_world
from abc_sim.task_eval.stagekit import ARMS, STATE_SPEC, actuator_limits, arm_ids, ik_link6

print("abc_sim from", abc_sim.__file__, flush=True)
# Jaw geometry (dr_fingers.py): the jaws close along link_6 x; the pads span link_6 y -0.052..-0.033 and
# z 0.121..0.1465, so the pinch centre is PINCH = (0, -0.043, 0.134) in link_6 (not on the link_6 axis), the tips
# end 1.25 cm beyond it and each jaw is 2 cm thick in x; open, the pads are 9.46 cm apart.
PINCH = np.array([0.0, -0.043, 0.134])
TIP_BEYOND = 0.0125
BASE = {0: np.array([0.252, 0.31]), 1: np.array([0.252, -0.31])}


def finger_geoms(m, arm):
    out = {}
    for side in ("left", "right"):
        bid = [b for b in range(m.nbody) if m.body(b).name.startswith(f"{arm}_") and f"{side}_finger" in m.body(b).name]
        gs = set()
        for b in range(m.nbody):
            c = b
            while c > 0:
                if c in bid:
                    gs |= {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b}
                    break
                c = int(m.body_parentid[c])
        out[side] = {g for g in gs if m.geom_contype[g] or m.geom_conaffinity[g]}
    return out


def finger_report(m, d, arm):
    ids = arm_ids(m, arm)
    p6, R6 = d.xpos[ids.link6], d.xmat[ids.link6].reshape(3, 3)
    rep = {}
    for side, gs in finger_geoms(m, arm).items():
        pts = np.concatenate([_geom_points_world(m, d, g) for g in gs]) if gs else np.zeros((1, 3))
        loc = (pts - p6) @ R6
        rep[side] = {"lo": np.round(loc.min(0), 4).tolist(), "hi": np.round(loc.max(0), 4).tolist()}
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0:12")
    p.add_argument("--tilts", default="0,30,50")
    p.add_argument("--clearances", default="0.003,0.008", help="jaw tips above the table (m)")
    p.add_argument("--lift", type=float, default=0.25)
    p.add_argument("--save-states", default=None, help="dir: save the state after descent / close / lift")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    s0, s1 = map(int, a.seeds.split(":"))
    env = abc_sim.make_env(task="load_plates_into_dish_rack", render_cameras=False)
    rows = []
    printed = False
    for seed in range(s0, s1):
        for tilt in map(float, a.tilts.split(",")):
            for depth in map(float, a.clearances.split(",")):
                env.forget_arm_state()
                env.reset(seed=seed, randomize=True,
                          options={"randomization": {"plate_count": 1, "randomize_variants": True}})
                m, d = env.model, env.data
                ev = env._task_evaluator
                meta = env._last_randomization.metadata
                pidx = ev.plate_indices[0]
                pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")
                pg = ev._plate_collision_geom_ids[pidx]
                pts = np.concatenate([_geom_points_world(m, d, g) for g in pg])
                c = (pts.min(0) + pts.max(0)) / 2
                R = float(np.linalg.norm(pts[:, :2] - c[:2], axis=1).max())
                ztop, zbot = float(pts[:, 2].max()), float(pts[:, 2].min())
                arm = 0 if c[1] > 0 else 1
                u = BASE[arm] - c[:2]
                u = np.array([*u / np.linalg.norm(u), 0.0])  # radial, towards the arm
                th = np.radians(tilt)
                approach = -np.array([0, 0, 1.0]) * np.cos(th) - u * np.sin(th)  # wrist leans outward
                xax = u - approach * (u @ approach)
                xax /= np.linalg.norm(xax)
                Rg = np.stack([xax, np.cross(approach, xax), approach], 1)
                site = np.array([*(c[:2] + u[:2] * (R - 0.003)), zbot + depth + TIP_BEYOND * np.cos(th)])
                row = {"seed": seed, "plate": meta["plate_variants"][0], "tilt": tilt, "depth": depth,
                       "radius": round(R, 4), "height": round(ztop - zbot, 4), "arm": arm}
                ok = None
                for closing in ("x",):  # the jaws close along link_6 x
                    if closing == "y":
                        Rg2 = np.stack([np.cross(xax, approach), xax, approach], 1)
                    else:
                        Rg2 = Rg
                    env.forget_arm_state()
                    env.reset(seed=seed, randomize=True,
                              options={"randomization": {"plate_count": 1, "randomize_variants": True}})
                    m, d = env.model, env.data
                    ids = arm_ids(m, ARMS[arm])
                    T = np.eye(4)
                    T[:3, :3] = Rg2
                    T[:3, 3] = site - Rg2 @ PINCH - approach * 0.06
                    q, pe, re = ik_link6(m, d, ARMS[arm], T, limits=actuator_limits(m, ARMS[arm]))
                    if q is None or pe > 3e-3 or re > 0.05:
                        row[closing] = f"pregrasp IK {pe:.3f}"
                        continue
                    d.qpos[ids.qadr] = q
                    d.qvel[ids.dadr] = 0
                    mujoco.mj_forward(m, d)
                    act = idle_action(env)
                    act[arm * 7:arm * 7 + 6] = q
                    act[arm * 7 + 6] = 1.0
                    for _ in range(10):
                        env.step(act, render_obs=False)
                    Tg = T.copy()
                    Tg[:3, 3] = site - Rg2 @ PINCH
                    good, why = move_link(env, arm, Tg, actions=64, grip=1.0)
                    if not good:
                        row[closing] = why
                        continue
                    def save(tag):
                        if a.save_states:
                            st = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
                            mujoco.mj_getState(m, d, st, STATE_SPEC)
                            np.save(f"{a.save_states}/seed{seed}_tilt{tilt:.0f}_cl{depth}_{tag}.npy", st)
                    act = idle_action(env)  # let the position controllers arrive before judging/closing
                    act[arm * 7 + 6] = 1.0
                    for _ in range(30):
                        env.step(act, render_obs=False)
                    save("descent")
                    com0 = float(d.subtree_com[pb][2])
                    ids6 = arm_ids(m, ARMS[arm]).link6
                    pinch_err = (d.xpos[ids6] + d.xmat[ids6].reshape(3, 3) @ PINCH) - site
                    row["pinch_err_mm"] = np.round(1000 * pinch_err, 1).tolist()
                    act = idle_action(env)
                    act[arm * 7 + 6] = 0.0
                    for _ in range(40):
                        env.step(act, render_obs=False)
                    save("closed")
                    Tl = Tg.copy()
                    Tl[:3, 3] += [0, 0, a.lift]
                    good, why = move_link(env, arm, Tl, actions=96, grip=0.0)
                    if not good:
                        row[closing] = "lift " + why
                        continue
                    act = idle_action(env)
                    act[arm * 7 + 6] = 0.0
                    for _ in range(30):
                        env.step(act, render_obs=False)
                    save("lifted")
                    pts = np.concatenate([_geom_points_world(m, d, g) for g in ev._plate_collision_geom_ids[pidx]])
                    fg = finger_geoms(m, ARMS[arm])
                    pgs = set(ev._plate_collision_geom_ids[pidx])
                    touch = {s: any((int(cc.geom1) in pgs and int(cc.geom2) in gs) or (int(cc.geom2) in pgs and int(cc.geom1) in gs)
                                    for cc in d.contact[: d.ncon]) for s, gs in fg.items()}
                    lift = float(pts[:, 2].min() - zbot)
                    nz = float(abs(d.xmat[pb].reshape(3, 3)[2, 2]))
                    res = {"lowest_up": round(lift, 3), "com_up": round(float(d.subtree_com[pb][2]) - com0, 3),
                           "fingers": touch, "normal_z": round(nz, 3),
                           "ok": bool(lift >= 0.01 and all(touch.values()))}
                    row[closing] = res
                    ok = ok or res["ok"]
                row["ok"] = bool(ok)
                rows.append(row)
                print(json.dumps(row), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    by = {}
    for r in rows:
        by.setdefault(r["tilt"], []).append(r["ok"])
    print("success by tilt:", {k: f"{sum(v)}/{len(v)}" for k, v in by.items()},
          "seeds with any success:", len({r['seed'] for r in rows if r['ok']}), "/", len({r['seed'] for r in rows}))


if __name__ == "__main__":
    main()
