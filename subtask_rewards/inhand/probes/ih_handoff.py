"""Phase 0/1 feasibility: can the two arms hand the object over in the shared middle region?

Per seed: reset, settle; move the object (weightless while placing) to a middle pose; arm A (spawn side) takes it
with a synthesized grasp (stagekit.hold_object: weightless close, gravity hold, checks); arm B is teleported open
onto a second grasp (must touch nothing), both hold (15 actions), A opens (15), A lifts away 10 cm and backs off (30),
B alone holds under gravity (30). Pass = B has both fingers on it, A has none and its grasp site is > 6 cm away,
object drifted < 3 cm since A opened and moves < 5 cm/s, and it touches only B's fingers.

Grasp modes, tried in order until one passes (each with the gripper yaw flips):
  ends_topdown   object lying with its long axis along y; A top-down near its end, B top-down near the other end
  top_side       A top-down on the upper part; B side-on (link_6 horizontal, approaching from B's side) lower part
  ends_side      lying object; A top-down near its end; B side-on at the other end (approaching along y)
  side_side      A side-on from A's side on the upper part, B side-on from B's side on the lower part
(env IH_MODES=comma list overrides the modes, IH_SEEDS=comma list the seeds)
usage: ih_handoff.py first last out.json [z_mid]"""
import json, sys, time
import numpy as np
import mujoco
from abc_sim.task_eval.stagekit import ARMS, GRIP_MAX, arm_ids, hold_object, inv, set_object_pose, ik_link6, qmul, quat_of
from ih_common import make, reset, idle, obj_pos
from ih_grasp import principal, section, topdown_T6, side_T6, two_arm_action, arm_contacts, teleport_arm, Rz, reachable

first, last, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
Z_MID = float(sys.argv[4]) if len(sys.argv) > 4 else 0.90
YAWS = np.deg2rad([90, 60, 120, 30, 150, 0])
X_MID = 0.55
SEED_BUDGET_S = 300
import os
MODES = os.environ.get("IH_MODES", "ends_topdown,top_side,ends_side,side_side").split(",")
SEEDS = [int(x) for x in os.environ["IH_SEEDS"].split(",")] if os.environ.get("IH_SEEDS") else None
MIDS = [(0.55, Z_MID), (0.50, Z_MID - 0.05), (0.60, Z_MID - 0.05), (0.50, Z_MID + 0.05)]
env = make()


def setup(seed, yaw_axis=np.pi / 2, mid=None):
    scene, md = reset(env, seed)
    idle(env, 100)
    A = 0 if md["side"] == "left" else 1
    c, u, (t0, t1), v = principal(env, scene)
    upright = abs(u[2]) > 0.7
    a = int(scene.qpos_adr[0])
    p, q = env.data.qpos[a:a + 3].copy(), env.data.qpos[a + 3:a + 7].copy()
    sA = 1.0 if A == 0 else -1.0
    if upright:
        phi = 0.0
    else:  # turn about z so the long axis points along +y (t_max end towards +y), then flip to A's side
        th = np.arctan2(u[1], u[0])
        phi = yaw_axis - th + (0.0 if sA > 0 else np.pi)
    R = Rz(phi)
    mx, mz = mid or (X_MID, Z_MID)
    target_c = np.array([mx, 0.0, mz])
    newp = target_c + R @ (p - c)
    newq = qmul(quat_of(R), q)
    set_object_pose(env, scene, 0, newp, newq)
    return scene, md, A, upright


def grasp_dict(env, scene, arm, T6):
    return {"arm": arm, "T_obj_link6": inv(scene.object_T(env.data, 0)) @ T6,
            "fingers": np.array([GRIP_MAX - 0.01, GRIP_MAX - 0.01])}


def plan(mode, env, scene, A, upright, flipA, flipB):
    B = 1 - A
    sA = 1.0 if A == 0 else -1.0
    c, u, (t0, t1), v = principal(env, scene)
    if u @ np.array([0, sA, 0]) < 0 and not upright:
        u = -u; t0, t1 = -t1, -t0
    L = t1 - t0
    if mode == "ends_topdown":
        if upright: return None
        tA, tB = t1 - 0.15 * L, t0 + 0.15 * L
        if L < 0.09: return None
        mA, _, _ = section(v, c, u, tA); mB, _, _ = section(v, c, u, tB)
        yaw = np.arctan2(u[1], u[0])
        return topdown_T6(mA, yaw, flipA), topdown_T6(mB, yaw, flipB)
    if mode == "top_side":
        lo, hi = v.min(0), v.max(0)
        H = hi[2] - lo[2]
        if H < 0.06: return None
        ptA = np.array([c[0], c[1], hi[2] - min(0.025, 0.3 * H)])
        ptB = np.array([c[0], c[1], lo[2] + min(0.03, 0.3 * H)])
        yaw = np.arctan2(u[1], u[0]) if not upright else 0.0
        return topdown_T6(ptA, yaw, flipA, below=0.0), side_T6(ptB, np.array([0, sA, 0]), np.pi / 2 * flipB)
    if mode == "side_side":
        lo, hi = v.min(0), v.max(0)
        H = hi[2] - lo[2]
        if H < 0.06: return None
        ptA = np.array([c[0], c[1], hi[2] - min(0.025, 0.3 * H)])
        ptB = np.array([c[0], c[1], lo[2] + min(0.03, 0.3 * H)])
        return side_T6(ptA, np.array([0, -sA, 0]), np.pi / 2 * flipA), side_T6(ptB, np.array([0, sA, 0]), np.pi / 2 * flipB)
    if mode == "ends_side":
        if upright or L < 0.04: return None
        tA, tB = t1 - 0.15 * L, t0 + 0.12 * L
        mA, _, _ = section(v, c, u, tA); mB, _, _ = section(v, c, u, tB)
        yaw = np.arctan2(u[1], u[0])
        return topdown_T6(mA, yaw, flipA), side_T6(mB, np.array([0, sA, 0]), np.pi / 2 * flipB)


def attempt(seed, mode, flipA, flipB, yaw_axis, mid):
    scene, md, A, upright = setup(seed, yaw_axis, mid)
    B = 1 - A
    P = plan(mode, env, scene, A, upright, flipA, flipB)
    if P is None:
        return None, "mode n/a"
    T6A, T6B = P
    T6A, tA = reachable(env, A, T6A)
    if T6A is None:
        return False, f"A: no IK at any tilt"
    T6B, tB = reachable(env, B, T6B)
    if T6B is None:
        return False, f"B: no IK at any tilt"
    T_objB = inv(scene.object_T(env.data, 0)) @ T6B
    ok, info = hold_object(env, scene, 0, grasp_dict(env, scene, A, T6A))
    if not ok:
        return False, "A: " + info.get("why", "?")
    qA = info["q"]
    T6B = scene.object_T(env.data, 0) @ T_objB  # B's grasp follows the object's current pose
    saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
    qB, why = teleport_arm(env, B, T6B)
    if qB is None:
        return False, "B: " + why
    hit = arm_contacts(env, B)
    if hit:
        return False, f"B teleport collides with {sorted(hit)}"
    for _ in range(15):
        env.step(two_arm_action(env, {A: qA, B: qB}, {A: 0.0, B: 0.0}), render_obs=False)
    o = scene.contact_obs(env.data)
    per = o["hits"].reshape(2, 2).sum(1)
    if per[A] < 2 or per[B] < 2:
        return False, f"both-hold failed fingers A{per[A]} B{per[B]}"
    a = int(scene.qpos_adr[0]); d0 = int(scene.dof_adr[0])
    p_rel = env.data.qpos[a:a + 3].copy()
    for _ in range(15):
        env.step(two_arm_action(env, {A: qA, B: qB}, {A: 1.0, B: 0.0}), render_obs=False)
    idsA = arm_ids(env.model, ARMS[A])
    T6 = np.eye(4); T6[:3, :3] = env.data.xmat[idsA.link6].reshape(3, 3); T6[:3, 3] = env.data.xpos[idsA.link6]
    sA = 1.0 if A == 0 else -1.0
    T6[:3, 3] += np.array([0.0, 0.06 * sA, 0.10])
    qU, pe, re = ik_link6(env.model, env.data, ARMS[A], T6)
    if pe > 0.01:
        return False, f"A retreat IK ({pe:.3f})"
    for i in range(30):
        w = (i + 1) / 30
        env.step(two_arm_action(env, {A: (1 - w) * qA + w * qU, B: qB}, {A: 1.0, B: 0.0}), render_obs=False)
    for _ in range(30):
        env.step(two_arm_action(env, {A: qU, B: qB}, {A: 1.0, B: 0.0}), render_obs=False)
    o = scene.contact_obs(env.data)
    per = o["hits"].reshape(2, 2).sum(1)
    drift = float(np.linalg.norm(env.data.qpos[a:a + 3] - p_rel))
    speed = float(np.linalg.norm(env.data.qvel[d0:d0 + 3]))
    siteA = env.data.site_xpos[scene.sites[A]]
    distA = float(np.linalg.norm(env.data.subtree_com[scene.bodies[0]] - siteA))
    others = set()
    for cc in env.data.contact[:env.data.ncon]:
        o1, o2 = scene.owner[cc.geom1], scene.owner[cc.geom2]
        for x, y in ((o1, o2), (o2, o1)):
            if x[0] == 0 and not (y[0] == 1 and y[1] // 2 == B):
                others.add("finger%d" % y[1] if y[0] == 1 else "other")
    good = per[B] == 2 and per[A] == 0 and distA > 0.06 and drift < 0.03 and speed < 0.05 and not others
    return bool(good), dict(fingers=per.tolist(), drift=round(drift, 4), speed=round(speed, 4), distA=round(distA, 3),
                            others=sorted(others), pos=np.round(env.data.qpos[a:a + 3], 3).tolist())


import collections
rows = []
t_start = time.time()
for seed in (SEEDS or range(first, last)):
    _, md = reset(env, seed)
    row = dict(seed=seed, cat=md["category"], var=md["variant"], side=md["side"], tries=[], ok=False)
    t_seed = time.time()
    for mode in MODES:
      for ya, mid in [(y, m) for m in MIDS for y in YAWS]:
        for fa in (0, 1):
            for fb in (0, 1):
                if time.time() - t_seed > SEED_BUDGET_S:
                    res, info = None, "time budget"
                    break
                try:
                    res, info = attempt(seed, mode, fa, fb, ya, mid)
                except Exception as e:  # noqa: BLE001
                    res, info = False, f"error {type(e).__name__}: {e}"[:120]
                if res is None:
                    break
                row["tries"].append([mode, round(float(np.rad2deg(ya))), mid, fa, fb, res, info])
                if res:
                    row.update(ok=True, mode=mode, yaw=float(np.rad2deg(ya)), mid=mid, info=info)
                    break
            if row["ok"] or res is None:
                break
        if row["ok"] or res is None:
            break
      if row["ok"]:
          break
    rows.append(row)
    print(seed, row["cat"], row["side"], "OK " + row.get("mode", "") if row["ok"] else "FAIL",
          row.get("info") if row["ok"] else collections.Counter(str(t[6])[:40] for t in row["tries"]).most_common(4), flush=True)
json.dump(dict(z_mid=Z_MID, x_mid=X_MID, rows=rows), open(out, "w"), default=str)
import collections
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    by[r["cat"]][0] += r["ok"]; by[r["cat"]][1] += 1
print({k: f"{a}/{n}" for k, (a, n) in sorted(by.items())}, f"total {sum(r['ok'] for r in rows)}/{len(rows)} {time.time()-t_start:.0f}s")
