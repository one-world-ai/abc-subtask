"""Phase 0 scorer probes for inhand_transfer_item_to_other_side (stock InhandTransferOtherSideEvaluator).

Per seed (spawn-launched seeds skipped, see ih_reset.py):
  held_far_B     object 3 cm above its rest height on the far side (y = -/+0.25), held top-down by the far arm B
  touch_far_B    held 6 mm above the far side by B, then B's fingers half open: it rests on the table, fingers near/on it
  reach_A        can the spawn-side arm A hold it top-down on the far side? (largest |y| past the middle, 0.10-0.30)
  drop_mid       released from rest in the middle (|y| <= 0.08, 10-25 cm up, random yaw) -> settles where?
  throw          released in the middle moving towards the far side at 0.4-1.2 m/s -> settles where?
  drop_far       released 15-25 cm above the far side (y = -/+0.25)
Every case records the stock success (as instantaneous success, the env's terminate-on-success signal) both at the
moment of interest and after settling, plus whether any robot geom touches the object and its speed.
usage: ih_scorer.py first last out.json"""
import collections, json, sys, time
import numpy as np
import mujoco
from abc_sim.task_eval.stagekit import ARMS, GRIP_MAX, hold_object, inv, set_object_pose, qmul, quat_of, axis_angle_quat
from ih_common import make, reset, score, idle, obj_pos
from ih_grasp import principal, section, topdown_T6, reachable, Rz

first, last, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
env = make()
rng = np.random.default_rng(first)


def robot_touch(scene):
    d = env.data; m = env.model
    for c in d.contact[:d.ncon]:
        for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if scene.owner[g1][0] == 0:
                b = int(m.geom_bodyid[g2]); n = m.body(b).name
                if n.startswith("left_") or n.startswith("right_"):
                    return True
    return False


def state(scene):
    r, s, mt = score(env)
    da = int(scene.dof_adr[0])
    return dict(success=s, reward=round(r, 3), robot_touch=robot_touch(scene),
                speed=round(float(np.linalg.norm(env.data.qvel[da:da + 3])), 3), pos=np.round(obj_pos(env, scene), 3).tolist())


def fresh(seed):
    scene, md = reset(env, seed)
    pen = min([0.0] + [float(c.dist) for c in env.data.contact[:env.data.ncon]
                       if scene.owner[c.geom1][0] == 0 or scene.owner[c.geom2][0] == 0])
    idle(env, 100)
    A = 0 if md["side"] == "left" else 1
    return scene, md, A, pen


def move_to(scene, xy, dz, yaw_extra=0.0):
    """Translate (and turn about z by yaw_extra) the settled object so its centre is over xy, lifted dz."""
    a = int(scene.qpos_adr[0])
    p, q = env.data.qpos[a:a + 3].copy(), env.data.qpos[a + 3:a + 7].copy()
    c, _, _, v = principal(env, scene)
    R = Rz(yaw_extra)
    newp = np.array([xy[0], xy[1], c[2] + dz]) + R @ (p - c)
    set_object_pose(env, scene, 0, newp, qmul(quat_of(R), q))


def topdown_hold(scene, arm):
    c, u, _, v = principal(env, scene)
    yaw = np.arctan2(u[1], u[0])
    mid, _, _ = section(v, c, u, 0.0)
    for flip in (0, 1):
        T6, _ = reachable(env, arm, topdown_T6(mid, yaw, flip))
        if T6 is None:
            continue
        g = {"arm": arm, "T_obj_link6": inv(scene.object_T(env.data, 0)) @ T6, "fingers": np.array([GRIP_MAX - 0.01] * 2)}
        ok, info = hold_object(env, scene, 0, g)
        if ok:
            return True, info
    return False, None


rows = []
t0 = time.time()
for seed in range(first, last):
    scene, md, A, pen = fresh(seed)
    row = dict(seed=seed, cat=md["category"], side=md["side"], pen0=round(pen, 4))
    if pen < -0.005:
        row["skipped"] = "spawn overlap (launch risk)"; rows.append(row); continue
    sA = 1.0 if A == 0 else -1.0
    B = 1 - A
    # held_far_B
    move_to(scene, (0.60, -sA * 0.25), 0.03)
    ok, _ = topdown_hold(scene, B)
    row["held_far_B"] = state(scene) if ok else "no hold"
    # touch_far_B: rest on far side, B's gripper open around it touching (after hold, open the fingers but stay)
    if ok:
        scene, md, A, pen = fresh(seed)
        move_to(scene, (0.60, -sA * 0.25), 0.006)
        ok2, info = topdown_hold(scene, B)
        if ok2:
            from abc_sim.task_eval.stagekit import hold_action
            act = hold_action(env, B, info["q"], 0.0)
            act[B * 7 + 6] = 0.35  # partly open: fingers still brushing the object
            for _ in range(40):
                env.step(act, render_obs=False)
            row["touch_far_B"] = state(scene)
    # reach_A: furthest far-side |y| at which A holds it top-down 3 cm up
    best = None
    for y in (0.10, 0.15, 0.20, 0.25, 0.30):
        scene, md, A, pen = fresh(seed)
        move_to(scene, (0.60, -sA * y), 0.03)
        ok, _ = topdown_hold(scene, A)
        if ok:
            best = y; row["held_far_A"] = state(scene)
        else:
            break
    row["reach_A_y"] = best
    # drops / throws
    for kind, n in (("drop_mid", 4), ("throw", 4), ("drop_far", 2)):
        res = []
        for _ in range(n):
            scene, md, A, pen = fresh(seed)
            if kind == "drop_far":
                move_to(scene, (rng.uniform(0.5, 0.7), -sA * 0.25), rng.uniform(0.15, 0.25), rng.uniform(-np.pi, np.pi))
            else:
                move_to(scene, (rng.uniform(0.45, 0.70), rng.uniform(-0.08, 0.08)), rng.uniform(0.10, 0.25), rng.uniform(-np.pi, np.pi))
            if kind == "throw":
                da = int(scene.dof_adr[0])
                env.data.qvel[da:da + 3] = [rng.uniform(-0.2, 0.2), -sA * rng.uniform(0.4, 1.2), rng.uniform(0.0, 0.5)]
            first_succ = None
            a = env.get_state().astype(np.float32)
            for i in range(300):
                env.step(a, render_obs=False)
                if first_succ is None and score(env)[1]:
                    first_succ = i
                    st_first = state(scene)
            end = state(scene)
            res.append(dict(first_success_step=first_succ, first=st_first if first_succ is not None else None, end=end))
        row[kind] = res
    rows.append(row)
    print(seed, row["cat"], row["side"], "heldB", (row["held_far_B"] or {}).get("success") if isinstance(row["held_far_B"], dict) else row["held_far_B"],
          "touchB", row.get("touch_far_B", {}).get("success"), "reachA", row["reach_A_y"],
          {k: sum(r["end"]["success"] for r in row[k]) for k in ("drop_mid", "throw", "drop_far")}, flush=True)
json.dump(rows, open(out, "w"))
print(f"{time.time()-t0:.0f}s")
