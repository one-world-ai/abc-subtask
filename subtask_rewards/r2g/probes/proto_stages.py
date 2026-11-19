"""Feasibility prototype: reach / pick / carry / place stages on a Real2Gym obj env, read from the env's own
evaluator (Real2GymEvaluator). A scripted pick-and-place is cut at stage boundaries; each boundary state is saved,
restored (exactness), held idle for 10 chunks of 32 (does the stage label survive?), and the solver is resumed from
the restored state (does the env's own check still reach success from a mid-task start?).

  ABC_REPO=<abc-r2g> ABC_TASKS_TOOLS=<abc_tasks/tools> NO_RENDER=1 python proto_stages.py ENV CONTAINER OUT.json [seeds]
"""
import json, os, sys
import mujoco, numpy as np
ROOT = os.environ["ABC_REPO"]
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools"), os.environ["ABC_TASKS_TOOLS"]]
import abc_sim  # noqa: E402
from scripted_rollouts import Rollout, down_frames, TABLE_Z  # noqa: E402
import real2gym_scripted as RS  # noqa: E402

SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
STAGES = ("reach", "pick", "carry", "place", "done")
NEAR_M = 0.05


def measure(env, ro, cont):
    """Stage facts from the sim state and the env's evaluator."""
    ev, d = env._task_evaluator, env.data
    spec = ev._spec
    n, arm = spec["object"], spec["arm"]
    touch = ev._pad_touch(d)
    pin = ev._pin(touch, arm, n)
    lift = float(d.xipos[ev._body[n]][2] - ev._ref[n][0][2])
    site = ro.robot.site(arm)[0]
    dist = float(np.linalg.norm(site - d.xipos[ev._body[n]]))
    free = pin and lift >= ev.lift_m and not ev._blocked(d, n, touch, (arm,))
    over = ev._in_box(d, cont, n, above=1.0)
    info = env.evaluate_task().to_info(squeeze=True)
    return dict(pin=bool(pin), lift=lift, dist=dist, free=bool(free), over=bool(over), goal_met=bool(info["goal_met"]),
                success=bool(info["success"]), pinch_s=float(info["pinch_s"]))


def label(m):
    if m["success"]:
        return 4
    if m["free"] and m["over"]:
        return 3            # held over the container: place
    if m["free"]:
        return 2            # held and lifted clear: carry
    if m["dist"] < NEAR_M and not m["pin"] and m["lift"] < 0.01:
        return 1            # gripper at the object: pick
    return 0


def plan(env, ro, cont):
    ev = env._task_evaluator
    n = ev._spec["object"]
    arm = ev._spec["arm"]
    c, Rc = RS.body(env, cont)
    ext = RS.load_library(ro.task)["objects"]
    ext = {o["name"]: o for o in ext}[cont]["extent"]
    p, R = RS.body(env, n)
    com = env.data.xipos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)].copy()
    grasp, closing = com.copy(), R[:, 1]
    dest = c[:2] + Rc[:2, :2] @ np.array([(ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2])
    place = np.array([dest[0], dest[1], max(RS.top_z(env, cont), c[2] + ext[4]) + grasp[2] - TABLE_Z + 0.02])
    return arm, grasp, closing, place


def segments(ro, arm, grasp, closing, place, lift=0.14):
    """The scripted solver as (stage it completes, callable) pieces."""
    st = {}
    def reach():
        st["R"] = ro.move(arm, grasp + [0, 0, lift], down_frames(ro, arm, grasp, closing), grip=1.0, speed=0.3)
        ro.move(arm, grasp, st["R"], speed=0.10)
    def pick():
        if "R" not in st:
            st["R"] = ro.robot.site(arm)[1]
        ro.grip(arm, 0.0, 0.8); ro.move(arm, grasp + [0, 0, lift], st["R"], speed=0.15)
    def carry():
        if "R" not in st:
            st["R"] = ro.robot.site(arm)[1]
        ro.move(arm, place + [0, 0, lift], st["R"], speed=0.2)
    def placef():
        if "R" not in st:
            st["R"] = ro.robot.site(arm)[1]
        ro.move(arm, place, st["R"], speed=0.08); ro.grip(arm, 1.0, 0.6)
        ro.move(arm, place + [0, 0, lift], st["R"], speed=0.2); ro.hold(1.0)
    return [("reach", reach), ("pick", pick), ("carry", carry), ("place", placef)]


def main():
    name, cont, out = sys.argv[1:4]
    seeds = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else "0,1,2,3,4,5,6,7").split(",")]
    task = name.split("__")[0][len("r2g_"):]
    env = abc_sim.make_env(task=name, render_cameras=False, max_episode_steps=100000, terminate_on_success=False)
    rows = []
    for seed in seeds:
        env.forget_arm_state(); env.reset(seed=seed)
        ro = Rollout(env, [], None, ""); ro.task = task
        ro.hold(0.3)
        arm, grasp, closing, place = plan(env, ro, cont)
        row = dict(seed=seed, arm=arm, valid=bool(env._task_evaluator._spec["valid"]), labels=[], snaps={})
        snaps = {}
        for k, (stage, fn) in enumerate(segments(ro, arm, grasp, closing, place)):
            m = measure(env, ro, cont)
            row["labels"].append((stage, STAGES[label(m)], round(m["lift"], 3), round(m["dist"], 3), round(m["pinch_s"], 2)))
            if label(m) == k:
                st = np.zeros(mujoco.mj_stateSize(env.model, SPEC)); mujoco.mj_getState(env.model, env.data, st, SPEC)
                snaps[stage] = (st, ro.robot.action.copy(), dict(env._task_evaluator._pinch_t))
            fn()
        m = measure(env, ro, cont)
        row["final"] = STAGES[label(m)]; row["success"] = m["success"]
        # per saved start: exact restore, idle hold, resume the solver
        for k, stage in enumerate(STAGES[:4]):
            if stage not in snaps:
                continue
            st, act, pinch = snaps[stage]
            res = {}
            for mode in ("idle", "solve", "solve_keep_pinch"):
                env.reset_to_snapshot(seed, st)
                back = np.zeros_like(st); mujoco.mj_getState(env.model, env.data, back, SPEC)
                res["restore_diff"] = float(np.abs(back - st).max())
                ro = Rollout(env, [], None, ""); ro.task = task; ro.robot.action[:] = act
                if mode == "solve_keep_pinch":
                    env._task_evaluator._pinch_t.update(pinch)
                if mode == "idle":
                    ro.step(320)
                    m = measure(env, ro, cont)
                    res["idle_label"] = STAGES[label(m)]; res["idle_success"] = m["success"]
                else:
                    for s2, fn in segments(ro, arm, grasp, closing, place)[k:]:
                        fn()
                    res[mode] = measure(env, ro, cont)["success"]
            row["snaps"][stage] = res
        rows.append(row); print(json.dumps(row), flush=True)
    json.dump(rows, open(out, "w"), indent=1)
    print("\nSUMMARY", name, "seeds", len(rows))
    print(" full solver success:", sum(r["success"] for r in rows), "/", len(rows))
    for stage in STAGES[:4]:
        got = [r["snaps"][stage] for r in rows if stage in r["snaps"]]
        print(f" {stage}: starts {len(got)}/{len(rows)} | exact restore {sum(g['restore_diff'] == 0 for g in got)} | "
              f"still in stage after 10 idle chunks {sum(g['idle_label'] == stage for g in got)} | idle success {sum(g['idle_success'] for g in got)} | "
              f"solver from start {sum(g['solve'] for g in got)} | with pinch time carried {sum(g['solve_keep_pinch'] for g in got)}")


main()
