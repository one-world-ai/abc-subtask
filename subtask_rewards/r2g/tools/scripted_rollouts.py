"""Scripted solver and reward-hack rollouts of the Real2Gym stage envs (gates G4 and G5).

  python scripted_rollouts.py TASK STAGE STARTS_ROOT OUT_DIR [--n 20] [--chunks 12]

Each behaviour runs from saved starts (evenly spaced rows of STARTS_ROOT/r2g_<task>/<stage>.npz) through the public
API: env.reset_to_stage(seed, row), env.step, env.subtask.chunk_reward() every 32 actions. A run ends at the first
+1 / -1 or after ``chunks`` chunks. Writes OUT_DIR/<task>__<stage>.json: one record per run and a summary per
behaviour (outcomes among the runs that could be planned, mean return, share as expected).
"""
import argparse, collections, json
from pathlib import Path

import numpy as np

import abc_sim
from abc_sim.task_eval.r2g_stages import PINCH
from abc_sim.task_eval.bottles_starts import idle_action, move_link
from abc_sim.task_eval.r2g_starts import TILTS_DEG, closing_yaws, over_container_pose, pinch_grasp
from abc_sim.task_eval.stagekit import ARMS, T, arm_ids, load_starts

CHUNK = 32


class Done(Exception):
    pass


class Run:
    """Counts actions, collects the chunk rewards and stops the behaviour at a terminal chunk."""

    def __init__(self, env, chunks):
        self.env, self.ev, self.max_steps = env, env.subtask, chunks * CHUNK
        self.steps, self.ret, self.outcome = 0, 0.0, 0
        self.actions, self.rewards = [], []
        self._step = env.step
        env.step = self.step

    def step(self, action, **kw):
        out = self._step(action, **kw)
        self.steps += 1
        self.actions.append(np.array(action, dtype=np.float64))
        if self.steps % CHUNK == 0:
            r, done = self.ev.chunk_reward()
            self.rewards.append(float(r))
            if done:
                self.outcome = int(np.sign(r))
                self.ret += r
                raise Done
            self.ret += r
        if self.steps >= self.max_steps:
            raise Done
        return out

    def close(self):
        self.env.step = self._step


def link6(env, arm):
    ids = arm_ids(env.model, ARMS[arm])
    return T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])


def shifted(pose, d):
    out = pose.copy()
    out[:3, 3] += np.asarray(d, float)
    return out


def hold(env, n, arm=None, grip=None):
    act = idle_action(env)
    if arm is not None:
        act[arm * 7 + 6] = grip
    for _ in range(n):
        env.step(act, render_obs=False)


def go(env, arm, goal, grip, actions=64):
    return move_link(env, arm, goal, actions=actions, grip=grip)[0]


def reach_with(env, sc, k, arm, grip=1.0):
    """Move ``arm`` open to a grasp pose on object k through a point 10 cm above it. False if nothing could be planned."""
    obj = sc.object_T(env.data, k)
    for tilt in TILTS_DEG:
        for yaw in closing_yaws(env, sc, k):
            goal = obj @ pinch_grasp(env, sc, k, yaw, arm, 0.01, tilt)["T_obj_link6"]
            if go(env, arm, shifted(goal, [0, 0, 0.10]), grip, 64):
                if go(env, arm, goal, grip, 48):
                    return True
                return False  # moved to the pre-grasp point already: one attempt per run
    for tilt in TILTS_DEG + (50.0,):  # no pre-grasp point reachable: straight to the grasp pose
        for yaw in closing_yaws(env, sc, k):
            if go(env, arm, obj @ pinch_grasp(env, sc, k, yaw, arm, 0.01, tilt)["T_obj_link6"], grip, 96):
                return True
    return False


def container_dir(env, sc, k):
    """Unit horizontal direction from object k to its container, and the container's largest half-size."""
    f = sc.full
    c = sc.goal_of[sc.names[k]]["containers"][0]
    ext = f._objs[c]["extent"]
    centre = env.data.xpos[f._body[c]] + env.data.xmat[f._body[c]].reshape(3, 3) @ np.array(
        [(ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2, 0.0])
    d = centre[:2] - env.data.xipos[sc.bodies[k]][:2]
    n = float(np.linalg.norm(d))
    return (d / n if n > 1e-6 else np.array([1.0, 0.0])), max(ext[1] - ext[0], ext[3] - ext[2]) / 2, c


# ---------------------------------------------------------------------------------------------- behaviours
# each returns False when it could not be planned (the run then says nothing about the reward)
def b_idle(env, sc, k, arm, rng):
    hold(env, 10 ** 6)


def reach_solver(env, sc, k, arm, rng):
    ok = reach_with(env, sc, k, arm)
    if ok:
        hold(env, 10 ** 6)
    return ok


def reach_away(env, sc, k, arm, rng):
    obj = env.data.xipos[sc.bodies[k]]
    here = link6(env, arm)
    d = here[:3, 3] - obj
    d[2] = 0.0
    d = d / max(float(np.linalg.norm(d)), 1e-6)
    # away from the object: sideways and up, else straight up, else down a little (whatever the arm can do from home)
    for step in ([0.08 * d[0], 0.08 * d[1], 0.05], [0.0, 0.0, 0.06], [0.06 * d[0], 0.06 * d[1], 0.0],
                 [0.05 * d[0], 0.05 * d[1], -0.04]):
        if go(env, arm, shifted(here, step), 1.0, 64):
            hold(env, 10 ** 6)
            return None
    return False


def reach_other_arm(env, sc, k, arm, rng):
    ok = reach_with(env, sc, k, 1 - arm)
    if ok:
        hold(env, 10 ** 6)
    return ok


def pick_solver(env, sc, k, arm, rng):
    d, site = env.data, sc.sites[arm]
    dz = float((d.site_xpos[site] + d.site_xmat[site].reshape(3, 3) @ PINCH)[2] - d.xipos[sc.bodies[k]][2])
    if dz > 0.003:  # starts are up to 3 cm above the object: bring the pads down to it before closing
        go(env, arm, shifted(link6(env, arm), [0, 0, -dz]), 1.0, 32)
    hold(env, 25, arm, 0.0)
    ok = go(env, arm, shifted(link6(env, arm), [0, 0, 0.08]), 0.0, 64)
    if ok:
        hold(env, 10 ** 6, arm, 0.0)
    return ok


def pick_close_only(env, sc, k, arm, rng):
    hold(env, 10 ** 6, arm, 0.0)


def pick_push(env, sc, k, arm, rng):
    d = container_dir(env, sc, k)[0]
    side = np.array([-d[1], d[0]])  # sideways, not towards the container
    here = link6(env, arm)
    ok = go(env, arm, shifted(here, [0.14 * side[0], 0.14 * side[1], 0.0]), 1.0, 64) or \
        go(env, arm, shifted(here, [-0.14 * side[0], -0.14 * side[1], 0.0]), 1.0, 64)
    if ok:
        hold(env, 10 ** 6)
    return ok


def pick_other_arm(env, sc, k, arm, rng):
    if not go(env, arm, shifted(link6(env, arm), [0, 0, 0.15]), 1.0, 48):
        return False
    other = 1 - arm
    if not reach_with(env, sc, k, other):
        return False
    hold(env, 25, other, 0.0)
    go(env, other, shifted(link6(env, other), [0, 0, 0.08]), 0.0, 64)
    hold(env, 10 ** 6, other, 0.0)


def carry_solver(env, sc, k, arm, rng):
    for _ in range(6):  # another point over the container / another height when one cannot be planned
        goal, _ = over_container_pose(env, sc, k, arm, rng)
        if goal is not None and go(env, arm, goal, 0.0, 96):
            hold(env, 10 ** 6, arm, 0.0)
            return None
    return False


def carry_wrong_way(env, sc, k, arm, rng):
    d = container_dir(env, sc, k)[0]
    here = link6(env, arm)
    if not go(env, arm, shifted(here, [-0.10 * d[0], -0.10 * d[1], 0.0]), 0.0, 64):
        return False
    hold(env, 10 ** 6, arm, 0.0)


def carry_open_early(env, sc, k, arm, rng):
    hold(env, 10 ** 6, arm, 1.0)


def place_solver(env, sc, k, arm, rng):
    f, n = sc.full, sc.names[k]
    c = sc.goal_of[n]["containers"][0]
    lo, _ = f._zrange(env.data, f._geoms[n])
    floor = env.data.xpos[f._body[c]][2] + f._objs[c]["extent"][4]
    drop = max(lo - floor - 0.015, 0.0)
    if not go(env, arm, shifted(link6(env, arm), [0, 0, -drop]), 0.0, 64):
        return False
    hold(env, 20, arm, 1.0)
    go(env, arm, shifted(link6(env, arm), [0, 0, 0.10]), 1.0, 48)
    hold(env, 10 ** 6, arm, 1.0)


def place_release_beside(env, sc, k, arm, rng):
    d, half, _ = container_dir(env, sc, k)
    obj = env.data.xipos[sc.bodies[k]]
    f = sc.full
    c = sc.goal_of[sc.names[k]]["containers"][0]
    centre = env.data.xpos[f._body[c]][:2]
    out = obj[:2] - centre
    out = out / np.linalg.norm(out) if np.linalg.norm(out) > 1e-3 else np.array([0.0, 1.0 if arm == 0 else -1.0])
    here = link6(env, arm)
    for sgn in (1.0, -1.0):
        step = sgn * out * (half + 0.10)
        if go(env, arm, shifted(here, [step[0], step[1], 0.0]), 0.0, 80):
            hold(env, 10 ** 6, arm, 1.0)
            return None
    return False


def place_hold_still(env, sc, k, arm, rng):
    hold(env, 10 ** 6, arm, 0.0)


def place_drop_high(env, sc, k, arm, rng):
    hold(env, 10 ** 6, arm, 1.0)


# behaviour -> (function, expected outcome: 1, -1, 0, "not+1" (0 or -1) or "any")
BEHAVIOURS = {
    "reach": {"solver": (reach_solver, 1), "move_away": (reach_away, "not+1"), "idle": (b_idle, 0),
              "other_arm": (reach_other_arm, "not+1")},
    "pick": {"solver": (pick_solver, 1), "close_only": (pick_close_only, "not+1"), "push": (pick_push, "not+1"),
             "idle": (b_idle, 0), "other_arm": (pick_other_arm, "not+1")},
    "carry": {"solver": (carry_solver, 1), "wrong_way": (carry_wrong_way, "not+1"),
              "open_early": (carry_open_early, -1), "idle": (b_idle, 0)},
    "place": {"solver": (place_solver, 1), "release_beside": (place_release_beside, -1),
              "hold_still": (place_hold_still, 0), "drop_high": (place_drop_high, "any")},
}


def as_expected(outcome, expect):
    return expect == "any" or outcome == expect or (expect == "not+1" and outcome != 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task"); ap.add_argument("stage"); ap.add_argument("starts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--chunks", type=int, default=12)
    a = ap.parse_args()
    starts = load_starts(a.starts, f"r2g_{a.task}", a.stage)
    rows = [starts[int(i)] for i in np.linspace(0, len(starts) - 1, min(a.n, len(starts))).round().astype(int)]
    env = abc_sim.make_env(task=f"r2g_{a.task}__{a.stage}", render_cameras=False)
    for old in (a.out / a.task / a.stage).glob("*.npz"):  # records of an earlier start set would replay differently
        old.unlink()
    runs = []
    for name, (fn, expect) in BEHAVIOURS[a.stage].items():
        for i, row in enumerate(rows):
            env.reset_to_stage(row["seed"], row)
            ev, sc = env.subtask, env.subtask.scene
            k, arm = int(ev.target), int(sc.arm[ev.target])
            run = Run(env, a.chunks)
            planned = True
            try:
                planned = fn(env, sc, k, arm, np.random.default_rng(1000 + i)) is not False
                if planned:
                    hold(env, 10 ** 6)
            except Done:
                pass
            finally:
                run.close()
            last = ev.last
            runs.append(dict(behaviour=name, seed=int(row["seed"]), object=sc.names[k], arm=ARMS[arm],
                             planned=bool(planned), outcome=run.outcome, ret=round(run.ret, 4), steps=run.steps,
                             failed=ev.failure, label=last.get("label"), stock_success=bool(last.get("stock_ever_success")),
                             wrong_arm=bool(last.get("wrong_arm")), expect=expect,
                             as_expected=bool(planned and as_expected(run.outcome, expect))))
            print(json.dumps(runs[-1]), flush=True)
            d = a.out / a.task / a.stage  # replayable record (actions from the saved start, chunk rewards) for the videos
            d.mkdir(parents=True, exist_ok=True)
            keep = planned and run.actions
            np.savez_compressed(d / f"{name}_{int(row['seed'])}.npz",
                                actions=np.asarray(run.actions if keep else np.zeros((0, 14))),
                                rewards=np.asarray(run.rewards if keep else []))
    env.close()
    summary = {}
    for name, (_, expect) in BEHAVIOURS[a.stage].items():
        rs = [r for r in runs if r["behaviour"] == name]
        pl = [r for r in rs if r["planned"]]
        summary[name] = dict(expect=expect, starts=len(rs), planned=len(pl),
                             plus=sum(r["outcome"] == 1 for r in pl), minus=sum(r["outcome"] == -1 for r in pl),
                             none=sum(r["outcome"] == 0 for r in pl), as_expected=sum(r["as_expected"] for r in pl),
                             mean_return=round(float(np.mean([r["ret"] for r in pl])), 3) if pl else None,
                             failures=dict(collections.Counter(r["failed"] for r in pl if r["failed"])))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.task}__{a.stage}.json").write_text(json.dumps(dict(task=a.task, stage=a.stage, summary=summary, runs=runs), indent=1))
    (a.out / a.task / f"summary_{a.stage}.json").write_text(json.dumps(dict(summary=summary), indent=1))
    print("SUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
