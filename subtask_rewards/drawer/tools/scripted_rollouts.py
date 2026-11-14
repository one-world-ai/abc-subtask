"""Drawer scripted solvers and reward hacks (gates G4/G5), run from the saved starts exactly like an RL loop:
physical 14-D joint actions, chunk_reward() after every 32 actions, the episode ends at the first terminal chunk.

  MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=<abc-subtask-drawer> python scripted_rollouts.py \
      --mode open --stage pull --starts ../../../outputs/subtask_rewards/drawer/starts --n 20 \
      --out ../../../outputs/subtask_rewards/drawer/scripted

Motions: Cartesian moves of the finger-pad centre (pinch) with per-action damped-LS IK on link_6 (seeded on the
previous action's joints, falling back to the joint-space line to an endpoint solved with restarts), like the
overlay's scripted_rollouts.py. An endpoint with no IK (> 3 mm / 0.05 rad) is a planning failure ("unplanned").
After a script ends the arm is held still for up to TAIL chunks (debounce + release windows), then the episode is cut.

EXPECT: "+1", "-1", "0" (no terminal), or a set such as "0|-1". Kinds: solver (G4), alt (another legal way to reach
the goal: checked for EXPECT, not counted as a hack), hack (G5: EXPECT on >= 80 % of the planned rollouts with at
least half the starts planned, +1 on <= 10 % of all starts, margin >= 0.5), info (reported only).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.drawer_stages import DRAWERS, PINCH
from abc_sim.task_eval.drawer_starts import (approach_dir, frame, handle_pose, idle_action, link6_for_pinch,
                                             pitched, side_dir)
from abc_sim.task_eval.stagekit import ARMS, actuator_limits, arm_ids, ik_link6

CHUNK = 32
TAIL = 4
MAX_CHUNKS = 40
ROT_TOL = 0.3  # rad, relaxed endpoint orientation tolerance of the planner
TASK = {"open": "open_drawer", "close": "close_drawer"}
NEIGHBOUR = {"top": "middle", "middle": "bottom", "bottom": "middle"}
FAR = {"top": "bottom", "middle": "top", "bottom": "top"}  # farthest other drawer (close reach wrong_drawer)


class Finished(Exception):
    pass


class Unplanned(Exception):
    pass


class Episode:
    """Steps the env, scores each 32-action chunk, stops at the first terminal chunk (raises Finished)."""

    def __init__(self, env, start):
        self.env, self.m, self.d = env, env.model, env.data
        env.reset_to_stage(int(start["seed"]), start)
        self.ev = env._task_evaluator
        self.k = self.ev.target
        self.scene = self.ev.scene
        self.act = idle_action(env).astype(np.float64)
        self.n, self.rewards, self.done, self.outcome = 0, [], False, "0"
        self.actions = []  # every action sent to env.step (for replay videos)
        self.qmax = self.qmin = float(self.d.qpos[self.scene.qpos_adr[self.k]])
        self.others_max = 0.0
        self.scratch = mujoco.MjData(self.m)
        self.dt = env._control_decimation * self.m.opt.timestep

    def step(self, n=1):
        for _ in range(n):
            a = self.act.astype(np.float32)
            self.actions.append(a.copy())
            self.env.step(a, render_obs=False)
            self.n += 1
            q = self.d.qpos[self.scene.qpos_adr]
            self.qmax, self.qmin = max(self.qmax, float(q[self.k])), min(self.qmin, float(q[self.k]))
            self.others_max = max(self.others_max, max(float(q[j]) for j in range(3) if j != self.k))
            if self.n % CHUNK == 0:
                r, done = self.ev.chunk_reward()
                self.rewards.append(r)
                if done:
                    self.done, self.outcome = True, "+1" if r > 0 else "-1"
                    raise Finished
                if len(self.rewards) >= MAX_CHUNKS:
                    raise Finished

    def tail(self, chunks=TAIL):
        """Hold still (current targets) until the end of the current chunk + ``chunks`` chunks."""
        self.step((CHUNK - self.n % CHUNK) % CHUNK + chunks * CHUNK)

    # -- geometry -------------------------------------------------------------------------------------------
    def bar(self, k=None):
        b, p = self.scene.bar(self.d)
        k = self.k if k is None else k
        return b[k].copy(), p[k].copy()

    def site(self, arm):
        s = self.scene.sites[arm]
        R = self.d.site_xmat[s].reshape(3, 3).copy()
        return self.d.site_xpos[s] + R @ PINCH, R

    # -- motion ---------------------------------------------------------------------------------------------
    def ik(self, arm, pinch, R, q_seed, seeds=1, iters=60):
        sd = self.scratch
        sd.qpos[:] = self.d.qpos
        sd.qpos[arm_ids(self.m, ARMS[arm]).qadr] = q_seed
        return ik_link6(self.m, sd, ARMS[arm], link6_for_pinch(self.m, arm, pinch, R), seeds=seeds, iters=iters,
                        rng=np.random.default_rng(self.n), limits=actuator_limits(self.m, ARMS[arm]))

    def solve(self, arm, pinch, Rs):
        """Endpoint IK: the first candidate rotation within 3 mm / 0.05 rad; else the first within 3 mm / ROT_TOL
        (the overlay's planner checked position only; a held handle or a push tolerates some wrist twist)."""
        cur = self.act[arm * 7:arm * 7 + 6].copy()
        Rs = Rs if isinstance(Rs, list) else [Rs]
        sols = []
        for R in Rs:
            q, pe, re = self.ik(arm, pinch, R, cur, seeds=12, iters=200)
            if q is not None and pe < 3e-3 and re < 0.05:
                return q, R
            sols.append((q, pe, re, R))
        for q, pe, re, R in sols:
            if q is not None and pe < 3e-3 and re < ROT_TOL:
                return q, R
        raise Unplanned(f"no IK for pinch {np.round(pinch, 3).tolist()}")

    def move(self, arm, pinch, R=None, grip=None, speed=0.15, min_t=0.3):
        """Cartesian move of ``arm``'s pinch to ``pinch`` (R: rotation or candidates; default: keep). -> R used."""
        p0, R0 = self.site(arm)
        pinch = np.asarray(pinch, float)
        qa = self.act[arm * 7:arm * 7 + 6].copy()
        qb, R = self.solve(arm, pinch, R0 if R is None else R)
        n = max(int(max(np.linalg.norm(pinch - p0) / speed, min_t) / self.dt), 1)
        qprev = qa
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)
            Ri = _slerp(R0, R, s)
            q, pe, _ = self.ik(arm, p0 + s * (pinch - p0), Ri, qprev)
            if q is None or pe > 0.01:
                q = qa + s * (qb - qa)
            q = qb if i == n else q
            self.act[arm * 7:arm * 7 + 6] = q
            qprev = q
            if grip is not None:
                self.act[arm * 7 + 6] = grip
            self.step(1)
        return R

    def grip(self, arm, value, t=0.6):
        self.act[arm * 7 + 6] = value
        self.step(max(int(t / self.dt), 1))

    def hold(self, t):
        self.step(max(int(t / self.dt), 1))


def _slerp(R0, R1, t):
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, np.ascontiguousarray(R0).ravel())
    mujoco.mju_mat2Quat(q1, np.ascontiguousarray(R1).ravel())
    if q0 @ q1 < 0:
        q1 = -q1
    q = (1 - t) * q0 + t * q1
    q /= np.linalg.norm(q)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


# ------------------------------------------------------------------------------------------------ primitives
def open_cands(pull):
    return [(approach_dir(pull, t), frame(approach_dir(pull, t), [0, 0, 1])) for t in (15, 30, 22, 40, 10)]


def close_cands(pull):
    return [(approach_dir(pull, t), frame(approach_dir(pull, t), [0, 0, 1])) for t in (30, 22, 40, 15)]


def pick_arm(ep, k=None, avoid=None):
    bar, _ = ep.bar(k)
    order = [1, 0] if bar[1] < 0 else [0, 1]
    return [a for a in order if a != avoid]


def approach_of(R):
    return R[:, 2].copy()


def plan(ep, arms, cands, points):
    """First (arm, approach, R) whose every waypoint ``points(a)`` has IK (checked before anything moves)."""
    for arm in arms:
        for a, R in cands:
            try:
                for p in points(a):
                    ep.solve(arm, p, R)
            except Unplanned:
                continue
            return arm, a, R
    raise Unplanned("no arm/approach reaches every waypoint")


def grasp_handle(ep, arms, k=None, close=True, pre=0.10):
    """Open gripper 10 cm in front of the bar, in to the bar (pads straddling it), close. -> arm used."""
    bar, pull = ep.bar(k)
    arm, a, R = plan(ep, [arms] if np.isscalar(arms) else arms, open_cands(pull),
                     lambda a: (bar - a * pre, bar - a * 0.004))
    ep.move(arm, bar - a * pre, R, grip=1.0, speed=0.3)
    ep.move(arm, bar - a * 0.004, R, speed=0.06)
    if close:
        ep.grip(arm, 0.0, 0.7)
    return arm


def fingertips_on_face(ep, arms, k=None, pre=0.08):
    """Closed fingertips onto the bar's front face (overlay close approach, vertical closing). -> arm used."""
    bar, pull = ep.bar(k)
    tip = lambda a: bar + pull * 0.006 - a * 0.019
    arm, a, R = plan(ep, [arms] if np.isscalar(arms) else arms, close_cands(pull),
                     lambda a: (tip(a) - a * pre, tip(a)))
    ep.move(arm, tip(a) - a * pre, R, grip=0.0, speed=0.3)
    ep.move(arm, tip(a), R, speed=0.06)
    return arm


def pull_by(ep, arm, dist, speed=0.10, k=None):
    """Move the gripper ``dist`` along drawer k's pull axis; the wrist may pitch up to 20 deg about the bar."""
    _, pull = ep.bar(k)
    p, R = ep.site(arm)
    ep.move(arm, p + pull * dist, pitched(R, side_dir(pull)), speed=speed)


def back_off(ep, arm, dist=0.10, grip=None, speed=0.2):
    p, R = ep.site(arm)
    _, pull = ep.bar()
    ep.move(arm, p - approach_of(R) * dist, pitched(R, side_dir(pull), (0, 10, -10, 20, -20, 30, -30)), grip=grip,
            speed=speed)


def away(ep, arm, speed=0.2):
    p, R = ep.site(arm)
    _, pull = ep.bar()
    for d in (np.array([-0.12, 0.0, 0.10]), np.array([0.0, 0.0, 0.12]), pull * 0.06 + np.array([0, 0, 0.08])):
        try:
            ep.solve(arm, p + d, pitched(R, side_dir(pull), (0, 15, -15, 30, -30)))
        except Unplanned:
            continue
        ep.move(arm, p + d, pitched(R, side_dir(pull), (0, 15, -15, 30, -30)), speed=speed)
        return
    raise Unplanned("no way away")


def pull_neighbour(ep, arms, dist=0.09, which=NEIGHBOUR):
    """A gripper grasps another drawer's handle (``which`` of the target) and pulls it ``dist`` open."""
    nb = DRAWERS.index(which[DRAWERS[ep.k]])
    arm = grasp_handle(ep, arms, k=nb)
    pull_by(ep, arm, dist, k=nb)


def open_far_low(ep, arms, dist=0.12):
    """Open the drawer farthest from the target, routed around the (open, protruding) target: first out to 30 cm in
    front of that drawer's handle at its height, then in at a shallow 5-15 deg, grasp, pull ``dist``."""
    nb = DRAWERS.index(FAR[DRAWERS[ep.k]])
    bar, pull = ep.bar(nb)
    cands = [(approach_dir(pull, t), frame(approach_dir(pull, t), [0, 0, 1])) for t in (10, 5, 15)]
    arm, a, R = plan(ep, arms, cands, lambda a: (bar - a * 0.30, bar - a * 0.10, bar - a * 0.004))
    ep.move(arm, bar - a * 0.30, R, grip=1.0, speed=0.3)
    ep.move(arm, bar - a * 0.10, R, speed=0.15)
    ep.move(arm, bar - a * 0.004, R, speed=0.06)
    ep.grip(arm, 0.0, 0.7)
    pull_by(ep, arm, dist, k=nb)


def other_arm_opens_neighbour(ep, avoid, dist=0.12):
    pull_neighbour(ep, [a for a in (0, 1) if a != avoid], dist)


def hook_pull(ep, arms, dist=0.17, lateral=0.045, pre=0.08):
    """No-grasp open (phase 0 hook): closed fingertips enter beside the bar's end, slide behind it, pull."""
    bar, pull = ep.bar()
    for arm in arms:
        side = side_dir(pull) * (-1 if arm == 0 else 1)
        for deg in (15, 25, 5):
            a = approach_dir(pull, deg)
            tip_in = bar - pull * 0.021 + side * lateral - a * 0.019
            tip_hk = bar - pull * 0.021 + side * 0.015 - a * 0.019
            pts = (tip_in - a * pre, tip_in, tip_hk, tip_hk + pull * dist)
            for R in (frame(a, [0, 0, 1]), frame(a, side)):
                try:
                    for p in pts:
                        ep.solve(arm, p, R)
                except Unplanned:
                    continue
                ep.move(arm, pts[0], R, grip=0.0, speed=0.3)
                ep.move(arm, pts[1], R, speed=0.05)
                ep.move(arm, pts[2], R, speed=0.03)
                ep.move(arm, pts[3], R, speed=0.08)
                return
    raise Unplanned("no hook plan")


# ------------------------------------------------------------------------------------------------ behaviours
def B(kind, expect, fn):
    return dict(kind=kind, expect=expect, fn=fn)


def _arm(ep, start):
    return int(start["arm"]) if int(start["arm"]) >= 0 else pick_arm(ep)[0]


def open_reach():
    def solver(ep, st):
        grasp_handle(ep, pick_arm(ep), close=False)

    def wrong_handle(ep, st):
        nb = DRAWERS.index(NEIGHBOUR[DRAWERS[ep.k]])
        grasp_handle(ep, pick_arm(ep, nb), k=nb, close=False)

    def wrong_pull(ep, st):
        pull_neighbour(ep, pick_arm(ep), 0.12)

    return dict(solver=B("solver", "+1", solver), idle=B("hack", "0", lambda ep, st: ep.hold(1.0)),
                move_away=B("hack", "0", lambda ep, st: away(ep, pick_arm(ep)[0])),
                wrong_handle=B("hack", "0", wrong_handle), wrong_pull=B("hack", "-1:wrong drawer", wrong_pull))


def open_grasp():
    def solver(ep, st):
        arm = _arm(ep, st)
        p, R = ep.site(arm)
        bar, _ = ep.bar()
        ep.move(arm, bar - approach_of(R) * 0.004, R, speed=0.05)
        ep.grip(arm, 0.0, 0.7)

    def close_air(ep, st):
        arm = _arm(ep, st)
        p, R = ep.site(arm)
        bar, _ = ep.bar()
        ep.move(arm, bar - approach_of(R) * 0.04, R, speed=0.05)
        ep.grip(arm, 0.0, 0.7)

    def hook(ep, st):
        arm = _arm(ep, st)
        back_off(ep, arm, 0.03, grip=0.0)  # stay inside the 5 cm band: leaving it is "back to reach"
        hook_pull(ep, [arm], lateral=0.035, pre=0.0)

    return dict(solver=B("solver", "+1", solver), idle=B("hack", "0", lambda ep, st: ep.hold(1.0)),
                close_in_air=B("hack", "0", close_air), hook=B("alt", "+1 iff q>=2cm", hook))


def open_pull():
    def solver(ep, st):
        pull_by(ep, _arm(ep, st), 0.15)  # goal 12 cm; 15 cm leaves margin, 17 cm (overlay) is out of reach near the base

    def partial(ep, st):
        pull_by(ep, _arm(ep, st), 0.10)
        ep.hold(1.0)

    def let_go(ep, st):
        arm = _arm(ep, st)
        pull_by(ep, arm, 0.04)
        ep.grip(arm, 1.0, 0.4)
        away(ep, arm)

    def neighbour(ep, st):
        # the gripping arm holds still (a pre-pull slipped the grasp: "back to reach" before any wrong drawer)
        other_arm_opens_neighbour(ep, avoid=_arm(ep, st))

    return dict(solver=B("solver", "+1", solver), partial_hold=B("hack", "0", partial),
                let_go_early=B("hack", "-1:back to", let_go), wrong_drawer=B("hack", "-1:wrong drawer", neighbour))


def open_release():
    def solver(ep, st):
        arm = _arm(ep, st)
        ep.grip(arm, 1.0, 0.4)
        back_off(ep, arm, 0.10)

    def push_back(ep, st):
        arm = _arm(ep, st)
        ep.grip(arm, 1.0, 0.4)  # let go, then push in with the open hand (palm / finger roots on the bar)
        _, pull = ep.bar()
        p, R = ep.site(arm)
        # the bar first slides 4-5 cm between the open fingers to the palm, then the drawer moves
        ep.move(arm, p - pull * 0.15, pitched(R, side_dir(pull)), speed=0.08)

    return dict(solver=B("solver", "+1", solver), keep_holding=B("hack", "0", lambda ep, st: ep.hold(2.0)),
                push_back_in=B("hack", "-1:back to", push_back))


def close_reach():
    def solver(ep, st):
        fingertips_on_face(ep, pick_arm(ep))

    def grab_pull(ep, st):
        arm = grasp_handle(ep, pick_arm(ep))
        pull_by(ep, arm, 0.06)

    def neighbour(ep, st):
        pull_neighbour(ep, pick_arm(ep), 0.12)

    def far(ep, st):
        open_far_low(ep, pick_arm(ep, DRAWERS.index(FAR[DRAWERS[ep.k]])))

    # adjacent drawer: the route passes the open target's front (pinch 3.4-4.9 cm from its bar, touching it), which
    # is the reach goal state itself -> reported, not gated; the gated wrong-drawer hack uses the farthest drawer
    return dict(solver=B("solver", "+1", solver), idle=B("hack", "0", lambda ep, st: ep.hold(1.0)),
                move_away=B("hack", "0", lambda ep, st: away(ep, pick_arm(ep)[0])),
                grab_and_pull=B("alt", "+1", grab_pull), wrong_drawer=B("hack", "-1:wrong drawer", far),
                wrong_drawer_adjacent=B("info", "-1:wrong drawer", neighbour))


def close_push():
    def push(ep, arm, dist, speed=0.08):
        _, pull = ep.bar()
        p, R = ep.site(arm)
        ep.move(arm, p - pull * dist, pitched(R, side_dir(pull)), speed=speed)

    def solver(ep, st):
        arm = _arm(ep, st)
        push(ep, arm, float(ep.d.qpos[ep.scene.qpos_adr[ep.k]]) + 0.012)

    def partial(ep, st):
        arm = _arm(ep, st)
        push(ep, arm, 0.3 * float(ep.d.qpos[ep.scene.qpos_adr[ep.k]]))
        ep.hold(1.0)

    def push_pull(ep, st):
        arm = _arm(ep, st)
        push(ep, arm, 0.3 * float(ep.d.qpos[ep.scene.qpos_adr[ep.k]]))
        back_off(ep, arm, 0.04)
        arm = grasp_handle(ep, [arm, 1 - arm], pre=0.04)
        pull_by(ep, arm, 0.20)

    def neighbour(ep, st):
        # round 3: the other arm opens the FAR drawer by the low route (reaching the adjacent one knocked the open
        # target away from the pushing hand: "back to reach" before any wrong drawer)
        arm = _arm(ep, st)
        push(ep, arm, 0.3 * float(ep.d.qpos[ep.scene.qpos_adr[ep.k]]))
        open_far_low(ep, [1 - arm])

    return dict(solver=B("solver", "+1", solver), partial_hold=B("hack", "0", partial),
                push_then_pull_open=B("hack", "0|-1", push_pull), wrong_drawer=B("hack", "-1:wrong drawer", neighbour))


def close_release():
    def solver(ep, st):
        back_off(ep, _arm(ep, st), 0.10)

    def pull_open(ep, st):
        arm = _arm(ep, st)
        ep.grip(arm, 1.0, 0.4)  # open on the bar face, slide the open fingers over the bar, close, pull
        bar, _ = ep.bar()
        p, R = ep.site(arm)
        ep.move(arm, bar - approach_of(R) * 0.004, R, speed=0.05)
        ep.grip(arm, 0.0, 0.7)
        pull_by(ep, arm, 0.06)

    return dict(solver=B("solver", "+1", solver), stay=B("hack", "0", lambda ep, st: ep.hold(2.0)),
                pull_open_leaving=B("hack", "-1:back to", pull_open))


BEHAVIOURS = {("open", "reach"): open_reach, ("open", "grasp"): open_grasp, ("open", "pull"): open_pull,
              ("open", "release"): open_release, ("close", "reach"): close_reach, ("close", "push"): close_push,
              ("close", "release"): close_release}


def as_expected(expect, outcome, qmax, q0, failed=""):
    """EXPECT "-1:<reason>" also requires the failure reason to start with <reason> (a hack must fail for the reason
    it is meant to test)."""
    if expect == "+1 iff q>=2cm":
        return (outcome == "+1") == (qmax >= 0.02) and outcome != "-1"
    for e in expect.split("|"):
        want, _, why = e.partition(":")
        if outcome == want and (not why or failed.startswith(why)):
            return True
    return False


_ENV = None
SAVE_DIR: Path | None = None


def run_one(args):
    global _ENV, SAVE_DIR
    mode, stage, name, start, save = args
    SAVE_DIR = None if save is None else Path(save)
    if _ENV is None:
        _ENV = abc_sim.make_env(task=f"{TASK[mode]}__{stage}", render_cameras=False)
    beh = BEHAVIOURS[(mode, stage)]()[name]
    t0 = time.process_time()
    ep = Episode(_ENV, start)
    q0 = ep.qmax
    planned, why = True, ""
    try:
        beh["fn"](ep, start)
        ep.tail()
    except Finished:
        pass
    except Unplanned as e:
        planned, why = False, str(e)
    ev = ep.ev
    rec = dict(seed=int(start["seed"]), target=DRAWERS[ep.k], behaviour=name, kind=beh["kind"], expect=beh["expect"],
               planned=planned, why=why, outcome=ep.outcome if planned else "unplanned",
               return_=round(float(sum(ep.rewards)), 4), chunks=len(ep.rewards), failed=ev._last.get("failed", ""),
               label=ev._last.get("label"), q0=round(q0, 4), q_max=round(ep.qmax, 4), q_min=round(ep.qmin, 4),
               q_end=round(float(ev._last.get("q_m", 0.0)), 4), others_max=round(ep.others_max, 4),
               stock_success=bool(ev._last.get("stock_success", False)), cpu_s=round(time.process_time() - t0, 2))
    rec["as_expected"] = planned and as_expected(beh["expect"], ep.outcome, ep.qmax, q0, rec["failed"])
    if SAVE_DIR is not None:  # <save>/<stage>/<behaviour>_<seed>.npz; unplanned runs saved with no actions
        d = SAVE_DIR / stage
        d.mkdir(parents=True, exist_ok=True)
        acts = np.asarray(ep.actions, np.float32).reshape(-1, 14) if planned else np.zeros((0, 14), np.float32)
        np.savez_compressed(d / f"{name}_{int(start['seed'])}.npz", actions=acts,
                            rewards=np.asarray(ep.rewards if planned else [], np.float64))
    return rec


def load_starts(path, n, target=None):
    z = np.load(path, allow_pickle=True)
    rows = []
    for i in range(len(z["seed"])):
        if len(rows) >= n:
            break
        if target is not None and DRAWERS[int(z["target"][i])] != target:
            continue
        rows.append(dict(seed=int(z["seed"][i]), stage=str(z["stage"]), target=int(DRAWERS.index(z["target"][i]))
                         if isinstance(z["target"][i], str) else int(z["target"][i]), arm=int(z["arm"][i]),
                         state=np.asarray(z["state"][i], np.float64), rest_height=np.asarray(z["rest_height"][i]),
                         labels=np.asarray(z["labels"][i])))
    return rows


def summarize(records, names, kinds):
    summary = {}
    for name in names:
        rs = [r for r in records if r["behaviour"] == name]
        per_target = {d: dict(tried=0, plus=0, as_expected=0) for d in DRAWERS}
        for r in rs:
            t = per_target[r["target"]]
            t["tried"] += 1
            t["plus"] += r["outcome"] == "+1"
            t["as_expected"] += r["as_expected"]
        summary[name] = dict(kind=kinds[name], tried=len(rs), planned=sum(r["planned"] for r in rs),
                             plus=sum(r["outcome"] == "+1" for r in rs), minus=sum(r["outcome"] == "-1" for r in rs),
                             zero=sum(r["outcome"] == "0" for r in rs), as_expected=sum(r["as_expected"] for r in rs),
                             mean_return=round(float(np.mean([r["return_"] for r in rs])), 4), per_target=per_target)
    sol = summary["solver"]
    summary["G4"] = bool(sol["plus"] / sol["tried"] >= 0.4 and sol["plus"] / max(sol["planned"], 1) >= 0.7
                         and sol["tried"] >= 12)
    hacks = {k: v for k, v in summary.items() if isinstance(v, dict) and v.get("kind") == "hack"}
    alts = {k: v for k, v in summary.items() if isinstance(v, dict) and v.get("kind") == "alt"}
    if not hacks:  # solver-only run (e.g. per-target G4 check)
        summary["margin"], summary["G5"] = None, None
        return summary
    summary["margin"] = round(sol["mean_return"] - max(v["mean_return"] for v in hacks.values()), 4)
    # EXPECT share over the rollouts that ran (planned), and only if at least half of the starts could be planned;
    # +1 share over all starts. "info" behaviours are reported, not gated.
    ok = lambda v: v["planned"] >= v["tried"] / 2 and v["as_expected"] / max(v["planned"], 1) >= 0.8
    summary["G5"] = bool(summary["margin"] >= 0.5
                         and all(ok(h) and h["plus"] / h["tried"] <= 0.1 for h in hacks.values())
                         and all(ok(a) for a in alts.values()))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("open", "close"))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--save", type=Path, help="also save actions + chunk rewards per rollout here (replay videos), "
                    "<save>/<stage>/<behaviour>_<seed>.npz and <save>/summary_<stage>.json")
    ap.add_argument("--target", choices=DRAWERS, help="only starts whose target is this drawer (per-target runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print("CHECKOUT", abc_sim.__file__, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    starts = load_starts(args.starts / f"{args.mode}_{args.stage}.npz", args.n, args.target)
    behs = BEHAVIOURS[(args.mode, args.stage)]()
    names = [n for n in behs if not args.only or n in args.only]
    jobs = [(args.mode, args.stage, n, s, None if args.save is None else str(args.save)) for n in names for s in starts]
    records = []
    with ProcessPoolExecutor(max_workers=min(args.workers, 8)) as pool:
        for rec in pool.map(run_one, jobs):
            records.append(rec)
            print(json.dumps(rec), flush=True)
    summary = summarize(records, names, {n: behs[n]["kind"] for n in names}) if "solver" in names else {}
    tag = (f"{args.mode}_{args.stage}" + ("" if not args.only else "_" + "_".join(args.only))
           + ("" if not args.target else f"_{args.target}"))
    (args.out / f"summary_{tag}.json").write_text(json.dumps(dict(summary=summary, rows=records), indent=1))
    if args.save is not None:
        (args.save / f"summary_{args.stage}.json").write_text(json.dumps(dict(summary=summary, rows=records), indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
