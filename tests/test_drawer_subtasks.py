"""open_drawer / close_drawer subtasks: sim-built starts, the subtask evaluators and their chunk reward (gate G2).

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_drawer_subtasks.py

(a) each built start classifies as its stage; (b) idle 10 chunks: no terminal, |return| < 0.05; (c) every listed
failure gives -1; (d) one big move = 8 small moves (shaping totals equal to 1e-6); (e) snapshot restore is exact.
"""

from __future__ import annotations

import os
import traceback

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402
from abc_sim.task_eval.drawer_stages import (CLOSE_LIVE, CLOSE_PROMPTS, CLOSE_STAGES, DRAWERS, OPEN_LIVE,  # noqa: E402
                                             OPEN_PROMPTS, OPEN_STAGES)
from abc_sim.task_eval.drawer_starts import (SEED_BASE, arm_action, handle_pose, idle_action, ik_pinch,  # noqa: E402
                                             set_drawer)
from abc_sim.task_eval.stagekit import ARMS, STATE_SPEC, arm_ids  # noqa: E402
from abc_sim.task_specs import list_task_specs, maybe_get_task_spec  # noqa: E402

CHUNK = 32
N_SEEDS = 3
TASKS = {"open": ("open_drawer", OPEN_STAGES[:-1]), "close": ("close_drawer", CLOSE_STAGES[:-1])}
_ENVS: dict = {}
_STARTS: dict = {}


def env_for(mode: str, stage: str):
    key = (mode, stage)
    if key not in _ENVS:
        _ENVS[key] = abc_sim.make_env(task=f"{TASKS[mode][0]}__{stage}", render_cameras=False)
    return _ENVS[key]


def starts(mode: str, stage: str, n: int = N_SEEDS):
    """The first n seeds of the stage that build (cached), as (seed, start)."""
    key = (mode, stage)
    if key not in _STARTS:
        env, out = env_for(mode, stage), []
        for i in range(12):
            start, _ = env._task_evaluator.build_start(env, SEED_BASE[mode] + i)
            if start is not None:
                out.append((SEED_BASE[mode] + i, start))
            if len(out) == n:
                break
        assert len(out) == n, f"{mode} {stage}: only {len(out)} starts built"
        _STARTS[key] = out
    return _STARTS[key]


def restore(mode, stage, start):
    env = env_for(mode, stage)
    env.reset_to_stage(start["seed"], start)
    return env, env._task_evaluator


def run_chunk(env, act, n: int = CHUNK):
    for _ in range(n):
        env.step(act, render_obs=False)
    return env._task_evaluator.chunk_reward()


def run(env, acts) -> tuple[float, bool]:
    for a in acts:
        env.step(a, render_obs=False)
    return env._task_evaluator.chunk_reward()


def arm_q(env, arm):
    return np.array(env.data.qpos[arm_ids(env.model, ARMS[arm]).qadr])


def path(env, arm, q_goal, n, grip):
    """n actions moving ``arm`` linearly in joint space from its current joints to q_goal."""
    q0 = arm_q(env, arm)
    return [arm_action(env, arm, q0 + (q_goal - q0) * (i + 1) / n, grip) for i in range(n)]


def retreat_q(env, arm, start_q, dist=0.12):
    """Joints that move the pinch back along the approach by ``dist`` (straight back out of the handle)."""
    from abc_sim.task_eval.stagekit import T, ik_link6, actuator_limits
    ids = arm_ids(env.model, ARMS[arm])
    Tg = T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])
    Tg[:3, 3] -= Tg[:3, :3][:, 2] * dist  # link_6 z = approach axis (grasp site frame shares it)
    q, pe, _ = ik_link6(env.model, env.data, ARMS[arm], Tg, limits=actuator_limits(env.model, ARMS[arm]))
    assert pe < 0.01, pe
    return q


def terminal_reward(env, acts_fn, chunks=10):
    total, r, done = 0.0, 0.0, False
    for _ in range(chunks):
        r, done = run_chunk(env, acts_fn(env))
        total += r
        if done:
            break
    return r, done, total


# ---------------------------------------------------------------------------------------------------- tests
def test_prompts_unique():
    """Spec prompts and every live prompt resolve to no other task (prompts are lookup keys)."""
    names = {s.name for s in list_task_specs()}
    for mode, (task, stages) in TASKS.items():
        fixed, live = (OPEN_PROMPTS, OPEN_LIVE) if mode == "open" else (CLOSE_PROMPTS, CLOSE_LIVE)
        for stage in stages:
            spec = maybe_get_task_spec(f"{task}__{stage}")
            assert spec is not None and spec.prompt == fixed[stage], (task, stage)
            assert maybe_get_task_spec(fixed[stage]).name == spec.name
            assert "env.reset_to_stage" in spec.description
            for d in DRAWERS:
                other = maybe_get_task_spec(live[stage].format(target=d))
                assert other is None or other.name == spec.name, (live[stage], other and other.name)
    assert all(f"{t}__{s}" in names for t, ss in TASKS.values() for s in ss)


def test_a_starts_classify():
    for mode, (_, stages) in TASKS.items():
        for stage in stages:
            for seed, start in starts(mode, stage):
                env, ev = restore(mode, stage, start)
                assert ev.LABELS[int(ev.tracker.committed[ev.target])] == stage, (mode, stage, seed)
                r = env.evaluate_task()
                assert ev._last["label"] == stage and not ev._last["terminal"], (mode, stage, seed, ev._last)
                assert env.prompt == ev.stage_prompt() and DRAWERS[ev.target] in env.prompt


def test_b_idle():
    for mode, (_, stages) in TASKS.items():
        for stage in stages:
            for seed, start in starts(mode, stage):
                env, ev = restore(mode, stage, start)
                total = 0.0
                for _ in range(10):
                    r, done = run_chunk(env, idle_action(env))
                    total += r
                    assert not done, (mode, stage, seed, ev._last)
                assert abs(total) < 0.05, (mode, stage, seed, total)


def _other(ev):
    return next(j for j in range(3) if j != ev.target)


def test_c_failures():
    # wrong drawer: another drawer pulled to 6 cm (synthetic), every stage of both tasks
    for mode, (_, stages) in TASKS.items():
        for stage in stages:
            seed, start = starts(mode, stage)[0]
            env, ev = restore(mode, stage, start)
            set_drawer(env, ev.scene, _other(ev), 0.06)
            r, done = run_chunk(env, idle_action(env))
            assert (r, done) == (-1.0, True) and ev._failed == "wrong drawer opened", (mode, stage, ev._last)
    # close: target pulled > 3 cm past its start opening (reach, push)
    for stage in ("reach", "push"):
        seed, start = starts("close", stage)[0]
        env, ev = restore("close", stage, start)
        set_drawer(env, ev.scene, ev.target, min(ev.start_opening()[ev.target] + 0.035, 0.215))
        r, done = run_chunk(env, idle_action(env))
        assert (r, done) == (-1.0, True) and ev._failed == "pulled further open", (stage, ev._last)
    # open grasp / pull, close push: the hand leaves the handle (open, back off 12 cm) -> back to an earlier stage
    for mode, stage in (("open", "grasp"), ("open", "pull"), ("close", "push")):
        for seed, start in starts(mode, stage):
            env, ev = restore(mode, stage, start)
            arm = int(start["arm"])
            q = retreat_q(env, arm, arm_q(env, arm))
            r, done = run(env, path(env, arm, q, 3 * CHUNK, 1.0))
            if not done:
                r, done, _ = terminal_reward(env, idle_action, chunks=3)
            assert (r, done) == (-1.0, True) and ev._failed.startswith("back to"), (mode, stage, seed, ev._last)
    # release stages: drawer moved back across the hysteresis band (open < 11 cm, close > 2.5 cm), synthetic
    for mode, q in (("open", 0.10), ("close", 0.03)):
        seed, start = starts(mode, "release")[0]
        env, ev = restore(mode, "release", start)
        set_drawer(env, ev.scene, ev.target, q)
        r, done, _ = terminal_reward(env, idle_action, chunks=2)
        assert (r, done) == (-1.0, True) and ev._failed.startswith("back to"), (mode, ev._last)


def _reach_goal_q(env, ev, mode, frac=0.5):
    """Joints putting the pinch ``frac`` of the way from where it is to a handle pose (Cartesian)."""
    rng = np.random.default_rng(0)
    near = int(ev.tracker.obs["near_arm"][ev.target])  # the reach distance is the nearer arm's
    for f in (frac, 0.7, 0.3):
        here = ev.scene.pinch(env.data)[near]
        for pinch, R, grip in handle_pose(ev.scene, env.data, ev.target, mode, rng, jitter=False):
            q, pe, re = ik_pinch(env.model, env.data, near, here + f * (pinch - here), R, rng=rng)
            if pe < 3e-3 and re < 0.05:
                return near, q, grip
    raise AssertionError("no IK to the handle")


def test_d_potential():
    """Moving towards the handle (reach) and pulling the drawer by its handle (pull): the shaping total of one
    8-chunk reward call equals the sum of 8 per-chunk calls."""
    for mode, stage in (("open", "reach"), ("close", "reach"), ("open", "pull")):
        seed, start = starts(mode, stage)[0]
        totals = []
        for per_chunk in (False, True):
            env, ev = restore(mode, stage, start)
            if stage == "reach":
                arm, q, grip = _reach_goal_q(env, ev, mode)
                acts = path(env, arm, q, 8 * CHUNK, grip)
            else:
                arm = int(start["arm"])
                q = retreat_q(env, arm, arm_q(env, arm), dist=0.06)
                acts = path(env, arm, q, 8 * CHUNK, 0.0)
            total = 0.0
            for c in range(8):
                for a in acts[c * CHUNK:(c + 1) * CHUNK]:
                    env.step(a, render_obs=False)
                if per_chunk:
                    r, done = ev.chunk_reward()
                    total += r
                    assert not done, (mode, stage, ev._last)
            if not per_chunk:
                r, done = ev.chunk_reward()
                assert not done, (mode, stage, ev._last)
                total = r
            totals.append(total)
        assert abs(totals[0] - totals[1]) < 1e-6 and abs(totals[0]) > 0.01, (mode, stage, totals)


def test_e_restore_exact():
    for mode, (_, stages) in TASKS.items():
        for stage in stages:
            seed, start = starts(mode, stage)[0]
            env, ev = restore(mode, stage, start)
            s = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
            mujoco.mj_getState(env.model, env.data, s, STATE_SPEC)
            assert np.array_equal(s, start["state"]), (mode, stage)
            ends = []
            for _ in range(2):
                env, ev = restore(mode, stage, start)
                act = idle_action(env)
                rs = [run_chunk(env, act)[0] for _ in range(2)]
                mujoco.mj_getState(env.model, env.data, s, STATE_SPEC)
                ends.append((s.copy(), rs))
            assert np.array_equal(ends[0][0], ends[1][0]) and ends[0][1] == ends[1][1], (mode, stage)


if __name__ == "__main__":
    print("CHECKOUT", abc_sim.__file__)
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name, flush=True)
            except Exception:
                fails += 1
                print("FAIL", name, flush=True)
                traceback.print_exc()
    print(f"{fails} failed")
    raise SystemExit(1 if fails else 0)
