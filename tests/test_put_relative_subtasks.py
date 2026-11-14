"""put_relative subtasks: sim-built starts, the subtask evaluator and its chunk reward, on scripted motions.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_put_relative_subtasks.py
"""

from __future__ import annotations

import os
import traceback

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402
from abc_sim.task_eval import put_relative_handoff as PH  # noqa: E402
from abc_sim.task_eval.put_relative_stages import (C, H, P, PL, R, STAGES, StageConfig, classify,  # noqa: E402
                                                   progress)
from abc_sim.task_eval.put_relative_starts import SEED_BASE, build_start  # noqa: E402
from abc_sim.task_eval.stagekit import T, arm_ids, ik_link6  # noqa: E402
from abc_sim.task_specs import maybe_get_task_spec  # noqa: E402

# Keep the original cases, including cross-table rejections. Add two validated same-side carry candidates:
# after the same-side restriction only offsets 3/5 of the original six can build carry, but a check needs four.
# Offsets 0/2/10 (cross-table) added with the handoff stage so it has three starts too; earlier stages already find
# theirs among the first eight.
SEEDS = tuple(SEED_BASE + i for i in (1, 3, 5, 7, 9, 11, 4, 8, 0, 2, 10))
N_SEEDS = 3  # per test: the first N_SEEDS of SEEDS that build a start of the stage (some seeds have none)
CHUNK = 32
ARMS = ("left", "right")


def env_for(stage: str):
    return abc_sim.make_env(task=f"put_relative__{stage}", render_cameras=False)


def arm_q(env, arm: int) -> np.ndarray:
    return np.array(env.data.qpos[arm_ids(env.model, ARMS[arm]).qadr])


def action(env, arm: int, q: np.ndarray, grip: float) -> np.ndarray:
    """``arm`` at q with gripper ``grip`` (1 open, 0 closed); the other arm at its current targets."""
    act = np.zeros(14, np.float32)
    for i, ci in enumerate(env._ctrl_indices):
        act[i] = env.data.ctrl[ci] / 0.0475 if i in env._gripper_set else env.data.ctrl[ci]
    act[arm * 7:arm * 7 + 6], act[arm * 7 + 6] = q, grip
    return act


def idle(env) -> np.ndarray:
    """Hold still: both arms at their current joint positions, grippers at their current commands. (Repeating the
    control targets is not idle right after a reset: they are still zero, so the arms would move.)"""
    act = np.zeros(14, np.float32)
    for i, ci in enumerate(env._ctrl_indices):
        act[i] = env.data.ctrl[ci] / 0.0475 if i in env._gripper_set else env.data.ctrl[ci]
    for a in (0, 1):
        act[a * 7:a * 7 + 6] = arm_q(env, a)
    return act


def starts(env, n: int = N_SEEDS):
    """Reset to the stage on the first ``n`` seeds that build; yields (seed, info)."""
    used = 0
    for seed in SEEDS:
        try:
            _, info = env.reset_to_stage(seed)
        except RuntimeError:
            continue
        yield seed, info
        used += 1
        if used == n:
            return
    raise AssertionError(f"only {used} of {len(SEEDS)} seeds build a start")


def run(env, acts) -> tuple[float, bool]:
    for a in acts:
        env.step(a, render_obs=False)
    return env._task_evaluator.chunk_reward()


def link6_goal(env, arm: int, dxyz) -> np.ndarray:
    ids = arm_ids(env.model, ARMS[arm])
    Tg = T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])
    Tg[:3, 3] += np.asarray(dxyz, dtype=np.float64)
    return Tg


def teleport(env, k: int, dxy) -> None:
    ev = env._task_evaluator
    a = int(ev.scene.qpos_adr[k])
    env.data.qpos[a:a + 2] += np.asarray(dxy, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


# -- tests ---------------------------------------------------------------------------------------------------------

def test_lookup():
    assert maybe_get_task_spec("put_relative").evaluator_name == "put_relative"
    for st in STAGES[:-1]:
        spec = maybe_get_task_spec(f"put_relative__{st}")
        assert spec.evaluator_name == "put_relative_subtask" and dict(spec.evaluator_options)["stage"] == st
        assert maybe_get_task_spec(spec.prompt).name == spec.name, "subtask prompt must look up its own task"


def test_starts_are_in_their_stage():
    for st in STAGES[:-1]:
        env = env_for(st)
        for seed, info in starts(env):
            m = env.evaluate_task().metrics
            assert m["label"] == st and not m["terminal"], (st, seed, m)
            assert info["prompt"] == env.prompt and env._task_evaluator.goal["mover_label"] in env.prompt, info
            assert not m["stock_success"], (st, seed, m)
        env.close()


def test_hold_still_is_not_terminal():
    for st in STAGES[:-1]:
        env = env_for(st)
        for seed, _ in starts(env):
            total = 0.0
            for _ in range(10):
                rew, done = run(env, [idle(env)] * CHUNK)
                assert not done, (st, seed, env._task_evaluator._last)
                total += rew
            assert abs(total) < 0.05, (st, seed, total)
        env.close()


def test_opening_the_gripper_mid_air_is_a_drop():
    checked = 0
    for st in ("carry", "place"):
        env = env_for(st)
        for seed, info in starts(env, 4):
            if env.evaluate_task().metrics["lift_m"] < 0.05:  # opening lower than drop height is a (rough) place
                continue
            arm = info["arm"]
            rew, done = run(env, [action(env, arm, arm_q(env, arm), 1.0)] * CHUNK)
            m = env.evaluate_task().metrics
            assert done and rew == -1.0 and m["failed"] == "dropped", (st, seed, rew, m)
            checked += 1
        env.close()
    assert checked >= 4, checked


def test_pushing_the_mover_is_a_failure():
    for st in ("reach", "pick"):
        env = env_for(st)
        for seed, _ in starts(env):
            ev = env._task_evaluator
            teleport(env, ev.target, [0.04, 0.0])  # moved 4 cm without being held
            rew, done = run(env, [idle(env)] * CHUNK)
            assert done and rew == -1.0 and ev._last["failed"] == "pushed", (st, seed, ev._last)
        env.close()


def test_pushing_into_the_zone_is_not_success():
    env = env_for("pick")
    for seed, _ in starts(env):
        ev = env._task_evaluator
        k, ref = ev.target, ev.scene.reference
        a, b = int(ev.scene.qpos_adr[k]), int(ev.scene.qpos_adr[ref])
        goal_xy = env.data.qpos[b:b + 2] + np.asarray(ev.goal["axis"]) * 0.14
        teleport(env, k, goal_xy - env.data.qpos[a:a + 2])
        rew, done = run(env, [idle(env)] * CHUNK)
        assert done and rew == -1.0 and ev._last["failed"] == "pushed", (seed, ev._last)
    env.close()


def test_moving_the_reference_is_a_failure():
    for st in ("reach", "carry"):
        env = env_for(st)
        for seed, _ in starts(env):
            ev = env._task_evaluator
            teleport(env, ev.scene.reference, [0.0, 0.05])
            rew, done = run(env, [idle(env)] * CHUNK)
            assert done and rew == -1.0 and ev._last["failed"] == "moved the reference", (st, seed, ev._last)
        env.close()


def test_carry_reward_follows_zone_gap():
    env = env_for("carry")
    checked = 0
    for seed, info in starts(env):
        ev, arm = env._task_evaluator, info["arm"]
        g0 = ev._last.get("zone_gap_m", env.evaluate_task().metrics["zone_gap_m"])
        q1, pe, _ = ik_link6(env.model, env.data, ARMS[arm], link6_goal(env, arm, [0.03, 0.02, 0.0]))
        assert q1 is not None and pe < 3e-3, (seed, pe)
        q0 = arm_q(env, arm)
        rew, done = run(env, [action(env, arm, q0 + (q1 - q0) * min(1, (t + 1) / 20), 0.0) for t in range(CHUNK)])
        m = env.evaluate_task().metrics
        if done:
            continue
        want = ev.shaping_total * (g0 - m["zone_gap_m"]) / ev.cfg.span["carry"]
        assert abs(rew - want) < 1e-6, (seed, rew, want)
        checked += 1
    env.close()
    assert checked >= 2, checked


def test_progress_reward_does_not_pay_for_dawdling():
    """Same carry start, same final gripper pose: one big move or eight small ones earn the same shaping total."""
    env = env_for("carry")
    totals = []
    for n in (1, 8):
        _, info = next(starts(env, 1))
        arm = info["arm"]
        q0 = arm_q(env, arm)
        q1, pe, _ = ik_link6(env.model, env.data, ARMS[arm], link6_goal(env, arm, [0.04, 0.0, 0.02]))
        assert q1 is not None and pe < 3e-3
        total = 0.0
        for c in range(8):
            q = q0 + (q1 - q0) * min(1.0, (c + 1) / n)
            rew, done = run(env, [action(env, arm, q, 0.0)] * CHUNK)
            assert not done, env._task_evaluator._last
            total += rew
        totals.append(total)
    env.close()
    assert abs(totals[0] - totals[1]) < 0.01, totals


def test_scripted_place_succeeds():
    """Lower the held mover to the table with IK, open, back off: place succeeds (+1) and the stock check agrees."""
    env = env_for("place")
    for seed, info in starts(env):
        arm, ev = info["arm"], env._task_evaluator
        lift = env.evaluate_task().metrics["lift_m"]
        q1, pe, _ = ik_link6(env.model, env.data, ARMS[arm], link6_goal(env, arm, [0, 0, -(lift - 0.006)]))
        assert q1 is not None and pe < 3e-3, (seed, pe)
        q0 = arm_q(env, arm)
        for t in range(2 * CHUNK):  # lower over 1.5 chunks
            env.step(action(env, arm, q0 + (q1 - q0) * min(1.0, (t + 1) / 48), 0.0), render_obs=False)
            if (t + 1) % CHUNK == 0:
                rew, done = ev.chunk_reward()
                assert not done, (seed, ev._last)
        q2, _, _ = ik_link6(env.model, env.data, ARMS[arm], link6_goal(env, arm, [0, 0, 0.08]))
        done = False
        for t in range(4 * CHUNK):  # open, then back off upward
            q = q1 if t < CHUNK else q1 + (q2 - q1) * min(1.0, (t - CHUNK + 1) / CHUNK)
            env.step(action(env, arm, q, 1.0), render_obs=False)
            if (t + 1) % CHUNK == 0:
                rew, done = ev.chunk_reward()
                if done:
                    break
        m = env.evaluate_task().metrics
        assert done and rew == 1.0 and m["stock_success"], (seed, rew, m)
    env.close()


def test_restore_is_exact():
    env = env_for("carry")
    st = next(s for s in (build_start(env, seed, "carry")[0] for seed in SEEDS) if s is not None)
    runs = []
    for _ in range(2):
        env.reset_to_stage(st["seed"], st)
        arm = st["arm"]
        q0 = arm_q(env, arm)
        for t in range(CHUNK):
            q = q0.copy()
            q[0] += 0.01 * t
            env.step(action(env, arm, q, 0.0), render_obs=False)
        runs.append(np.array(env.data.qpos))
    env.close()
    assert np.array_equal(runs[0], runs[1])


# -- handoff (cross-table episodes) ----------------------------------------------------------------------------------

def _obs(**kw):
    """Synthetic one-object cross-table observation: lifted 10 cm, held by G alone, far from the zone."""
    base = dict(cross=[True], lift=[0.10], dist=[0.0], touched=[True], grasped=[True], speed=[0.0], gap=[0.2],
                f_give=[2], f_recv=[0], touch_give=[True], touch_recv=[False], gap_give=[0.0], env_touch=[False],
                com=[[0.55, 0.0, 0.95]], mid_gap=[0.0])
    base.update(kw)
    out = {k: np.asarray(v) for k, v in base.items()}
    out["com"] = out["com"].reshape(1, 3)
    return out


def test_cross_table_labels():
    cfg = StageConfig()
    lab = lambda o, prev: int(classify(o, np.array([prev]), cfg)[0])  # noqa: E731
    assert lab(_obs(), P) == H  # the pick goal is a handoff state
    both = _obs(f_recv=[2], touch_recv=[True])
    assert lab(both, H) == H  # both hold: still handing
    let_go = _obs(f_give=[0], touch_give=[False], f_recv=[2], touch_recv=[True], gap_give=[0.03])
    assert lab(let_go, H) == H  # G open but within 6 cm
    recv = _obs(f_give=[0], touch_give=[False], f_recv=[2], touch_recv=[True], gap_give=[0.08])
    assert lab(recv, H) == C
    assert lab(_obs(**{**{k: v for k, v in recv.items()}, "env_touch": [True]}), H) == H  # not in the air
    assert lab(_obs(f_give=[0], touch_give=[False], f_recv=[2], touch_recv=[True], gap_give=[0.08], gap=[0.0]),
               C) == PL
    # G holding it over the zone is not place (and is a failure in handoff, below)
    assert lab(_obs(gap=[0.0]), H) == H
    # same-side observations take the unchanged path: no handoff label
    same = {k: v for k, v in _obs().items() if k in ("lift", "dist", "touched", "grasped", "speed", "gap")}
    assert lab(same, P) == C
    # handoff progress: G holds (0) -> V closes (0.5) -> G opens (1.0) -> G 6 cm clear (1.25), + 0.25 in the middle
    ps = [float(progress(H, o, cfg)[0]) for o in (_obs(), both, _obs(f_give=[0], touch_give=[False], f_recv=[2],
                                                                     touch_recv=[True]), recv)]
    assert np.allclose(ps, [0.25, 0.75, 1.25, 1.5]), ps


def test_handoff_failures():
    env = env_for("handoff")
    checked = 0
    for seed, info in starts(env, 2):
        ev = env._task_evaluator
        G = info["arm"]
        rew, done = run(env, [action(env, G, arm_q(env, G), 1.0)] * CHUNK)  # G lets go, V is not there
        assert done and rew == -1.0 and ev._last["failed"] == "dropped during the handoff", (seed, ev._last)
        checked += 1
        env.reset_to_stage(seed)
        ev = env._task_evaluator
        k = ev.target
        a = int(ev.scene.qpos_adr[k])
        o = ev.tracker.obs
        # synthetic: G holding it over the zone
        why = ev.step_failure(k, {**o, "gap": np.zeros_like(o["gap"])}, H)
        assert why == "giver carried it to the goal", why
        why = ev.step_failure(k, {**o, "env_touch": np.ones_like(o["env_touch"])}, H)
        assert why == "touched the table during the handoff", why
        del a
    env.close()
    assert checked == 2


def test_cross_carry_giver_touch_is_failure():
    env = env_for("carry")
    checked = 0
    for seed, info in starts(env, 4):
        ev = env._task_evaluator
        o = ev.tracker.obs
        if not bool(o["cross"][ev.target]):
            continue
        why = ev.step_failure(ev.target, {**o, "touch_give": np.ones_like(o["touch_give"])}, C)
        assert why == "giver touched the mover again", why
        m = env.evaluate_task().metrics
        assert m["f_recv"] == 2 and m["f_give"] == 0 and m["gap_give_m"] >= 0.06, m
        checked += 1
    env.close()
    assert checked >= 1, checked


def test_scripted_handoff_succeeds():
    """The scripted handoff from a handoff start: +1 (received in the air), shaping positive."""
    env = env_for("handoff")
    seed, info = next(starts(env, 1))
    ev = env._task_evaluator
    G = info["arm"]
    view = PH.MoverView(env.model, ev.scene.names[ev.target], ev.scene.joints[ev.target])
    rewards = []
    step = env.step

    def traced(a, **kw):
        res = step(a, **kw)
        traced.n += 1
        if traced.n % CHUNK == 0 and not traced.done:
            r, traced.done = ev.chunk_reward()
            rewards.append(r)
        return res
    traced.n, traced.done = 0, False
    env.step = traced
    ok, why = PH.handoff_solver(env, view, G, avoid=PH.zone_avoider(ev.scene, ev.tracker.rest_height))
    assert ok, why
    env.step = step
    if not traced.done:
        rewards.append(run(env, [idle(env)] * CHUNK)[0])
    assert ev._success and not ev._failed and rewards[-1] == 1.0, (seed, rewards, ev._last)
    assert sum(rewards[:-1]) > 0, rewards


def main() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
