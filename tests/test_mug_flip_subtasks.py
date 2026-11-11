"""Mug-flip subtasks: sim-built starts, the subtask evaluator and its chunk reward, on scripted motions.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_mug_flip_subtasks.py
"""

from __future__ import annotations

import os
import traceback

os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402
from abc_sim.task_eval.mug_flip_stages import STAGES  # noqa: E402
from abc_sim.task_eval.mug_flip_starts import arm_ids, build_start, ik_link6, load_grasp_library, _T  # noqa: E402
from abc_sim.task_eval.mug_flip_subtask import MugFlipSubtaskEvaluator  # noqa: E402
from abc_sim.task_specs import maybe_get_task_spec  # noqa: E402

SEEDS = (900000001, 900000004, 900000007)
CHUNK = 32
LIB = load_grasp_library()


def env_for(stage: str):
    return abc_sim.make_env(task=f"turn_mug_right_side_up__{stage}", render_cameras=False)


def arm_q(env, arm: int) -> np.ndarray:
    return np.array(env.data.qpos[arm_ids(env.model, ("left", "right")[arm]).qadr])


def action(env, arm: int, q: np.ndarray, grip: float) -> np.ndarray:
    """``arm`` at q with gripper ``grip`` (1 open, 0 closed); the other arm where its current targets are."""
    act = np.zeros(14, np.float32)
    for i, ci in enumerate(env._ctrl_indices):
        act[i] = env.data.ctrl[ci] / 0.0475 if i in env._gripper_set else env.data.ctrl[ci]
    act[arm * 7:arm * 7 + 6], act[arm * 7 + 6] = q, grip
    return act


def run(env, acts) -> tuple[float, bool]:
    for a in acts:
        env.step(a, render_obs=False)
    return env._task_evaluator.chunk_reward()


def start(env, seed, stage):
    obs, info = env.reset_to_stage(seed)
    return info


# -- tests ---------------------------------------------------------------------------------------------------------

def test_lookup():
    assert maybe_get_task_spec("turn_mug_right_side_up").evaluator_name == "mug_flip_upright"
    for st in STAGES[:-1]:
        spec = maybe_get_task_spec(f"turn_mug_right_side_up__{st}")
        assert spec.evaluator_name == "mug_flip_subtask" and dict(spec.evaluator_options)["stage"] == st
        assert maybe_get_task_spec(spec.prompt).name == spec.name, "subtask prompt must look up its own task"


def test_starts_are_in_their_stage():
    for st in STAGES[:-1]:
        env = env_for(st)
        for seed in SEEDS:
            info = start(env, seed, st)
            r = env.evaluate_task()
            assert r.metrics["label"] == st, (st, seed, r.metrics)
            assert not r.metrics["terminal"], (st, seed, r.metrics)
            assert info["prompt"] == env.prompt
        env.close()


def test_hold_still_is_not_terminal():
    for st in ("rotate", "place"):
        env = env_for(st)
        for seed in SEEDS:
            info = start(env, seed, st)
            arm = info["arm"]
            rew, done = run(env, [action(env, arm, arm_q(env, arm), 0.0)] * CHUNK)
            m = env.evaluate_task().metrics
            assert not done and rew in (0.0, 1.0, -1.0), (st, seed, rew, m)
            assert m["grasp_arm"] in (arm, 2), (st, seed, m)
        env.close()


def test_opening_the_gripper_is_a_drop():
    for st in ("rotate", "place"):
        env = env_for(st)
        for seed in SEEDS:
            info = start(env, seed, st)
            arm = info["arm"]
            rew, done = run(env, [action(env, arm, arm_q(env, arm), 1.0)] * CHUNK)
            m = env.evaluate_task().metrics
            assert done and rew == -1.0 and m["failed"] == "dropped", (st, seed, rew, m)
        env.close()


def test_rotate_reward_follows_tilt():
    """Turning the wrist (joint 6, then 5) moves the held mug's tilt; the chunk reward's sign follows it."""
    env = env_for("rotate")
    checked = 0
    for seed in SEEDS:
        for joint in (5, 4):
            for direction in (+1, -1):
                info = start(env, seed, "rotate")
                arm = info["arm"]
                q0 = arm_q(env, arm)
                z0 = env.evaluate_task().metrics["upright_z"]
                acts = []
                for t in range(CHUNK):
                    q = q0.copy()
                    q[joint] += direction * 0.6 * (t + 1) / CHUNK
                    acts.append(action(env, arm, q, 0.0))
                rew, done = run(env, acts)
                m = env.evaluate_task().metrics
                dz = m["upright_z"] - z0
                if done:  # dropped or reached upright: the terminal reward must match
                    assert (rew == 1.0) == (m["failed"] == "" and m["label"] in ("place", "done")), (seed, m)
                    continue
                eps = env._task_evaluator.cfg.eps["rotate"]
                want = 1.0 if dz > eps else -1.0 if dz < -eps else 0.0
                assert rew == want, (seed, joint, direction, dz, rew)
                checked += 1
    env.close()
    assert checked >= 6, checked


def test_scripted_place_succeeds():
    """Lower the held upright mug to the table with IK, open: the place subtask succeeds (+1, terminal)."""
    env = env_for("place")
    ok = 0
    for seed in SEEDS:
        info = start(env, seed, "place")
        arm = info["arm"]
        ev = env._task_evaluator
        ids = arm_ids(env.model, ("left", "right")[arm])
        lift = env.evaluate_task().metrics["lift_m"]
        T6 = _T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])
        T_goal = T6.copy()
        T_goal[2, 3] -= lift - 0.004
        q1, pe, _ = ik_link6(env.model, env.data, ("left", "right")[arm], T_goal)
        assert q1 is not None and pe < 3e-3, (seed, pe)
        q0 = arm_q(env, arm)
        done = False
        for t in range(3 * CHUNK):  # lower over 2 chunks, hold
            q = q0 + (q1 - q0) * min(1.0, (t + 1) / (2 * CHUNK))
            env.step(action(env, arm, q, 0.0), render_obs=False)
            if (t + 1) % CHUNK == 0:
                rew, done = ev.chunk_reward()
                assert not done, (seed, t, ev._last)
        for t in range(2 * CHUNK):  # open and back off upward
            q = q1.copy()
            env.step(action(env, arm, q, 1.0), render_obs=False)
            if (t + 1) % CHUNK == 0:
                rew, done = ev.chunk_reward()
                if done:
                    break
        m = env.evaluate_task().metrics
        assert done and rew == 1.0, (seed, rew, m)
        ok += 1
    env.close()
    assert ok == len(SEEDS)


def test_restore_is_exact():
    """A saved start replays identically: same actions from the same restored start give the same states."""
    env = env_for("rotate")
    st, _ = build_start(env, SEEDS[0], "rotate", LIB)
    runs = []
    for _ in range(2):
        env.reset_to_stage(SEEDS[0], st)
        arm = st["arm"]
        q0 = arm_q(env, arm)
        for t in range(CHUNK):
            q = q0.copy()
            q[5] += 0.02 * t
            env.step(action(env, arm, q, 0.0), render_obs=False)
        runs.append(np.array(env.data.qpos))
    env.close()
    assert np.array_equal(runs[0], runs[1])


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
