"""Mug-rack subtask reward contract (gate G2): real simulator states and adversarial transitions. CPU, no renderer.

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_mug_rack_subtasks.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")
import sys
import traceback

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.mug_rack_stages import (A, DONE, I, P, R, RL, STAGE_PROMPTS, STAGES, UNTHREADABLE,
                                               StageConfig, classify)
from abc_sim.task_eval.mug_rack_starts import SEED_BASE, build_start, idle_action
from abc_sim.task_eval.stagekit import ARMS, arm_ids, axis_angle_quat, qmul
from abc_sim.task_specs import get_task_spec, list_task_specs

TRAINABLE = STAGES[:-1]
_STARTS: dict[str, list[dict]] = {}


def env_for(stage):
    return abc_sim.make_env(task=f"hang_mug_on_mug_rack__{stage}", render_cameras=False)


def starts(stage, n=3, max_seeds=12):
    """The first n starts built for ``stage`` on seeds SEED_BASE + 0.. (cached)."""
    if stage not in _STARTS:
        env = env_for(stage)
        out = []
        for i in range(max_seeds):
            s, _ = build_start(env, SEED_BASE + i, stage)
            if s is not None:
                out.append(s)
            if len(out) == n:
                break
        env.close()
        _STARTS[stage] = out
    assert len(_STARTS[stage]) >= 1, (stage, "no start built")
    return _STARTS[stage]


def chunk(env, action, n=32):
    for _ in range(n):
        env.step(action, render_obs=False)
    return env._task_evaluator.chunk_reward()


def updates(env, n=8):
    for _ in range(n):
        env.evaluate_task()
    return env._task_evaluator.chunk_reward()


# -- lookup ----------------------------------------------------------------------------------------------------------

def test_lookup_and_unique_prompts():
    assert get_task_spec("hang_mug_on_mug_rack").evaluator_name == "mug_on_rack"
    for stage in TRAINABLE:
        spec = get_task_spec(f"hang_mug_on_mug_rack__{stage}")
        assert get_task_spec(spec.prompt).name == spec.name
        assert spec.evaluator_kwargs() == {"stage": stage}
        assert spec.prompt == STAGE_PROMPTS[stage]
        assert "reset_to_stage" in spec.description
    prompts = [s.prompt.lower() for s in list_task_specs()]
    assert all(prompts.count(STAGE_PROMPTS[s].lower()) == 1 for s in TRAINABLE)


def test_unthreadable_mugs_are_never_targets():
    env = env_for("reach")
    seen = 0
    for i in range(12):
        env.forget_arm_state()
        try:
            env.reset(seed=SEED_BASE + i)
            env.evaluate_task()
        except RuntimeError as exc:
            ev = env._task_evaluator
            assert all(v in UNTHREADABLE for v in ev.scene.variants), (i, ev.scene.variants)
            assert "threadable" in str(exc)
            seen += 1
            continue
        ev = env._task_evaluator
        assert ev.scene.variants[ev.target] not in UNTHREADABLE
    env.close()
    assert seen >= 1, "expected a seed whose only mug is unthreadable in the first 12"


# -- (a) starts classify, (b) idle, (e) exact restore -------------------------------------------------------------------

def test_built_starts_classify_idle_and_restore_exactly():
    for stage in TRAINABLE:
        env = env_for(stage)
        for start in starts(stage):
            env.reset_to_stage(start["seed"], start)
            assert env.evaluate_task().metrics["label"] == stage, (stage, start["seed"])
            action = idle_action(env)
            total = 0.0
            for _ in range(10):
                reward, done = chunk(env, action)
                assert not done, (stage, start["seed"], env._task_evaluator._last)
                total += reward
            assert abs(total) < 0.05, (stage, start["seed"], total)
            runs = []
            for _ in range(2):
                env.reset_to_stage(start["seed"], start)
                ev = env._task_evaluator
                runs.append(np.concatenate([env.data.qpos.copy(), env.data.qvel.copy()]))
                for _ in range(32):
                    env.step(action, render_obs=False)
                runs.append(env.data.qpos.copy())
            assert np.array_equal(runs[0], runs[2]) and np.array_equal(runs[1], runs[3]), stage
        env.close()


# -- (c) every failure fires -1 ---------------------------------------------------------------------------------------

def test_fell_off_the_table():
    env = env_for("reach")
    env.reset_to_stage(starts("reach")[0]["seed"], starts("reach")[0])
    ev = env._task_evaluator
    env.data.qpos[ev.scene.qpos_adr[ev.target] + 2] -= 0.2
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True) and ev._failed == "fell off the table", ev._failed
    env.close()


def test_knocked_over_in_reach_and_pick():
    for stage in ("reach", "pick"):
        env = env_for(stage)
        s = starts(stage)[0]
        env.reset_to_stage(s["seed"], s)
        ev = env._task_evaluator
        a = ev.scene.qpos_adr[ev.target]
        if stage == "pick":  # move the open gripper away so no finger touches it
            ids = arm_ids(env.model, ARMS[s["arm"]])
            env.data.qpos[ids.qadr] = env.get_init_q()[s["arm"] * 7:s["arm"] * 7 + 6]
        env.data.qpos[a + 3:a + 7] = qmul(axis_angle_quat([1.0, 0, 0], np.pi / 2), env.data.qpos[a + 3:a + 7].copy())
        mujoco.mj_forward(env.model, env.data)
        assert updates(env) == (-1.0, True) and ev._failed == "knocked over", (stage, ev._failed)
        env.close()


def test_retreating_from_pick_regresses():
    env = env_for("pick")
    s = starts("pick")[0]
    env.reset_to_stage(s["seed"], s)
    ids = arm_ids(env.model, ARMS[s["arm"]])
    env.data.qpos[ids.qadr] = env.get_init_q()[s["arm"] * 7:s["arm"] * 7 + 6]
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True) and env._task_evaluator._failed == "back to reach"
    env.close()


def test_opening_in_align_and_insert_drops():
    for stage in ("align", "insert"):
        env = env_for(stage)
        s = starts(stage)[0]
        env.reset_to_stage(s["seed"], s)
        action = idle_action(env)
        action[s["arm"] * 7 + 6] = 1.0
        total, done = 0.0, False
        for _ in range(6):
            r, done = chunk(env, action)
            total += r
            if done:
                break
        ev = env._task_evaluator
        assert done and r == -1.0, (stage, ev._last)
        assert ev._failed in ("dropped", "back to pick", "back to reach", "fell off the table"), (stage, ev._failed)
        env.close()


def test_release_yank_off_the_peg_fails():
    env = env_for("release")
    s = starts("release")[0]
    env.reset_to_stage(s["seed"], s)
    ev = env._task_evaluator
    k = ev.target
    a = ev.scene.qpos_adr[k]
    peg = int(ev.tracker.obs["best_peg"][k])
    env.data.qpos[a:a + 3] += 0.08 * ev.scene.peg_axis[peg]  # hole 8 cm further out: off the peg
    ids = arm_ids(env.model, ARMS[s["arm"]])
    env.data.qpos[ids.qadr] = env.get_init_q()[s["arm"] * 7:s["arm"] * 7 + 6]
    mujoco.mj_forward(env.model, env.data)
    r, done = chunk(env, idle_action(env))
    for _ in range(3):
        if done:
            break
        r, done = chunk(env, idle_action(env))
    assert done and r == -1.0, ev._last
    env.close()


def test_release_onto_table_is_not_hanging():
    env = env_for("release")
    s = starts("release")[0]
    env.reset_to_stage(s["seed"], s)
    ev = env._task_evaluator
    obs, k = ev.tracker.obs, ev.target
    ev._untouched = ev.cfg.hold
    obs["touched"][k] = False
    obs["speed"][k] = 0.0
    obs["hanging"][k] = False
    assert ev.step_failure(k, obs, RL) == "released, not hanging"
    env.close()


def test_knocking_a_hanging_mug_off_fails():
    env = env_for("reach")
    s = starts("reach")[0]
    env.reset_to_stage(s["seed"], s)
    ev = env._task_evaluator
    others = [j for j in range(len(ev.scene.names)) if j != ev.target]
    if not others:
        s = next(x for x in starts("reach", n=3) if len(x["rest_height"]) > 1)
        env.reset_to_stage(s["seed"], s)
        ev = env._task_evaluator
        others = [j for j in range(len(ev.scene.names)) if j != ev.target]
    ev._hung[others[0]] = True  # pretend it was hanging at the start; it is on the table now
    assert updates(env) == (-1.0, True) and ev._failed == "knocked a hanging mug off", ev._failed
    env.close()


# -- classification rules on synthetic observations ------------------------------------------------------------------

def _obs(m=1, **kw):
    base = dict(dist=np.full(m, 0.05), touched=np.ones(m, bool), grasped=np.ones(m, bool),
                grasped_free=np.ones(m, bool), lift=np.full(m, 0.1), radial=np.full(m, 0.005),
                axial=np.full(m, 0.02), angle=np.full(m, 10.0), other_contact=np.zeros(m, bool),
                threaded=np.zeros(m, bool), depth=np.full(m, -1.0), speed=np.zeros(m), hanging=np.zeros(m, bool))
    base.update({k: np.asarray(v) for k, v in kw.items()})
    return base


def test_handle_grasp_is_not_a_pick_goal():
    cfg = StageConfig()
    o = _obs(grasped_free=[False], radial=[0.5])
    assert classify(o, np.array([P]), cfg)[0] == P


def test_aligned_needs_no_other_contact_and_hysteresis_holds_insert():
    cfg = StageConfig()
    assert classify(_obs(), np.array([A]), cfg)[0] == I
    assert classify(_obs(other_contact=[True]), np.array([A]), cfg)[0] == A
    # in insert, touching the peg and 2 cm off the axis still counts as lined up
    assert classify(_obs(other_contact=[True], radial=[0.02], axial=[-0.01]), np.array([I]), cfg)[0] == I
    assert classify(_obs(radial=[0.02]), np.array([A]), cfg)[0] == A


def test_threaded_deep_and_release_rules():
    cfg = StageConfig()
    deep = dict(threaded=[True], depth=[0.03])
    assert classify(_obs(**deep), np.array([I]), cfg)[0] == RL
    assert classify(_obs(threaded=[True], depth=[0.015]), np.array([I]), cfg)[0] == I  # not yet 2 cm deep
    assert classify(_obs(threaded=[True], depth=[0.015]), np.array([RL]), cfg)[0] == RL  # exit at 1 cm
    # let go, still swinging: stays in release; at rest and hanging: done; at rest not hanging: not done
    assert classify(_obs(**deep, touched=[False], grasped=[False], speed=[0.1]), np.array([RL]), cfg)[0] == RL
    assert classify(_obs(**deep, touched=[False], grasped=[False], hanging=[True]), np.array([RL]), cfg)[0] == DONE
    assert classify(_obs(**deep, touched=[False], grasped=[False], hanging=[False], dist=[0.3]),
                    np.array([RL]), cfg)[0] < RL
    # hanging but a finger still touches: not done
    assert classify(_obs(**deep, hanging=[True]), np.array([RL]), cfg)[0] == RL


def test_drop_onto_tree_never_done():
    """An un-threaded mug resting on the tree (peg in the cup) is never done, whatever the stock band says."""
    cfg = StageConfig()
    o = _obs(touched=[False], grasped=[False], hanging=[False], threaded=[False], dist=[0.3], lift=[0.15])
    assert classify(o, np.array([I]), cfg)[0] not in (RL, DONE)


# -- (d) potential-based shaping ---------------------------------------------------------------------------------------

def test_big_move_equals_eight_small_moves():
    for stage in TRAINABLE:
        env = env_for(stage)
        s = starts(stage)[0]
        env.reset_to_stage(s["seed"], s)
        ev = env._task_evaluator
        initial = ev._phi_chunk
        totals = []
        for count in (1, 8):
            ev._phi_chunk = initial
            total = 0.0
            for value in np.linspace(initial, initial + 0.03, count + 1)[1:]:
                ev._last["progress"] = value
                reward, done = ev.chunk_reward()
                assert not done
                total += reward
            totals.append(total)
        assert abs(totals[0] - totals[1]) < 1e-6, (stage, totals)
        env.close()


def main():
    only = sys.argv[1:]
    tests = [fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)
             and (not only or any(o in name for o in only))]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("ok", fn.__name__, flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
            print("FAIL", fn.__name__, flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed", flush=True)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
