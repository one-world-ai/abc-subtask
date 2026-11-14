"""Dish-rack subtask reward contract (G2): real simulator states, synthetic transitions. CPU, no renderer.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_dishrack_subtasks.py
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "disable")
import traceback

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.bottles_starts import idle_action
from abc_sim.task_eval.dishrack import DishrackEvaluator
from abc_sim.task_eval.dishrack_stages import C, DONE, I, P, RL, STAGES, StageConfig, classify
from abc_sim.task_eval.dishrack_starts import SEED_BASE, build_start
from abc_sim.task_eval.stagekit import ARMS, STATE_SPEC, arm_ids, axis_angle_quat, qmul
from abc_sim.task_specs import get_task_spec

TASK = "load_plates_into_dish_rack"
_STARTS: dict = {}


def env_for(stage="carry"):
    return abc_sim.make_env(task=f"{TASK}__{stage}", render_cameras=False)


SAVED = os.path.join(os.path.dirname(__file__), "..", "..", "simulation", "outputs", "subtask_rewards", "dishrack",
                     "starts")


def starts(stage, n=3, first=0):
    """The first ``n`` saved (G3-valid) starts of ``stage`` if the start sets exist, else the first ``n`` buildable
    ones from SEED_BASE + first (cached for the whole run)."""
    key = (stage, n, first)
    path = os.path.join(SAVED, f"{stage}.npz")
    if key not in _STARTS and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        _STARTS[key] = [dict(seed=int(z["seed"][i]), target=int(z["target"][i]), arm=int(z["arm"][i]), stage=stage,
                             state=z["state"][i], rest_height=z["rest_height"][i], labels=z["labels"][i])
                        for i in range(first, first + n)]
    if key not in _STARTS:
        env = env_for(stage)
        out, i = [], first
        while len(out) < n and i < first + 40:
            st, _ = build_start(env, SEED_BASE + i, stage)
            if st is not None:
                out.append(st)
            i += 1
        env.close()
        assert len(out) == n, (stage, "fewer buildable starts than needed", len(out))
        _STARTS[key] = out
    return _STARTS[key]


def restored(stage, i=0):
    env = env_for(stage)
    st = starts(stage)[i]
    env.reset_to_stage(st["seed"], st)
    return env, env._task_evaluator, st


def updates(env, n=6):
    for _ in range(n):
        env.evaluate_task()
    return env._task_evaluator.chunk_reward()


def test_lookup_preserves_stock():
    assert get_task_spec(TASK).evaluator_name == "dishrack_plates_in_rack"
    prompts = set()
    for stage in ("carry", "insert", "release"):
        spec = get_task_spec(f"{TASK}__{stage}")
        assert get_task_spec(spec.prompt).name == spec.name
        assert spec.evaluator_kwargs() == {"stage": stage}
        prompts.add(spec.prompt)
    assert len(prompts) == 3


def test_qpos_only_is_refused():
    env = env_for()
    env.reset(seed=SEED_BASE)
    try:
        env._task_evaluator.evaluate_qpos_batch(env.data.qpos[None])
    except RuntimeError as error:
        assert "real contacts and velocities" in str(error)
    else:
        raise AssertionError("qpos-only scoring must not manufacture zero velocities")
    env.close()


def test_stock_rule_matches_the_fixed_scorer():
    """The scene's per-plate stock_ok equals DishrackEvaluator's plate_upright_in_rack_mask (reset, held, seated,
    released states)."""
    checked = 0
    for stage in ("carry", "release"):
        env, ev, st = restored(stage)
        stock = DishrackEvaluator(model=env.model, spec=get_task_spec(TASK))
        action = idle_action(env)
        for phase in range(3):
            if phase == 2 and stage == "release":
                action[st["arm"] * 7 + 6] = 1.0   # open: the released plate must score in both
            for _ in range(40):
                env.step(action, render_obs=False)
            mujoco.mj_forward(env.model, env.data)
            obs = ev.scene.observe(env.data, ev.tracker.rest_height)
            ref = stock.evaluate_model_data(env.model, env.data).metrics["plate_upright_in_rack_mask"][0]
            assert list(obs["stock_ok"]) == list(ref), (stage, phase, obs["stock_ok"], ref)
            checked += 1
        env.close()
    assert checked == 6


def test_built_starts_classify_idle_and_restore_exactly():
    for stage in ("carry", "insert", "release"):
        env = env_for(stage)
        for st in starts(stage):
            env.reset_to_stage(st["seed"], st)
            assert env.evaluate_task().metrics["label"] == stage, (stage, st["seed"], env._task_evaluator._last)
            action = idle_action(env)
            total = 0.0
            for _ in range(10):
                for _ in range(32):
                    env.step(action, render_obs=False)
                reward, done = env._task_evaluator.chunk_reward()
                assert not done, (stage, st["seed"], env._task_evaluator._last)
                total += reward
            assert abs(total) < 0.05, (stage, st["seed"], total)
            runs = []
            for _ in range(2):
                env.reset_to_stage(st["seed"], st)
                for _ in range(32):
                    env.step(action, render_obs=False)
                s = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
                mujoco.mj_getState(env.model, env.data, s, STATE_SPEC)
                runs.append(s)
            assert np.array_equal(*runs), (stage, st["seed"], "restore is not exact")
        env.close()


def test_restore_reproduces_the_saved_state():
    for stage in ("carry", "release"):
        env, ev, st = restored(stage)
        s = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
        mujoco.mj_getState(env.model, env.data, s, STATE_SPEC)
        assert np.array_equal(s, st["state"])
        env.close()


def test_rack_moved_or_tipped_fails():
    for tip in (False, True):
        env, ev, _ = restored("carry")
        adr = ev.scene.rack_adr
        if tip:
            env.data.qpos[adr + 3:adr + 7] = qmul(axis_angle_quat(np.array([1.0, 0, 0]), np.deg2rad(15)),
                                                env.data.qpos[adr + 3:adr + 7].copy())
        else:
            env.data.qpos[adr] += 0.04
        mujoco.mj_forward(env.model, env.data)
        assert updates(env) == (-1.0, True)
        assert ev._failed == "rack moved or tipped"
        env.close()


def test_dropping_fails_in_carry_and_insert():
    for stage in ("carry", "insert"):
        env, ev, st = restored(stage)
        ids = arm_ids(env.model, ARMS[st["arm"]])
        action = idle_action(env)
        action[st["arm"] * 7 + 6] = 1.0
        for _ in range(64):
            env.step(action, render_obs=False)
            if ev._failed:
                break
        reward, done = ev.chunk_reward()
        assert (reward, done) == (-1.0, True), (stage, ev._last)
        assert ev._failed in ("dropped", "plate hit the table", "back to pick"), ev._failed
        env.close()


def test_hitting_the_table_outside_the_rack_fails():
    env, ev, st = restored("carry")
    k = st["target"]
    a = ev.scene.qpos_adr[k]
    env.data.qpos[a + 2] -= ev.tracker.obs["low"][k] + 0.002   # lowest point 2 mm into the table
    mujoco.mj_forward(env.model, env.data)
    obs = ev.scene.observe(env.data, ev.tracker.rest_height)
    assert obs["world_contact"][k] and not obs["over_fp"][k]
    assert ev.step_failure(k, obs, C) == "plate hit the table"
    env.close()


def test_falling_off_the_table_fails():
    env, ev, st = restored("carry")
    env.data.qpos[ev.scene.qpos_adr[st["target"]] + 2] -= 0.40
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True)
    # 40 cm down it also touches the floor (a world geom): "plate hit the table" may fire first; all are -1
    assert ev._failed in ("fell off the table", "dropped", "plate hit the table"), ev._failed
    env.close()


def test_knocking_a_racked_plate_out_fails():
    env, ev, st = restored("carry")
    obs = ev.tracker.obs
    k = st["target"]
    others = [j for j in range(len(ev.scene.names)) if j != k]
    if not others:
        others = [k]   # one-plate scene: the rule is per plate; exercise it on a synthetic racked flag
    ev._racked = np.zeros(len(ev.scene.names), bool)
    ev._racked[others[0]] = True
    obs = dict(obs)
    obs["in_rack_geom"] = obs["in_rack_geom"].copy()
    obs["in_rack_geom"][others[0]] = False
    why = ""
    for _ in range(StageConfig().hold):
        why = ev.step_failure(k, obs, C)
    assert why == "knocked a racked plate out"
    env.close()


def test_insert_flat_and_jam_fail():
    env, ev, st = restored("insert")
    k = st["target"]
    obs = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in ev.tracker.obs.items()}
    obs["nz"][k], obs["rack_contact"][k], obs["touched"][k] = 0.95, True, True
    why = ""
    for _ in range(StageConfig().flat_steps + 1):
        why = ev.step_failure(k, obs, I)
    assert why == "lying flat on the rack", why
    env.close()
    env, ev, st = restored("insert")
    obs = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in ev.tracker.obs.items()}
    obs["rack_contact"][k], obs["touched"][k], obs["in_rack_geom"][k] = True, True, False
    obs["rack_force_w"][k] = 1.0      # resting on a tine: not a jam
    why = ""
    for _ in range(StageConfig().jam_steps + 4):
        why = ev.step_failure(k, obs, I)
    assert why == "", why
    obs["rack_force_w"][k] = 3.0      # pressed onto it
    why = ""
    for _ in range(StageConfig().jam_steps):
        why = ev.step_failure(k, obs, I)
    assert why == "jammed on the rack", why
    env.close()


def test_release_pull_out_and_leave_fail():
    env, ev, st = restored("release")
    k = st["target"]
    obs = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in ev.tracker.obs.items()}
    obs["touched"][k] = True
    obs["low"][k] = ev._low0 + 0.025
    assert ev.step_failure(k, obs, RL) == "pulled out"
    env.close()
    env, ev, st = restored("release")
    obs = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in ev.tracker.obs.items()}
    obs["touched"][k] = False
    obs["in_rack_geom"][k] = False
    why = ""
    for _ in range(StageConfig().left_steps):
        why = ev.step_failure(k, obs, RL)
    assert why == "left the slot or fell over", why
    env.close()


def test_goal_tests_need_their_conditions():
    """Carry goal needs a grasp; a plate in the rack geometry but off every slot is not seated; DONE needs the jaws
    5 cm away and 64 still updates (tracker)."""
    env, ev, st = restored("release")
    k = st["target"]
    cfg = ev.cfg
    base = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in ev.tracker.obs.items()}
    m = len(ev.scene.names)
    o = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in base.items()}
    assert classify(o, np.full(m, RL), cfg)[k] == RL
    o["at_slot"][k] = o["at_slot_exit"][k] = False      # lean on the rack off every slot
    assert classify(o, np.full(m, I), cfg)[k] != RL
    o = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in base.items()}
    o["touched"][k] = o["grasped"][k] = False
    o["stock_ok"][k] = True
    o["speed"][k] = 0.0
    o["jaw_gap"][k] = 0.03
    assert classify(o, np.full(m, RL), cfg)[k] == RL     # released but jaws still near
    o["jaw_gap"][k] = 0.08
    assert classify(o, np.full(m, RL), cfg)[k] == DONE
    assert classify(o, np.full(m, I), cfg)[k] != DONE    # a plate that arrived without the insert goal (dropped in)
    o = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in base.items()}
    o["grasped"][k] = False
    o["aligned"][k] = o["aligned_exit"][k] = True
    assert classify(o, np.full(m, C), cfg)[k] != I
    tr = ev.tracker
    tr.committed[:] = RL
    tr.candidate[:] = DONE
    tr.count[:] = cfg.done_hold - 2
    raw = np.full(m, DONE)
    tr.count = np.where(raw == tr.candidate, tr.count + 1, 1)
    need = np.where(raw == DONE, cfg.done_hold, cfg.hold)
    assert not ((raw != tr.committed) & (tr.count >= need))[k]
    env.close()


def test_big_move_equals_eight_small_moves():
    env, ev, _ = restored("carry")
    initial = ev._phi_chunk
    totals = []
    for count in (1, 8):
        ev._phi_chunk = initial
        total = 0.0
        for value in np.linspace(initial, initial + 0.06, count + 1)[1:]:
            ev._last["progress"] = value
            reward, done = ev.chunk_reward()
            assert not done
            total += reward
        totals.append(total)
    assert abs(totals[0] - totals[1]) < 1e-6
    env.close()


def main():
    tests = [fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
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
