"""Sweep hand-pick chain H reward contract: real simulator measurements and adversarial transitions. CPU, no renderer.

Plain script: ``MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_sweep_subtasks.py``.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "disable")
import traceback
import mujoco
import numpy as np
import abc_sim
from abc_sim.task_eval.sweep_stages import D, DONE, STAGES, StageConfig, classify
from abc_sim.task_eval.sweep_starts import SEED_BASE, build_start, idle_action
from abc_sim.task_eval.stagekit import ARMS, arm_ids, axis_angle_quat, qmul, set_object_pose
from abc_sim.task_specs import get_task_spec


def env_for(stage="reach", seed=SEED_BASE):
    env = abc_sim.make_env(task=f"sweep_away_paper_scraps_from_table__{stage}", render_cameras=False)
    env.reset(seed=seed)
    env.evaluate_task()
    return env


def updates(env, n=6):
    for _ in range(n):
        env.evaluate_task()
    return env._task_evaluator.chunk_reward()


def place_in_bin(env, k, height=0.04):
    ev, data = env._task_evaluator, env.data
    scene = ev.scene
    adr = scene.qpos_adr[k]
    bin_pos = data.qpos[scene.bin_adr:scene.bin_adr + 3]
    pos = bin_pos + data.xmat[scene.bin_body].reshape(3, 3) @ [0, scene.geometry._bin_bottom_y + height, 0]
    offset = data.qpos[adr:adr + 3] - data.subtree_com[scene.bodies[k]]
    set_object_pose(env, scene, k, pos + offset, data.qpos[adr + 3:adr + 7].copy())


def test_lookup_preserves_stock():
    assert get_task_spec("sweep_away_paper_scraps_from_table").evaluator_name == "sweep_away"
    for stage in STAGES[:-1]:
        spec = get_task_spec(f"sweep_away_paper_scraps_from_table__{stage}")
        assert get_task_spec(spec.prompt).name == spec.name
        assert spec.evaluator_kwargs() == {"stage": stage}


def test_active_set_comes_from_randomizer():
    for seed in (SEED_BASE + 3, SEED_BASE):  # 3 and 4 active scraps
        env = env_for(seed=seed)
        ev = env._task_evaluator
        joints = env._last_randomization.metadata["trash_joints"]
        active = ev.tracker.obs["active"]
        assert active.sum() == len(joints), (active, joints)
        assert active[ev.target]
        assert not ev.tracker.obs["off_table"].any(), "parked inactive scraps must not count as off the table"
        assert ev.stock._fixed_active_trash_mask and np.array_equal(ev.stock._active_trash_mask[0], active)
        env.close()


def test_qpos_only_cannot_claim_a_settled_goal():
    env = env_for()
    try:
        env._task_evaluator.evaluate_qpos_batch(env.data.qpos[None])
    except RuntimeError as error:
        assert "real contacts and velocities" in str(error)
    else:
        raise AssertionError("qpos-only scoring must not manufacture zero velocities")
    env.close()


def test_bin_motion_and_tip_fail_before_success():
    for case in ("move", "tilt", "tip"):
        env = env_for()
        adr = env._task_evaluator.scene.bin_adr
        if case == "move":
            env.data.qpos[adr] += 0.06
        else:
            angle = np.deg2rad(25) if case == "tilt" else np.pi / 2  # 25 deg: past the stage's 20, inside the scorer's 45
            env.data.qpos[adr + 3:adr + 7] = qmul(axis_angle_quat(np.array([1., 0, 0]), angle),
                                               env.data.qpos[adr + 3:adr + 7].copy())
        mujoco.mj_forward(env.model, env.data)
        assert updates(env) == (-1.0, True), case
        assert env._task_evaluator._failed == "bin moved or tipped"
        env.close()


def test_push_fails():
    for stage in ("reach", "pick"):
        env = env_for(stage)
        ev = env._task_evaluator
        ev.begin(env.model, env.data, ev.target, labels=np.full(len(ev.scene.names), STAGES.index(stage)))
        adr = ev.scene.qpos_adr[ev.target]
        env.data.qpos[adr] += 0.06
        mujoco.mj_forward(env.model, env.data)
        assert updates(env) == (-1.0, True)
        assert ev._failed == "pushed"
        env.close()


def test_target_off_the_table_fails():
    env = env_for()
    ev = env._task_evaluator
    env.data.qpos[ev.scene.qpos_adr[ev.target] + 2] -= 0.20
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True)
    assert ev._failed == "fell off the table"
    env.close()


def test_other_scrap_off_the_table_or_on_the_floor_fails():
    for drop in (0.20, 0.70):  # off the table edge; on the floor
        env = env_for()
        ev = env._task_evaluator
        other = next(i for i in np.flatnonzero(ev.tracker.obs["active"]) if i != ev.target)
        adr = ev.scene.qpos_adr[other]
        env.data.qpos[adr] += 0.5
        env.data.qpos[adr + 2] -= drop
        mujoco.mj_forward(env.model, env.data)
        assert updates(env) == (-1.0, True)
        assert ev._failed == "a scrap fell off the table", ev._failed
        env.close()


def test_knocking_a_previously_binned_scrap_out_fails():
    env = env_for()
    ev, data = env._task_evaluator, env.data
    scene, target = ev.scene, ev.target
    other = next(i for i in np.flatnonzero(ev.tracker.obs["active"]) if i != target)
    place_in_bin(env, other)
    ev.begin(env.model, data, target)
    assert ev._binned[other]
    adr = scene.qpos_adr[other]
    data.qpos[adr] -= 0.25
    data.qpos[adr + 2] = 0.76
    mujoco.mj_forward(env.model, data)
    assert updates(env) == (-1.0, True)
    assert ev._failed == "knocked a binned scrap out"
    env.close()


def test_inactive_scrap_is_never_a_target():
    env = env_for(seed=SEED_BASE + 3)
    ev = env._task_evaluator
    obs = dict(ev.tracker.obs)
    inactive = np.flatnonzero(~obs["active"])
    assert len(inactive)
    obs["dist"] = np.where(obs["active"], 1.0, 0.0)
    assert obs["active"][ev.pick_target(obs)]
    try:
        ev.begin(env.model, env.data, int(inactive[0]))
    except ValueError:
        pass
    else:
        raise AssertionError("an inactive scrap was accepted as a target")
    env.close()


def test_retreating_from_pick_regresses():
    env = env_for("pick")
    start, _ = build_start(env, SEED_BASE, "pick")
    assert start is not None
    env.reset_to_stage(start["seed"], start)
    arm = start["arm"]
    ids = arm_ids(env.model, ARMS[arm])
    env.data.qpos[ids.qadr] = env.get_init_q()[arm * 7:arm * 7 + 6]
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True)
    assert env._task_evaluator._failed == "back to reach"
    env.close()


def test_floating_inside_stock_region_is_not_settled():
    env = env_for()
    ev, data = env._task_evaluator, env.data
    scene, k = ev.scene, ev.target
    rest = scene.heights(data)
    place_in_bin(env, k, height=scene.geometry._bin_top_y - scene.geometry._bin_bottom_y - 0.03)
    obs = scene.observe(data, rest)
    assert obs["in_bin"][k] and not obs["supported"][k]
    assert classify(obs, np.full(len(scene.names), D), StageConfig())[k] != DONE
    env.close()


def test_scrap_against_outside_of_bin_base_is_not_done():
    """The phase-0 false positive: a scrap on the table touching the bin's outside wall."""
    env = env_for()
    ev, data = env._task_evaluator, env.data
    scene, k = ev.scene, ev.target
    rest = scene.heights(data)
    bin_pos = data.qpos[scene.bin_adr:scene.bin_adr + 3]
    adr = scene.qpos_adr[k]
    pos = data.qpos[adr:adr + 3].copy()
    pos[:2] = bin_pos[:2] + [0.088, 0]
    set_object_pose(env, scene, k, pos, data.qpos[adr + 3:adr + 7].copy())
    for _ in range(200):
        mujoco.mj_step(env.model, data)
    obs = scene.observe(data, rest)
    assert not obs["in_bin"][k]
    assert classify(obs, np.full(len(scene.names), D), StageConfig())[k] != DONE
    env.close()


def test_big_move_equals_eight_small_moves():
    env = env_for()
    ev = env._task_evaluator
    initial = ev._phi_chunk
    totals = []
    for count in (1, 8):
        ev._phi_chunk = initial
        total = 0
        for value in np.linspace(initial, initial + 0.06, count + 1)[1:]:
            ev._last["progress"] = value
            reward, done = ev.chunk_reward()
            assert not done
            total += reward
        totals.append(total)
    assert abs(totals[0] - totals[1]) < 1e-6
    env.close()


def test_every_release_is_validated_even_after_regrasp():
    env = env_for("drop")
    ev = env._task_evaluator
    obs = ev.tracker.obs
    k = ev.target
    ev._released = True
    ev._was_touched = True
    obs["touched"][k] = False
    obs["over"][k] = False
    assert ev.step_failure(k, obs, D) == "released outside the opening"
    env.close()


def test_unsupported_slow_flight_is_not_a_landing():
    env = env_for("drop")
    ev = env._task_evaluator
    obs, k = ev.tracker.obs, ev.target
    ev._released, ev._was_touched = True, False
    obs["touched"][k] = False
    obs["speed"][k] = 0
    obs["angular_speed"][k] = 0
    obs["resting_contact"][k] = False
    obs["in_bin"][k] = False
    assert ev.step_failure(k, obs, D) == ""
    assert classify(obs, np.full(len(ev.scene.names), D), ev.cfg)[k] == D
    env.close()


def test_safe_lowering_and_release_inside_bin_are_allowed():
    env = env_for("drop")
    ev = env._task_evaluator
    obs, k = ev.tracker.obs, ev.target
    obs["in_bin"][k], obs["over"][k] = True, False
    obs["above_rim"][k] = -0.05
    obs["touched"][k], obs["grasped"][k] = True, True
    assert classify(obs, np.full(len(ev.scene.names), D), ev.cfg)[k] == D
    ev._was_touched = True
    obs["touched"][k] = False
    assert ev.step_failure(k, obs, D) == ""
    env.close()


def test_dropped_into_bin_from_above_reaches_done():
    """Released over the opening, the scrap falls in, settles on the bin floor and the stage ends +1 (= stock in)."""
    env = env_for("drop")
    start, _ = build_start(env, SEED_BASE, "drop")
    assert start is not None
    env.reset_to_stage(start["seed"], start)
    ev = env._task_evaluator
    action = idle_action(env)
    action[start["arm"] * 7 + 6] = 1
    result = (0.0, False)
    for _ in range(10):
        for _ in range(32):
            env.step(action, render_obs=False)
        result = ev.chunk_reward()
        if result[1]:
            break
    assert result == (1.0, True), (result, ev._last)
    assert ev._last["in_bin"] and ev._last["label"] == "done"
    env.close()


def test_built_starts_classify_idle_and_restore_exactly():
    for stage in STAGES[:-1]:
        env = env_for(stage)
        checked = 0
        for i in range(12):
            start, _ = build_start(env, SEED_BASE + i, stage)
            if start is None:
                continue
            env.reset_to_stage(start["seed"], start)
            assert env.evaluate_task().metrics["label"] == stage
            action = idle_action(env)
            total = 0
            for _ in range(10):
                for _ in range(32):
                    env.step(action, render_obs=False)
                reward, done = env._task_evaluator.chunk_reward()
                assert not done, (stage, i, env._task_evaluator._last)
                total += reward
            assert abs(total) < 0.05, (stage, i, total)
            runs = []
            for _ in range(2):
                env.reset_to_stage(start["seed"], start)
                for _ in range(32):
                    env.step(action, render_obs=False)
                state = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
                mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                runs.append(state)
            assert np.array_equal(*runs)
            checked += 1
            if checked == 3:
                break
        env.close()
        assert checked == 3, (stage, "fewer than 3 valid start seeds", checked)


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
