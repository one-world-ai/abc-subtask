"""Bottle reward contract: real simulator measurements and adversarial transitions. CPU, no renderer."""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "disable")
import traceback
import mujoco
import numpy as np
import abc_sim
from abc_sim.task_eval.bottles_stages import C, D, DONE, P, R, STAGES, StageConfig, classify
from abc_sim.task_eval.bottles_starts import SEED_BASE, build_start, idle_action
from abc_sim.task_eval.stagekit import ARMS, arm_ids, axis_angle_quat, qmul, set_object_pose
from abc_sim.task_specs import get_task_spec


def env_for(stage="reach", seed=SEED_BASE):
    env = abc_sim.make_env(task=f"put_plastic_bottles_in_bin__{stage}", render_cameras=False)
    env.reset(seed=seed)
    env.evaluate_task()
    return env


def updates(env, n=6):
    for _ in range(n):
        env.evaluate_task()
    return env._task_evaluator.chunk_reward()


def test_lookup_preserves_stock():
    assert get_task_spec("put_plastic_bottles_in_bin").evaluator_name == "put_bottles_in_bin"
    for stage in STAGES[:-1]:
        spec = get_task_spec(f"put_plastic_bottles_in_bin__{stage}")
        assert get_task_spec(spec.prompt).name == spec.name
        assert spec.evaluator_kwargs() == {"stage": stage}


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
    for tip in (False, True):
        env = env_for()
        adr = env._task_evaluator.scene.bin_adr
        if tip:
            env.data.qpos[adr + 3:adr + 7] = qmul(axis_angle_quat(np.array([1., 0, 0]), np.pi / 2),
                                               env.data.qpos[adr + 3:adr + 7].copy())
        else:
            env.data.qpos[adr] += 0.06
        mujoco.mj_forward(env.model, env.data)
        assert updates(env) == (-1.0, True)
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


def test_falling_off_the_table_fails():
    env = env_for()
    ev = env._task_evaluator
    env.data.qpos[ev.scene.qpos_adr[ev.target] + 2] -= 0.20
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True)
    assert ev._failed == "fell off the table"
    env.close()


def test_knocking_a_previously_binned_bottle_out_fails():
    env = env_for()
    ev, data = env._task_evaluator, env.data
    scene, target = ev.scene, ev.target
    other = next(i for i in range(len(scene.names)) if i != target)
    adr = scene.qpos_adr[other]
    bin_pos = data.qpos[scene.bin_adr:scene.bin_adr + 3]
    pos = bin_pos + data.xmat[scene.bin_body].reshape(3, 3) @ [0, scene.geometry._bin_bottom_y + 0.04, 0]
    offset = data.qpos[adr:adr + 3] - data.subtree_com[scene.bodies[other]]
    set_object_pose(env, scene, other, pos + offset, data.qpos[adr + 3:adr + 7].copy())
    ev.begin(env.model, data, target)
    assert ev._binned[other]
    data.qpos[adr] += 0.4
    mujoco.mj_forward(env.model, data)
    assert updates(env) == (-1.0, True)
    assert ev._failed == "knocked a binned bottle out"
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
    pos = data.qpos[scene.bin_adr:scene.bin_adr + 3].copy()
    pos[2] += scene.geometry._bin_top_y + 0.03
    adr = scene.qpos_adr[k]
    offset = data.qpos[adr:adr + 3] - data.subtree_com[scene.bodies[k]]
    set_object_pose(env, scene, k, pos + offset, data.qpos[adr + 3:adr + 7].copy())
    obs = scene.observe(data, rest)
    assert obs["in_bin"][k] and not obs["supported"][k]
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


def test_airborne_bottle_contact_cycle_is_not_support():
    env = env_for()
    ev, data = env._task_evaluator, env.data
    scene = ev.scene
    assert len(scene.names) >= 2
    rest = scene.heights(data)
    bin_pos = data.qpos[scene.bin_adr:scene.bin_adr + 3].copy()
    rim = bin_pos + data.xmat[scene.bin_body].reshape(3, 3) @ [0, scene.geometry._bin_top_y, 0]
    quat = data.qpos[scene.qpos_adr[0] + 3:scene.qpos_adr[0] + 7].copy()
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    for k, dz in ((0, 0.05), (1, 0.075)):
        center = rim + [0, 0, dz]
        offset = rotation.reshape(3, 3) @ scene.geometry._bottle_center_offsets[k]
        set_object_pose(env, scene, k, center - offset, quat)
    obs = scene.observe(data, rest)
    assert obs["in_bin"][:2].all()
    assert obs["resting_contact"][:2].all(), "fixture must contain an actual bottle-to-bottle contact"
    assert not obs["touched"][:2].any()
    assert not obs["supported"][:2].any()
    assert not (classify(obs, np.full(len(scene.names), D), ev.cfg)[:2] == DONE).any()
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
                runs.append(env.data.qpos.copy())
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
