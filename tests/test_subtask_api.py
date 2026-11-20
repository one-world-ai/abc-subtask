"""Subtask catalogue (self-registration) and the public subtask API: env.subtask, env.advance_stage, start files.
CPU, no renderer."""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "disable")
import tempfile
import traceback
from pathlib import Path

import numpy as np

import abc_sim
from abc_sim.task_eval.bottles_starts import SEED_BASE, idle_action, move_link
from abc_sim.task_eval.stagekit import ARMS, T, arm_ids, load_starts, read_starts, save_starts
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator
from abc_sim.task_eval.subtasks import specs_for, stage_spec, subtask_evaluators, subtask_specs
from abc_sim.task_specs import get_task_spec, list_task_specs

TASKS = {
    "pouring", "throw_plastic_bottles_in_bin",
    "turn_mug_right_side_up", "put_plastic_bottles_in_bin", "sort_lego_blocks", "grab_specific_object_from_clutter",
    "put_relative", "sort_nuts_and_bolts", "sweep_away_paper_scraps_from_table", "inhand_transfer_item_to_other_side",
    "open_drawer", "close_drawer", "hang_mug_on_mug_rack", "load_plates_into_dish_rack",
}


def test_every_subtask_module_registers_itself():
    evaluators = subtask_evaluators()
    assert {cls.TASK for cls in evaluators.values()} == TASKS, sorted(cls.TASK for cls in evaluators.values())
    for name, cls in evaluators.items():
        assert issubclass(cls, SubtaskEvaluator) and cls.NAME == name and cls.ENV_TASK, cls
    specs = subtask_specs()
    assert len(specs) == 55, len(specs)
    assert len({s.name for s in specs}) == len(specs)


def test_specs_come_from_the_class():
    for cls in subtask_evaluators().values():
        for spec in specs_for(cls):
            stage = spec.evaluator_kwargs()["stage"]
            assert spec.name == f"{cls.TASK}__{stage}" and spec.env_task == cls.ENV_TASK
            assert spec.prompt == cls.STAGE_PROMPTS[stage] and spec.evaluator_name == cls.NAME
            assert get_task_spec(spec.name) == spec and get_task_spec(spec.prompt) == spec
    assert stage_spec("put_plastic_bottles_in_bin", "carry").name == "put_plastic_bottles_in_bin__carry"
    try:
        stage_spec("put_plastic_bottles_in_bin", "handoff")
    except KeyError as e:
        assert "reach" in str(e)
    else:
        raise AssertionError("unknown stage accepted")


def test_stock_tasks_unchanged():
    names = {s.name for s in list_task_specs()}
    assert "put_plastic_bottles_in_bin" in names and "turn_mug_right_side_up" in names
    assert get_task_spec("put_plastic_bottles_in_bin").evaluator_name == "put_bottles_in_bin"
    assert len(names) == len(list_task_specs())


def test_env_subtask_is_none_for_a_whole_task():
    env = abc_sim.make_env(task="put_plastic_bottles_in_bin", render_cameras=False)
    assert env.subtask is None
    for call in (lambda: env.reset_to_stage(SEED_BASE), lambda: env.advance_stage()):
        try:
            call()
        except RuntimeError as e:
            assert "not a subtask" in str(e)
        else:
            raise AssertionError("whole-task env accepted a subtask call")
    env.close()


def _row(seed, n_obj=3, **extra):
    return {"seed": seed, "target": 1, "arm": 0, "state": np.arange(5 + seed % 3, dtype=np.float64),
            "rest_height": np.zeros((n_obj, 3)), "labels": np.zeros(n_obj, np.int64), **extra}


def test_start_files_round_trip_and_normalise():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [_row(7), _row(8, n_obj=4)]  # ragged state and object count
        save_starts(root / "some_task" / "carry.npz", rows, stage="carry")
        starts = load_starts(root, "some_task", "carry")
        assert len(starts) == 2 and starts.stage == "carry" and list(starts.seeds) == [7, 8]
        for got, want in zip(starts, rows):
            assert got["stage"] == "carry" and got["target"] == 1 and got["arm"] == 0
            assert np.array_equal(got["state"], want["state"]) and got["labels"].shape == want["labels"].shape
        # mug-flip layout: target under "mug", no stage key, extra columns
        z = {"seed": np.array([3]), "mug": np.array([2]), "arm": np.array([1]), "variant": np.array(["mug_6"]),
             "state": np.empty(1, object), "rest_height": np.empty(1, object), "labels": np.empty(1, object)}
        z["state"][0], z["rest_height"][0], z["labels"][0] = np.ones(4), np.ones(2), np.zeros(2, np.int64)
        np.savez(root / "rotate.npz", **z)
        row = read_starts(root / "rotate.npz")[0]
        assert row["target"] == 2 and row["mug"] == 2 and row["stage"] == "rotate" and row["variant"] == "mug_6"
        assert starts.sample(np.random.default_rng(0))["seed"] in (7, 8)
        try:
            load_starts(root, "some_task", "drop")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing start file accepted")


def _pick_episode():
    """A real bottles pick episode that reaches +1: built start, close, lift 8 cm."""
    env = abc_sim.make_env(task="put_plastic_bottles_in_bin__pick", render_cameras=False)
    for seed in range(SEED_BASE, SEED_BASE + 6):
        _, info = env.reset_to_stage(seed)
        ev, arm = env.subtask, info["arm"]
        action = idle_action(env)
        action[arm * 7 + 6] = 0.0
        for _ in range(32):
            env.step(action, render_obs=False)
        ids = arm_ids(env.model, ARMS[arm])
        target = T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])
        target[:3, 3] += (0, 0, 0.08)
        move_link(env, arm, target, actions=64, grip=0.0)
        hold = idle_action(env)  # fixed targets: re-reading qpos every step would let the arm sag
        for _ in range(10):
            for _ in range(32):
                env.step(hold, render_obs=False)
            reward, done = ev.chunk_reward()
            if done:
                break
        if ev.success:
            return env, arm
    raise AssertionError("no pick episode reached +1 on 6 seeds")


def test_advance_stage_continues_in_place():
    env, arm = _pick_episode()
    old = env.subtask
    assert old.next_stage() == "carry", old.next_stage()
    qpos = env.data.qpos.copy()
    try:
        env.advance_stage("drop")
    except ValueError as e:
        assert "not 'drop'" in str(e)
    else:
        raise AssertionError("advanced to a stage the target is not in")
    info = env.advance_stage()
    ev = env.subtask
    assert ev is not old and ev.stage_name == "carry" and info["subtask"] == "carry"
    assert info["prompt"] == ev.stage_prompt() == env.prompt and info["prompt"] != old.stage_prompt()
    assert np.array_equal(env.data.qpos, qpos), "advance_stage changed the sim state"
    assert ev.target == old.target and np.array_equal(ev.tracker.rest_height, old.tracker.rest_height)
    hold = idle_action(env)
    for _ in range(3):  # holding still: in stage, no failure, no success
        for _ in range(32):
            env.step(hold, render_obs=False)
        reward, done = ev.chunk_reward()
        assert not done and not ev.failure and abs(reward) < 0.05, (reward, done, ev.failure, ev.last)
    assert ev.last["label"] == "carry" and ev.last["subtask"] == "carry"
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
