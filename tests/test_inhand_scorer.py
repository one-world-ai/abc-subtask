"""In-hand transfer scorer: the item counts only when it rests on the far side, released and still.

Run (CPU, no GL):
    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_inhand_scorer.py

Phase-0 audit (subtask_rewards/inhand/STATUS.md): the stock rule (far-side y band, height <= table + 10 cm, on the
table) accepted an item still held low over the far side by the receiving gripper, and an item flying or sliding
through the band; with terminate-on-success either ended the episode as a success.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.stagekit import GRIP_MAX, ObjectScene, hold_object, inv, set_object_pose

TASK = "inhand_transfer_item_to_other_side"
SEED = 0  # a can spawned on the left (+y) side; the far side is y <= -0.10
R_DOWN = np.diag([1.0, -1.0, -1.0])  # link_6 z (to the fingers) -> world -z; fingers close along link_6 x
SITE_Z = 0.1347


def _scene(seed=SEED):
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    env.reset(seed=seed, randomize=True)
    joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
    scene = ObjectScene(env.model, [("task_object", joint)])
    hold = env.get_state().astype(np.float32)
    for _ in range(100):  # let the item settle where it spawned
        env.step(hold, render_obs=False)
    return env, scene


def _evaluate(env):
    result = env.evaluate_task()
    return bool(np.asarray(result.success).ravel()[0]), result.metrics


def _move_far(env, scene, lift=0.0, y=-0.25):
    a = int(scene.qpos_adr[0])
    pos, quat = env.data.qpos[a:a + 3].copy(), env.data.qpos[a + 3:a + 7].copy()
    set_object_pose(env, scene, 0, [0.60, y, pos[2] + lift], quat)


def _hold_far_with_right_arm(env, scene, lift):
    """Right gripper holds the item top-down ``lift`` above its rest height on the far side."""
    _move_far(env, scene, lift)
    com = env.data.subtree_com[scene.bodies[0]].copy()
    for yaw in np.linspace(0.0, np.pi, 6, endpoint=False):
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        T6 = np.eye(4)
        T6[:3, :3] = Rz @ R_DOWN
        T6[:3, 3] = com + np.array([0.0, 0.0, SITE_Z - 0.01])
        grasp = {"arm": 1, "T_obj_link6": inv(scene.object_T(env.data, 0)) @ T6,
                 "fingers": np.array([GRIP_MAX - 0.01, GRIP_MAX - 0.01])}
        ok, _ = hold_object(env, scene, 0, grasp)
        if ok:
            return True
        _move_far(env, scene, lift)
    return False


def test_resets_score_zero():
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    for seed in range(10):
        env.reset(seed=seed, randomize=True)
        result = env.evaluate_task()
        assert float(np.asarray(result.reward).ravel()[0]) == 0.0, seed
        assert not bool(np.asarray(result.success).ravel()[0]), seed


def test_held_low_over_far_side_is_not_success():
    env, scene = _scene()
    assert _hold_far_with_right_arm(env, scene, lift=0.03)
    success, m = _evaluate(env)
    assert m["crossed_to_other_side"][0] and m["object_placed"][0] and m["object_on_table"][0]  # old rule: success
    assert not m["object_released"][0]
    assert not success
    # the qpos-only path sees the same contacts
    qres = env._task_evaluator.evaluate_qpos_batch(np.asarray(env.data.qpos, np.float32)[None])
    assert not bool(qres.success[0]) and not bool(qres.metrics["object_released"][0])


def test_moving_through_far_band_is_not_success():
    env, scene = _scene()
    _move_far(env, scene, lift=0.002)
    dof = int(scene.dof_adr[0])
    env.data.qvel[dof:dof + 3] = [0.3, 0.0, 0.0]
    success, m = _evaluate(env)
    assert m["crossed_to_other_side"][0] and m["object_released"][0]
    assert not m["object_at_rest"][0]
    assert not success


def test_resting_on_far_side_is_success():
    env, scene = _scene()
    _move_far(env, scene, lift=0.002)
    hold = env.get_state().astype(np.float32)
    for _ in range(150):
        env.step(hold, render_obs=False)
    success, m = _evaluate(env)
    assert m["object_released"][0] and m["object_at_rest"][0]
    assert success


def test_released_after_hold_settles_to_success():
    """Held low on the far side (not success), then the gripper opens and backs away: success once it rests."""
    env, scene = _scene()
    assert _hold_far_with_right_arm(env, scene, lift=0.01)
    assert not _evaluate(env)[0]
    act = env.get_state().astype(np.float32)
    act[13] = 1.0  # open the right gripper
    for _ in range(40):
        env.step(act, render_obs=False)
    act[7:13] = env.get_init_q()[7:13]  # back to the right arm's home pose
    for _ in range(150):
        env.step(act, render_obs=False)
    success, m = _evaluate(env)
    assert m["object_released"][0] and m["object_at_rest"][0], m
    assert success


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS", t.__name__, flush=True)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", t.__name__, type(exc).__name__, exc, flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
