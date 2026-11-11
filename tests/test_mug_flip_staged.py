"""Checks for the staged mug-flip reward (abc_sim/task_eval/mug_flip_staged.py).

Runs with pytest or plain python (no pytest in the sim venv):

    MUJOCO_GL=disable PYTHONPATH=<this abc checkout> python tests/test_mug_flip_staged.py

Scenes are built from the real task (turn_mug_right_side_up_staged, seed below), and mug poses are set directly.
Contact flags are passed to ``_score`` by hand where a test is about the scoring logic. Contact detection itself
is covered by the replays of recorded episodes (simulation/experiments/mug_staged_reward).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402
from abc_sim.task_eval import make_task_evaluator  # noqa: E402
from abc_sim.task_eval.mug_flip_staged import MugFlipStagedEvaluator  # noqa: E402

SEED = 703077116  # exp12 world-00: two mugs, both upside down at reset
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
_ENV = None


def _scene():
    global _ENV
    if _ENV is None:
        _ENV = abc_sim.make_env(task="turn_mug_right_side_up_staged", prompt="x", render_cameras=False)
    _ENV.reset(seed=SEED, randomize=True)
    ev = make_task_evaluator(_ENV.model, "turn_mug_right_side_up_staged")
    assert isinstance(ev, MugFlipStagedEvaluator)
    ev.reset(nworld=1)
    return _ENV.model, np.array(_ENV.data.qpos), ev


def _adr(model, k):
    return int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"mug_{k}_jnt")])


def _mug_ids(ev):
    return [int(n.split("_")[1]) for n in ev.mug_names]


def _score(ev, qpos, grasped=(), touched=(), dist=0.3):
    m = len(ev.mug_names)
    g = np.zeros((1, m), bool)
    t = np.zeros((1, m), bool)
    for k in grasped:
        g[0, k] = t[0, k] = True
    for k in touched:
        t[0, k] = True
    return ev._score(qpos[None].astype(np.float32), g, t, np.full((1, m), dist, np.float32))


def _set(model, qpos, k_index, ev, *, quat=None, dz=0.0):
    adr = _adr(model, _mug_ids(ev)[k_index])
    q = qpos.copy()
    q[adr + 2] += dz
    if quat is not None:
        q[adr + 3 : adr + 7] = quat
    return q


def test_stock_task_keeps_its_own_evaluator():
    from abc_sim.task_eval.mugs import MugFlipUprightEvaluator
    from abc_sim.task_specs import get_task_spec

    model, _, _ = _scene()
    for name in ("turn_mug_right_side_up", "turn mug right side up", "mug_flip", "sim_turn_the_mug_right_side_up"):
        assert get_task_spec(name).name == "turn_mug_right_side_up", name
        assert type(make_task_evaluator(model, name)) is MugFlipUprightEvaluator, name
    assert get_task_spec("turn_mug_right_side_up_staged").evaluator_name == "mug_flip_upright_staged"


def test_reset_scores_zero_and_matches_stock():
    model, q0, ev = _scene()
    stock = make_task_evaluator(model, "turn_mug_right_side_up")
    r = ev.evaluate_qpos_batch(q0[None])
    s = stock.evaluate_qpos_batch(q0[None].astype(np.float32))
    assert float(r.reward[0]) == 0.0 and not r.success[0]
    assert float(r.metrics["base_reward"][0]) == float(s.reward[0])


def test_all_mugs_upright_on_table_is_success_and_full_reward():
    model, q0, ev = _scene()
    _score(ev, q0)  # first call records rest heights
    q = q0
    for i in range(len(ev.mug_names)):
        q = _set(model, q, i, ev, quat=UPRIGHT)
    r = _score(ev, q)
    assert r.success[0] and float(r.reward[0]) == 1.0
    assert (r.metrics["mug_stage"][0] == 6).all()


def test_success_is_identical_to_stock_evaluator():
    model, q0, ev = _scene()
    stock = make_task_evaluator(model, "turn_mug_right_side_up")
    rng = np.random.default_rng(0)
    first = True
    for _ in range(50):
        q = q0.copy()
        for i in range(len(ev.mug_names)):
            if rng.random() < 0.6:
                q = _set(model, q, i, ev, quat=UPRIGHT, dz=float(rng.choice([0.0, 0.1])))
        if first:  # both evaluators take rest heights from their first call
            ev.evaluate_qpos_batch(q0[None]), stock.evaluate_qpos_batch(q0[None].astype(np.float32))
            first = False
        a = ev.evaluate_qpos_batch(q[None])
        b = stock.evaluate_qpos_batch(q[None].astype(np.float32))
        assert bool(a.success[0]) == bool(b.success[0])
        assert float(a.metrics["base_reward"][0]) == float(b.reward[0])


def test_upright_in_the_air_gets_no_place_or_release():
    model, q0, ev = _scene()
    _score(ev, q0)
    _score(ev, q0, grasped=[0])
    q = _set(model, q0, 0, ev, quat=UPRIGHT, dz=0.10)
    r = _score(ev, q, grasped=[0])
    v = float(r.metrics["mug_staged_value"][0, 0])
    w = ev.weights
    assert abs(v - (w["reach"] + w["grasp"] + w["lift"] + w["rotate"])) < 1e-5, v
    assert int(r.metrics["mug_stage"][0, 0]) == 4  # working on "place"
    assert not r.success[0]


def test_touching_without_both_fingers_is_not_a_grasp():
    model, q0, ev = _scene()
    _score(ev, q0)
    r = _score(ev, q0, touched=[0])
    assert float(r.metrics["mug_staged_value"][0, 0]) < ev.weights["reach"] + 1e-6
    assert int(r.metrics["mug_stage"][0, 0]) == 0


def test_rotation_only_counts_while_touching():
    model, q0, ev = _scene()
    _score(ev, q0)
    _score(ev, q0, grasped=[0])  # grasp latched
    sideways = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # tilt between down and up
    q = _set(model, q0, 0, ev, quat=sideways, dz=0.05)
    free = _score(ev, q)  # tumbling, no finger contact: no rotation credit
    assert float(free.metrics["mug_rotate"][0, 0]) == 0.0
    held = _score(ev, q, touched=[0])
    assert float(held.metrics["mug_rotate"][0, 0]) > 0.2


def test_regrasp_loops_earn_nothing():
    model, q0, ev = _scene()
    _score(ev, q0)
    lifted = _set(model, q0, 0, ev, dz=0.05)
    rewards = []
    for _ in range(5):
        _score(ev, lifted, grasped=[0])
        rewards.append(float(_score(ev, q0).reward[0]))  # dropped back where it was
    assert max(rewards) == min(rewards), rewards


def test_knocked_over_after_place_loses_place_and_release():
    model, q0, ev = _scene()
    _score(ev, q0)
    up = _set(model, q0, 0, ev, quat=UPRIGHT)
    placed = float(_score(ev, up).metrics["mug_staged_value"][0, 0])
    assert abs(placed - 1.0) < 1e-6
    over = float(_score(ev, q0).metrics["mug_staged_value"][0, 0])
    assert over < placed - ev.weights["place"] - ev.weights["release"] + 1e-6


def test_qpos_path_matches_model_data_path():
    model, q0, ev = _scene()
    ev2 = make_task_evaluator(model, "turn_mug_right_side_up_staged")
    d = mujoco.MjData(model)
    d.qpos[:] = q0
    mujoco.mj_forward(model, d)
    a = ev.evaluate_qpos_batch(q0[None])
    b = ev2.evaluate_model_data(model, d)
    assert np.allclose(a.reward, b.reward)
    assert np.allclose(a.metrics["grasp_site_dist_m"], b.metrics["grasp_site_dist_m"], atol=1e-5)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
