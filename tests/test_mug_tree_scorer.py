"""Mug-on-rack scorer: only mugs hanging by their handle count.

Run (CPU, no GL):
    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_mug_tree_scorer.py

The poses below are mug_1 resting poses. They were found by dropping the mug at random poses near the peg tips and
letting it settle, then checked on renders (2026-11-13). The 2026-11-13 mug_rack audit
(subtask_rewards/mug_rack/STATUS.md) added a physical label: a mug hangs by its handle when it stays put with only
its handle pieces colliding (probes/mr_counterfactual.py). HANDLE_ON_BRANCH, previously counted, fails that test
(handle alone: falls 31 cm; without the handle: moves 1.7 cm) -- its handle only leans on a peg -- and is now a
rejection case together with the other handle-lean rests that the handle-touch rule accepted.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco
import numpy as np

import abc_sim

TASK = "hang_mug_on_mug_rack"

# mug_1 freejoint qpos (xyz + wxyz), seed 0 (mug_1 is model mug_12). All touch only the tree.
HANDLE_ON_BRANCH = [0.67359, 0.081512, 1.020887, 0.758412, 0.035963, 0.543495, -0.357953]  # handle leans only
HANDLE_ON_TRUNK_TOP = [0.652127, 0.025553, 0.99119, -0.752291, 0.652387, -0.087469, 0.02826]
BRANCH_INSIDE_CUP = [0.627518, 0.030745, 0.893638, 0.457554, -0.319174, 0.406772, -0.723401]
RIM_ON_BRANCH = [0.770526, 0.037113, 0.940083, 0.913754, -0.091763, -0.113746, 0.379072]
# seed 0: body on a peg, handle touching the tree above the CoM (accepted before the threading rule)
HANDLE_LEAN_SEED0 = [0.65009, 0.063983, 0.952334, -0.504676, 0.36907, 0.038367, -0.779498]

# (seed, mug_1 model, qpos): real hangs from other seeds/models, peg through the handle; the handle alone holds them.
THREADED_HANGS = [
    (1, "mug_9", [0.509278, -0.001269, 1.003463, 0.145141, -0.11804, 0.605874, -0.773251]),
    (1, "mug_9", [0.522746, -0.011738, 0.86386, 0.784688, -0.618667, 0.038633, 0.004792]),
    (2, "mug_4", [0.624404, -0.040853, 0.852436, 0.24622, -0.212692, -0.640999, 0.695168]),
    # mug_8's handle sits 4.5 mm off the cup's mid-plane
    (33, "mug_8", [0.625984, 0.132329, 0.99461, 0.71782, -0.693351, -0.012283, 0.062028]),
]
# Rests the handle-touch rule accepted but the handle does not hold (peg in the cup / body on a peg + handle lean).
HANDLE_LEANS = [
    (18, "mug_7", [0.56065, 0.056809, 0.884689, -0.869007, 0.349056, -0.335625, 0.101701]),  # peg inside the cup
    (1, "mug_9", [0.55461, 0.025533, 1.010247, 0.754138, -0.316154, 0.429817, -0.382857]),
    (6, "mug_10", [0.660662, 0.070225, 0.978803, 0.336052, -0.095653, 0.612527, -0.709035]),
    (33, "mug_8", [0.665694, 0.080872, 1.027295, 0.217339, -0.380906, 0.355801, -0.825276]),
]


def _env(seed: int = 0, model: str | None = None):
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    env.reset(seed=seed)
    if model is not None:
        assert env._last_randomization.metadata["mug_variants"][0] == model, env._last_randomization.metadata
    return env


def _hanging_mask(env, mug1_qpos=None, settle_steps: int = 0) -> list[bool]:
    model, data = env.model, env.data
    if mug1_qpos is not None:
        adr = env._task_evaluator._mug_qpos_addrs[0]
        data.qpos[adr : adr + 7] = mug1_qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    result = env.evaluate_task()
    names = env._task_evaluator.mug_names
    hanging = set(result.metrics["mugs_hanging"][0])
    return [name in hanging for name in names]


def test_resets_score_zero():
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    for seed in range(10):
        env.reset(seed=seed)
        assert float(np.asarray(env.evaluate_task().reward).ravel()[0]) == 0.0, seed


def test_every_mug_model_has_a_handle():
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    for seed in range(10):
        env.reset(seed=seed)
        for info in env._task_evaluator._mug_handles:
            assert info["handle_sectors"].size > 0, seed


def test_every_mug_model_has_a_handle_hole():
    # All 19 models (scale 1); the smallest holes (mug_1, mug_7, mug_10) are narrower than a peg but still measured.
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    for v in range(19):
        request = {"mug_variants": [f"mug_{v}"], "mug_count": 1, "randomize_scales": False}
        env.reset(seed=0, options={"randomization": request})
        assert len(env._task_evaluator._hole(0)) > 100, v


def test_handle_hangs_count():
    env = _env()
    assert _hanging_mask(env, HANDLE_ON_TRUNK_TOP)[0]
    for seed, model, qpos in THREADED_HANGS:
        env = _env(seed, model)
        assert _hanging_mask(env, qpos)[0], (seed, model)


def test_handle_leaning_on_the_tree_does_not_count():
    env = _env()
    assert not _hanging_mask(env, HANDLE_ON_BRANCH)[0]
    env = _env()
    assert not _hanging_mask(env, HANDLE_LEAN_SEED0)[0]
    for seed, model, qpos in HANDLE_LEANS:
        env = _env(seed, model)
        assert not _hanging_mask(env, qpos)[0], (seed, model)


def test_hangs_survive_settling():
    # The poses are at rest: two more seconds of physics keep them hanging.
    env = _env()
    assert _hanging_mask(env, HANDLE_ON_TRUNK_TOP, settle_steps=1000)[0]
    seed, model, qpos = THREADED_HANGS[3]
    env = _env(seed, model)
    assert _hanging_mask(env, qpos, settle_steps=1000)[0]


def test_peg_inside_cup_or_rim_rest_does_not_count():
    env = _env()
    assert not _hanging_mask(env, BRANCH_INSIDE_CUP)[0]
    env = _env()
    assert not _hanging_mask(env, RIM_ON_BRANCH)[0]


def test_mug_on_table_beside_trunk_does_not_count():
    env = _env()
    ev = env._task_evaluator
    adr = ev._mug_qpos_addrs[0]
    z = float(env.data.qpos[adr + 2])
    pose = [ev._axis_xy[0] + 0.10, ev._axis_xy[1], z + 0.01, 1.0, 0.0, 0.0, 0.0]
    assert not _hanging_mask(env, pose, settle_steps=1500)[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
