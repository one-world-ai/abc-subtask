"""Sweep scorer: a scrap counts only inside the upright bin's tapered inner wall.

Run (CPU, no GL):
    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_sweep_scorer.py

The old rule used the rim radius (~8.5-9 cm) at every height of a bin that is only ~6.9 cm wide at its floor, and
no tilt check. It accepted scraps lying on the table against the outside of the bin base (reset seed 2; 26 of 931
random table rests next to an upright bin), a scrap under the lifted edge of a tilted bin (DreamZero exp7 seed
970910683 step 600, recorded "success" at 600 steps) and scraps inside a bin lying on its side (96 of 96).
Poses come from subtask_rewards/sweep/probes/sw_settle.py and sw_replay.py (2026-11-13).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco
import numpy as np

import abc_sim

TASK = "sweep_away_paper_scraps_from_table"

# Seed 2, bin settled upright. trash_1 resting on the table against the outside of the bin base (old rule: in).
UPRIGHT_BIN_SEED2 = [0.624961, 0.053274, 0.836338, 0.698479, 0.698479, -0.110126, -0.110119]
AGAINST_BASE_SEED2 = [0.652016, -0.034606, 0.766206, -0.366002, 0.928703, -0.057008, 0.017389]
INSIDE_SEED2 = [0.623714, 0.055867, 0.771751, -0.893897, -0.308898, 0.304155, 0.114109]
# Seed 0, bin settled upright, and settled lying on its side (axis 84 deg from vertical).
UPRIGHT_BIN_SEED0 = [0.657347, 0.104194, 0.829865, 0.705954, 0.705954, 0.040351, 0.040358]
AGAINST_BASE_SEED0 = [0.59529, 0.042956, 0.767696, -0.381248, 0.841839, 0.375371, -0.071085]
INSIDE_SEED0 = [0.668139, 0.107699, 0.772381, 0.143647, -0.072159, 0.923633, -0.347937]
TIPPED_BIN_SEED0 = [0.658658, 0.111765, 0.821809, 0.048736, 0.997688, 0.003607, 0.047222]
INSIDE_TIPPED_SEED0 = [0.658861, 0.084246, 0.775729, 0.756118, 0.521001, 0.122911, 0.376479]
# DreamZero exp7, seed 970910683, step 600: bin pushed and tilted 20 deg; trash_1 on the table under its lifted
# edge, trash_2 inside it.
DZ_SEED = 970910683
DZ_BIN = [0.535201, -0.081086, 0.852013, 0.584926, 0.611993, -0.47816, -0.233857]
DZ_TRASH = {
    "trash_1": [0.505362, -0.021789, 0.764957, 0.265992, -0.69163, -0.530196, 0.412054],
    "trash_2": [0.511702, -0.071394, 0.782149, -0.2945, 0.341719, 0.878402, 0.157822],
}
# OpenWAM canon 14B exp12 world-06 (seed 157082072), step 2250: real clear, all three scraps in the upright bin.
W06_SEED = 157082072
W06_BIN = [0.651381, -0.044047, 0.832319, 0.599157, 0.599155, -0.375526, -0.375505]
W06_TRASH = {
    "trash_1": [0.636645, -0.056125, 0.77258, -0.455591, -0.861282, 0.053074, -0.218664],
    "trash_2": [0.620845, -0.016938, 0.777595, -0.379746, 0.561001, -0.735507, 0.010044],
    "trash_3": [0.673143, -0.056811, 0.774978, 0.27352, 0.351169, 0.654522, 0.611121],
}


def _env(seed: int):
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    env.reset(seed=seed)
    return env


def _pose(env, bin_qpos, trash: dict[str, list[float]], settle_steps: int = 0):
    """Set the bin and some scraps, move the other scraps far from the bin, return the evaluation."""
    model, data, ev = env.model, env.data, env._task_evaluator
    data.qpos[ev._bin_qpos_adr : ev._bin_qpos_adr + 7] = bin_qpos
    for k, (name, adr) in enumerate(zip(ev.trash_names, ev._trash_qpos_addrs)):
        if name in trash:
            data.qpos[adr : adr + 7] = trash[name]
        elif data.qpos[adr + 2] > 0.0:  # active scrap not under test: park it on the far table corner
            data.qpos[adr : adr + 3] = [0.35 + 0.05 * k, -0.55, 0.77]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    return env.evaluate_task()


def _in_bin(result) -> set[str]:
    return set(result.metrics["scraps_in_bin"][0])


def test_resets_score_zero():
    # Seed 2 scored 1/4 at reset under the old rule (a scrap 9.3 cm from the bin axis, on the table).
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    for seed in range(20):
        env.reset(seed=seed)
        result = env.evaluate_task()
        assert float(np.asarray(result.reward).ravel()[0]) == 0.0, seed
        assert not bool(np.asarray(result.success).ravel()[0]), seed


def test_bin_wall_tapers_for_every_scale():
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    for seed in range(10):
        env.reset(seed=seed)
        ev = env._task_evaluator
        assert 0.05 < ev._bin_radius_bottom < ev._bin_radius_top < 0.1, seed
        assert ev._bin_top_y - ev._bin_bottom_y > 0.15, seed


def test_scrap_against_outside_of_base_does_not_count():
    assert "trash_1" not in _in_bin(_pose(_env(2), UPRIGHT_BIN_SEED2, {"trash_1": AGAINST_BASE_SEED2}))
    assert "trash_1" not in _in_bin(_pose(_env(0), UPRIGHT_BIN_SEED0, {"trash_1": AGAINST_BASE_SEED0}))


def test_scrap_under_tilted_bin_edge_does_not_count():
    result = _pose(_env(DZ_SEED), DZ_BIN, DZ_TRASH)
    assert 0.9 < float(result.metrics["bin_up_z"][0]) < 0.96  # tilted ~20 deg, still "upright" by the rule
    assert _in_bin(result) == {"trash_2"}
    assert not bool(np.asarray(result.success).ravel()[0])


def test_tipped_bin_never_counts():
    result = _pose(_env(0), TIPPED_BIN_SEED0, {"trash_1": INSIDE_TIPPED_SEED0})
    assert not bool(result.metrics["bin_upright"][0])
    assert _in_bin(result) == set()


def test_scraps_inside_upright_bin_count():
    assert "trash_1" in _in_bin(_pose(_env(2), UPRIGHT_BIN_SEED2, {"trash_1": INSIDE_SEED2}))
    assert "trash_1" in _in_bin(_pose(_env(0), UPRIGHT_BIN_SEED0, {"trash_1": INSIDE_SEED0}))


def test_scraps_inside_survive_settling():
    result = _pose(_env(0), UPRIGHT_BIN_SEED0, {"trash_1": INSIDE_SEED0}, settle_steps=1000)
    assert "trash_1" in _in_bin(result)
    result = _pose(_env(0), UPRIGHT_BIN_SEED0, {"trash_1": AGAINST_BASE_SEED0}, settle_steps=1000)
    assert "trash_1" not in _in_bin(result)


def test_recorded_real_clear_still_succeeds():
    result = _pose(_env(W06_SEED), W06_BIN, W06_TRASH, settle_steps=500)
    assert _in_bin(result) == set(W06_TRASH)
    assert bool(np.asarray(result.success).ravel()[0])
    assert float(np.asarray(result.reward).ravel()[0]) == 1.0


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
