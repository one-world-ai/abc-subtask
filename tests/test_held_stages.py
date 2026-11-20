"""Counterexamples unique to the held pour and throw tasks."""
from pathlib import Path

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.stagekit.starts import read_starts

ROOT = Path(__file__).resolve().parents[1] / "subtask_rewards/held_starts"


def restored(task, stage):
    """Start an episode at the first saved grasp."""
    env = abc_sim.make_env(task=f"{task}__{stage}", render_cameras=False)
    row = read_starts(ROOT / task / (stage + ".npz"))[0]
    env.reset_to_stage(row["seed"], row)
    return env


def test_transient_bead_membership_does_not_complete_pour():
    env = restored("pouring", "pour")
    try:
        ev, data = env.subtask, env.data
        # Isolate retention after a qualifying tilt; synthesize just the bead positions.
        ev.tilted = True
        cup = data.body("cup_1")
        center = cup.xpos + cup.xmat.reshape(3, 3)[:, 2] * (ev.stock._container_floor_z + ev.stock._container_rim_z) / 2
        starts = data.qpos.copy()
        for adr in ev.stock._bead_qpos_addrs:
            data.qpos[adr:adr+3] = center
        mujoco.mj_forward(env.model, data)
        assert ev.stock.evaluate_qpos_batch(data.qpos[None]).scalar_success()
        env.evaluate_task()
        assert not ev.failure and not ev.success and ev.stable_since is not None
        data.qpos[:] = starts
        data.time += ev.HOLD_SECONDS
        mujoco.mj_forward(env.model, data)
        env.evaluate_task()
        assert ev.stable_since is None and not ev.success
    finally:
        env.close()


def test_release_inside_bin_without_outside_flight_fails():
    env = restored("throw_plastic_bottles_in_bin", "throw")
    try:
        ev, data = env.subtask, env.data
        k = ev.target
        adr = ev.scene.qpos_adr[k]
        bin_pos = data.body("bin_container").xpos.copy()
        data.qpos[adr:adr+3] = bin_pos + np.array([0, 0, ev.stock._bin_top_z * 0.6])
        mujoco.mj_forward(env.model, data)
        result = ev.stock.evaluate_qpos_batch(data.qpos[None])
        assert result.metrics["bottle_in_bin_mask"][0, k]
        env.evaluate_task()
        assert ev.failure == "released without outside flight" and ev.chunk_reward() == (-1, True)
    finally:
        env.close()


def test_seed_alone_restores_the_saved_grasp():
    for task, stage in (("pouring", "pour"), ("throw_plastic_bottles_in_bin", "throw")):
        env = restored(task, stage)
        try:
            pose = env.data.qpos.copy()
            row = read_starts(ROOT / task / (stage + ".npz"))[0]
            env.reset_to_stage(row["seed"])
            assert np.array_equal(env.data.qpos, pose)
        finally:
            env.close()
