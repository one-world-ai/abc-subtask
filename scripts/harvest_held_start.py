"""Reconstruct candidate grasps from ABC sim_224 poses and settle them in the training simulator."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.stagekit.robot import STATE_SPEC
from abc_sim.task_eval.stagekit.starts import save_starts
from scripts.replay_abc_goal import episode_model


def snapshot(env):
    """Copy the complete integration state, including controls and solver warm start."""
    state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
    return state


def harvest(episode):
    """Keep the first recorded grasp that survives a stationary half-second under gravity."""
    meta = json.loads((episode / "episode_metadata.json").read_text())
    task = {"sim_pouring_beads": "pouring__pour",
            "sim_throw_plastic_bottles_in_bin": "throw_plastic_bottles_in_bin__throw"}[meta["task_name"]]
    seed = json.loads((episode / "randomization.json").read_text())["seed"]
    poses = np.load(episode / "scene_qpos.npy")
    recorded = episode_model(episode, meta)
    env = abc_sim.make_env(task=task, render_cameras=False)
    try:
        env.reset(seed=seed)
        m, d, ev = env.model, env.data, env.subtask
        if [(j.name, j.type.item()) for j in (m.jnt(i) for i in range(m.njnt))] != [
                (recorded.jnt(i).name, recorded.jnt(i).type.item()) for i in range(recorded.njnt)]:
            raise ValueError("Recorded and current joint layouts differ")
        rest = poses[0, ev.scene.qpos_adr + 2].copy()
        for frame in range(0, len(poses), 4):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = poses[frame]
            mujoco.mj_forward(m, d)
            for k in np.flatnonzero(ev.scene.contact_obs(d)["grasped"]):
                if not ev.valid_start(d, k, rest):
                    continue
                action = d.qpos[env._qpos_indices].copy()
                arm = int(ev.scene.contact_obs(d)["grasp_arm"][k])
                action[[6, 13]] = np.clip(action[[6, 13]] / 0.0475, 0, 1)
                action[(0 if arm == 2 else arm) * 7 + 6] = 0
                before = d.xpos[ev.scene.bodies[k]].copy()
                for _ in range(16):
                    env.step(action, render_obs=False)
                if ev.valid_start(d, k, rest) and np.linalg.norm(d.xpos[ev.scene.bodies[k]] - before) < 0.03:
                    row = dict(seed=seed, target=int(k), arm=arm, rest_height=rest,
                               labels=np.zeros(len(rest), dtype=int), state=snapshot(env),
                               episode=episode.name, frame=frame)
                    env.reset_to_stage(seed, row)
                    if not np.array_equal(snapshot(env), row["state"]):
                        raise ValueError("Snapshot did not restore exactly")
                    return task, row
                break
        return task, None
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("episodes", nargs="+", type=Path)
    args = parser.parse_args()
    groups = {}
    for p in args.episodes:
        task, row = harvest(p)
        print(p.name, task, (row["seed"], row["frame"]) if row else "rejected", flush=True)
        if row:
            groups.setdefault(task, []).append(row)
    for task, rows in groups.items():
        parent, stage = task.split("__")
        seeds = sorted({row["seed"] for row in rows})
        held_out = set(seeds[-3:])
        for row in rows:
            row["split"] = "eval" if row["seed"] in held_out else "train"
        save_starts(args.output / parent / (stage + ".npz"), rows, stage=stage)
