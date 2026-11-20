"""Run held-start counterexamples and optionally rescore donor episode traces."""
import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np

import abc_sim
from abc_minimal.episode_io import load_episode
from abc_sim.task_eval.stagekit.starts import read_starts
from scripts.harvest_held_start import snapshot


def action_replay(env, row, path):
    """Drive the restored simulator with the donor's recorded commands on its 30 Hz clock."""
    meta, _, actions = load_episode(path)
    env.reset_to_stage(row["seed"], row)
    origin = env.data.time
    while True:
        index = int(row["frame"]) + round((env.data.time - origin) * meta["fps"])
        if index >= len(actions):
            return "unfinished"
        _, _, terminated, truncated, _ = env.step(actions[index], render_obs=False)
        if terminated or truncated:
            return env.subtask.failure or ("success" if env.subtask.success else "unfinished")


def check(stage_file, episodes, details=None):
    """Check restore, negative controls, fault latching, and donor traces for every saved row."""
    rows = read_starts(stage_file)
    task = stage_file.parent.name + "__" + stage_file.stem
    env = abc_sim.make_env(task=task, render_cameras=False)
    counts = Counter()
    try:
        for row in rows:
            for mode in ("idle", "open", "sweep", "velocity"):
                env.reset_to_stage(row["seed"], row)
                ev = env.subtask
                counts["restore_exact"] += int(np.array_equal(snapshot(env), row["state"]))
                action = env.data.ctrl[env._ctrl_indices].copy()
                action[[6, 13]] /= 0.0475
                a = min(row["arm"], 1) * 7
                if mode in ("open", "sweep"):
                    action[a+6] = 1
                if mode == "velocity":
                    env.data.qvel[ev.scene.dof_adr[row["target"]]] = 4.1
                    env.evaluate_task()
                    counts["fault_latched"] += int(ev.failure == "sim_fault")
                    continue
                for step in range(600):
                    if mode == "sweep":
                        action[a] += 0.015 if step < 50 else -0.015 if step < 100 else 0
                    env.step(action, render_obs=False)
                    if ev.success or ev.failure:
                        break
                counts[mode + "_success"] += int(ev.success)
                counts[mode + "_sim_fault"] += int(ev.failure == "sim_fault")
                counts["early_success"] += int(ev.success and step < 5)
            if str(row["episode"]) in episodes:
                path = episodes[str(row["episode"])]
                if (path / "states_actions.bin").exists():
                    action_verdict = action_replay(env, row, path)
                    counts["actions_" + action_verdict] += 1
                else:
                    action_verdict = "missing actions"
                poses = np.load(path / "scene_qpos.npy")
                fps = json.loads((path / "episode_metadata.json").read_text())["fps"]
                env.reset_to_stage(row["seed"], row)
                ev = env.subtask
                origin = env.data.time
                for frame in range(int(row["frame"])+1, len(poses)):
                    mujoco.mj_differentiatePos(env.model, env.data.qvel, 1/fps, poses[frame-1], poses[frame])
                    env.data.qpos[:] = poses[frame]
                    env.data.time = origin + (frame-int(row["frame"]))/fps
                    mujoco.mj_forward(env.model, env.data)
                    env.evaluate_task()
                    if ev.success or ev.failure:
                        break
                trace_verdict = ev.failure or ("success" if ev.success else "unfinished")
                counts["trace_" + trace_verdict] += 1
                if details is not None:
                    details.append(dict(episode=path.name, seed=row["seed"], split=str(row["split"]),
                                        start_frame=int(row["frame"]), trace=trace_verdict,
                                        trace_end_frame=frame, action_replay=action_verdict))
        return dict(task=task, starts=len(rows), **counts)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes", nargs="*", type=Path, default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    reports = []
    for file in sorted(args.root.glob("*/*.npz")):
        details = []
        result = check(file, {p.name: p for p in args.episodes}, details)
        print(json.dumps(result), flush=True)
        reports.append(dict(summary=result, episodes=details))
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2) + "\n")
