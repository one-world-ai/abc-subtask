"""Restored stage starts continue exactly like the recorded episode.

For a replayed world (simulation/experiments/mug_stages/replay_stages.py output + its recorded job dir), restore
each saved start with ``env.reset_to_snapshot``, apply the recorded actions from that step on for ``--steps``
actions, and compare against the recording: arm state at every model call and the traced tilt/lift of every mug and the
stage of the start's mug.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_stage_starts.py \
        ../simulation/outputs/openwam/experiment12/step-57000/tasks/turn_mug_right_side_up/world-30 \
        ../simulation/outputs/mug_stages/exp12_57000/world-30.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402
from abc_sim.task_eval.mug_flip_stages import LABELS, MugScene, MugStageTracker  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("job_dir", type=Path)
    p.add_argument("npz", type=Path)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--max-starts", type=int, default=12)
    a = p.parse_args()
    trial = next(a.job_dir.glob("trial-*"))
    episode = json.loads((trial / "episode.json").read_text())
    execute = int(episode["policy_metadata"]["execute_chunk_dim"])
    calls = [c for c in map(json.loads, (a.job_dir / "server-trace.jsonl").open()) if "actions" in c]
    actions = np.concatenate([np.asarray(c["actions"], np.float32)[:execute] for c in calls])
    call_state = {i * execute: np.asarray(c["state"], np.float32) for i, c in enumerate(calls)}
    z = np.load(a.npz)
    total = z["trace_label"].shape[0] - 1
    env = abc_sim.make_env(task="turn_mug_right_side_up", prompt=episode["prompt"], render_cameras=False)
    idx = np.linspace(0, len(z["snap_step"]) - 1, min(a.max_starts, len(z["snap_step"]))).astype(int)
    failures = 0
    for i in idx:
        s0, k = int(z["snap_step"][i]), int(z["snap_mug"][i])
        env.reset_to_snapshot(int(z["seed"]), z["snap_state"][i])
        tracker = MugStageTracker(MugScene(env.model))
        tracker.reset(env.data, rest_height=z["rest_height"], labels=z["trace_label"][s0])
        worst, label_miss = 0.0, 0
        for t in range(s0, min(s0 + a.steps, total)):
            if t in call_state:
                worst = max(worst, float(np.abs(env.get_state() - call_state[t]).max()))
            env.step(actions[t], render_obs=False)
            tracker.update(env.data)
            o = tracker.obs
            worst = max(worst, float(np.abs(o["upz"] - z["trace_upz"][t + 1]).max()),
                        float(np.abs(o["lift"] - z["trace_lift"][t + 1]).max()))
            # Only the start's own mug: the other mugs may have had a stage change pending (debounce) at the cut.
            label_miss += int(tracker.committed[k] != z["trace_label"][t + 1][k])
        ok = worst < 1e-5 and label_miss == 0
        failures += not ok
        print(f"{'ok ' if ok else 'FAIL'} start {i:3d} step {s0:5d} mug {k} {LABELS[int(z['snap_stage'][i])]:6s} "
              f"max|diff| {worst:.2e} label mismatches {label_miss}")
    env.close()
    print(f"{len(idx) - failures}/{len(idx)} restored starts match the recording")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
