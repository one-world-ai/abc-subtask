"""Replay recorded ABC rollouts bit-exactly and re-score them with the current evaluator.

Each job dir is one trial of simeval/run_eval_parallel.py (trial-*/episode.json, trial-*/reward_trace.json,
server-trace.jsonl). The world is rebuilt from its seed, the recorded actions are stepped, and the sim state is
compared with the state stored at every model call (must match exactly). The evaluator is run at every step where
the recording has a reward.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        .claude/skills/abc-scorer-audit/scripts/rescore_replay.py --task hang_mug_on_mug_rack \
        --out <dir> <job_dir> [<job_dir> ...]

Writes <out>/<world>.json (summary + per-check rows with the evaluator's metrics) and prints one summary per world.
Run several job dirs as separate processes for speed (one world takes minutes for 22k steps).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402

# PYTHONPATH must point at the abc checkout under test; `python -I` would silently import the venv's installed copy.
print(f"abc_sim from {abc_sim.__file__}", flush=True)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def rescore(job: Path, task: str, keep_metrics: bool) -> dict:
    trial = next(job.glob("trial-*"))
    episode = json.loads((trial / "episode.json").read_text())
    recorded = {r["step"]: r for r in json.loads((trial / "reward_trace.json").read_text())["trace"]}
    calls = [c for c in map(json.loads, (job / "server-trace.jsonl").open()) if "actions" in c]
    execute = int(episode["policy_metadata"]["execute_chunk_dim"])
    total = int(episode["steps_executed"])

    env = abc_sim.make_env(task=task, prompt=episode["prompt"], render_cameras=False)
    env.reset(seed=int(episode["seed"]), randomize=True)
    rows, steps = [], 0

    def score() -> None:
        result = env.evaluate_task()
        row = {"step": steps, "old": recorded[steps]["reward"], "new": float(np.asarray(result.reward).ravel()[0]),
               "new_success": bool(np.asarray(result.success).ravel()[0])}
        if keep_metrics:
            row["metrics"] = _plain(result.metrics)
        rows.append(row)

    if 0 in recorded:
        score()
    for index, call in enumerate(calls):
        if steps >= total:
            break
        state = np.asarray(env.get_state(), dtype=np.float32)
        drift = float(np.abs(state - np.asarray(call["state"], dtype=np.float32)).max())
        if drift > 0:
            raise SystemExit(f"{job.name}: replay diverged at call {index} (step {steps}, |diff| {drift:.3g}); "
                             "the abc checkout differs from the one that recorded this run")
        for action in np.asarray(call["actions"], dtype=np.float32)[:execute]:
            if steps >= total:
                break
            env.step(action, render_obs=False)
            steps += 1
            if steps in recorded:
                score()
    env.close()
    old = [r["old"] for r in rows]
    new = [r["new"] for r in rows]
    summary = {
        "world": job.name, "seed": episode["seed"], "steps": steps, "recorded_success": episode["success"],
        "old_final": old[-1], "old_max": max(old), "new_final": new[-1], "new_max": max(new),
        "new_success": any(r["new_success"] for r in rows),
        "first_old_full": next((r["step"] for r in rows if r["old"] >= 1), None),
        "first_new_success": next((r["step"] for r in rows if r["new_success"]), None),
        "cut_short_by_old_success": bool(episode["success"]) and steps < int(episode.get("step_budget", steps + 1)),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("job_dirs", type=Path, nargs="+")
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", action="store_true", help="store the evaluator metrics at every check")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for job in args.job_dirs:
        result = rescore(job, args.task, args.metrics)
        (args.out / f"{job.name}.json").write_text(json.dumps(result))
        print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
