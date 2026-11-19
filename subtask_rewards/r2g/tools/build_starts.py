"""Build a start set for one Real2Gym stage (G3): seeds SEED_BASE + i, each start held idle for 10 chunks of 32.

  python build_starts.py TASK STAGE OUT_ROOT [--n 100]
Writes OUT_ROOT/r2g_<task>/<stage>.npz (stagekit.save_starts) and <stage>.json (counts, reasons, CPU time).
A start is kept when it was built, is still in its stage after the idle hold, earned |return| < 0.05 and did not end.
"""
import argparse, collections, json, time
from pathlib import Path

import abc_sim
from abc_sim.task_eval.bottles_starts import idle_action
from abc_sim.task_eval.r2g_starts import SEED_BASE
from abc_sim.task_eval.stagekit import save_starts

ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("stage"); ap.add_argument("out", type=Path)
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
env = abc_sim.make_env(task=f"r2g_{a.task}__{a.stage}", render_cameras=False)
rows, why, targets, cpu = [], collections.Counter(), collections.Counter(), []
built = 0
for i in range(a.n):
    seed = SEED_BASE + i
    t0 = time.process_time()
    start, log = env.subtask.build_start(env, seed)
    cpu.append(time.process_time() - t0)
    if start is None:
        why["build: " + str(log[-1].get("why", "?"))[:60]] += 1
        continue
    built += 1
    env.reset_to_stage(seed, start)
    ev = env.subtask
    act, total, done = idle_action(env), 0.0, False
    for _ in range(10):
        for _ in range(32):
            env.step(act, render_obs=False)
        r, done = ev.chunk_reward()
        total += r
        if done:
            break
    if done or abs(total) >= 0.05 or ev.last["label"] != a.stage:
        why[f"idle: {ev.failure or ('success' if ev.success else ev.last['label'])}"] += 1
        continue
    name = ev.scene.names[start["target"]]
    targets[name] += 1
    rows.append({**start, "object": name})
out = a.out / f"r2g_{a.task}"
out.mkdir(parents=True, exist_ok=True)
if rows:
    save_starts(out / f"{a.stage}.npz", rows, stage=a.stage)
summary = dict(task=a.task, stage=a.stage, seeds=a.n, built=built, kept=len(rows), targets=dict(targets),
               dropped=dict(why), cpu_s_mean=round(sum(cpu) / len(cpu), 2), cpu_s_max=round(max(cpu), 2))
(out / f"{a.stage}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
