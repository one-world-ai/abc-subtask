"""Phase 0: bit-exact replay of one recorded sweep trial; per recorded check prints the evaluator's score, bin tilt
and every active scrap in the bin's frame (local +y up). Optionally saves qpos at given steps.

    sw_replay.py <job_dir> [--trial N] [--dump step,step] [--out file.json]
"""
import argparse, json
from pathlib import Path
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__, flush=True)
p = argparse.ArgumentParser(); p.add_argument("job", type=Path); p.add_argument("--trial", type=int, default=0)
p.add_argument("--dump", default=""); p.add_argument("--stride", type=int, default=1, help="batched jobs: trials interleave round-robin"); p.add_argument("--out", type=Path); a = p.parse_args()
trials = sorted(a.job.glob("trial-*")); trial = trials[a.trial]
ep = json.loads((trial / "episode.json").read_text())
rec = {r["step"]: r for r in json.loads((trial / "reward_trace.json").read_text())["trace"]}
calls = [c for c in map(json.loads, (a.job / "server-trace.jsonl").open()) if "actions" in c]
# multi-trial jobs: skip the calls of earlier trials (each trial's first call has the reset state)
if a.stride > 1:
    calls = calls[a.trial :: a.stride]
elif a.trial:
    lens = []
    for t in trials[: a.trial]:
        e = json.loads((t / "episode.json").read_text()); lens.append(e)
    ex = int(ep["policy_metadata"]["execute_chunk_dim"])
    skip = sum(-(-int(e["steps_executed"]) // ex) for e in lens); calls = calls[skip:]
ex = int(ep["policy_metadata"]["execute_chunk_dim"]); total = int(ep["steps_executed"])
dump = {int(s) for s in a.dump.split(",") if s}
env = abc_sim.make_env(task="sweep_away_paper_scraps_from_table", prompt=ep["prompt"], render_cameras=False)
env.reset(seed=int(ep["seed"]), randomize=True)
m, d = env.model, env.data; b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
rows, dumps, steps = [], {}, 0
def check():
    r = env.evaluate_task(); ev = env._task_evaluator
    R = d.xmat[b].reshape(3, 3); rel = {}
    for n in r.metrics["active_scraps"][0]:
        bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n); v = R.T @ (d.xpos[bb] - d.xpos[b])
        rel[n] = [round(float(np.hypot(v[0], v[2])), 4), round(float(v[1]), 4), round(float(d.xpos[bb][2]), 4)]
    row = dict(step=steps, old=rec.get(steps, {}).get("reward"), new=float(np.ravel(r.reward)[0]),
               success=bool(np.ravel(r.success)[0]), bin_up_z=round(float(R[2, 1]), 4),
               bin_pos=np.round(d.xpos[b], 4).tolist(), in_bin=list(r.metrics["scraps_in_bin"][0]), rel=rel)
    rows.append(row); print(json.dumps(row), flush=True)
    if steps in dump: dumps[steps] = d.qpos.copy().tolist()
if 0 in rec: check()
for i, c in enumerate(calls):
    if steps >= total: break
    st = np.asarray(env.get_state(), np.float32); drift = float(np.abs(st - np.asarray(c["state"], np.float32)).max())
    if drift > 0: raise SystemExit(f"diverged at call {i} step {steps}: {drift}")
    for act in np.asarray(c["actions"], np.float32)[:ex]:
        if steps >= total: break
        env.step(act, render_obs=False); steps += 1
        if steps in rec or steps in dump: check()
print("replayed", steps, "steps, drift 0 at", i, "calls", flush=True)
if a.out: a.out.write_text(json.dumps(dict(job=str(a.job), seed=ep["seed"], rows=rows, qpos=dumps)))
