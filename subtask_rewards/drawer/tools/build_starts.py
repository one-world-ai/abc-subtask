"""Build the fixed-seed drawer start sets (100 per stage, seeds SEED_BASE[mode] + i) with the 10-chunk idle check.

  MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=<abc-subtask-drawer> python build_starts.py --mode open \
      --stage pull --out ../../../outputs/subtask_rewards/drawer/starts

Gate G3 per stage: >= 80/100 valid; <= 10 s CPU per start (build); held stages (open pull/release, close release):
>= 90 % still in the stage after 10 idle chunks, with the extra held checks of the stage table (open: both fingers
still on the handle and q within 1 cm; close release: q <= 1.5 cm). Valid = built, idle-kept, |idle return| < 0.05,
build within budget. Per-target counts are reported (approved D7).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import abc_sim
from abc_sim.task_eval.drawer_stages import DRAWERS
from abc_sim.task_eval.drawer_starts import SEED_BASE, build_start, idle_action

TASK = {"open": "open_drawer", "close": "close_drawer"}
HELD = {("open", "pull"), ("open", "release"), ("close", "release")}
_ENV = None


def one(args):
    global _ENV
    mode, stage, index = args
    if _ENV is None:
        _ENV = abc_sim.make_env(task=f"{TASK[mode]}__{stage}", render_cameras=False)
    seed = SEED_BASE[mode] + index
    t0 = time.process_time()
    start, log = build_start(_ENV, seed, mode, stage)
    rec = dict(seed=seed, built=start is not None, build_cpu_s=round(time.process_time() - t0, 3), log=log)
    if start:
        rec["target"] = DRAWERS[start["target"]]
        _ENV.reset_to_stage(seed, start)
        ev = _ENV._task_evaluator
        k = ev.target
        q0 = float(_ENV.data.qpos[ev.scene.qpos_adr[k]])
        act = idle_action(_ENV)
        total, terminal, chunk = 0.0, False, 0
        for chunk in range(10):
            for _ in range(32):
                _ENV.step(act, render_obs=False)
            r, terminal = ev.chunk_reward()
            total += r
            if terminal:
                break
        obs = ev.tracker.obs
        q1 = float(obs["q"][k])
        held_ok = True
        if (mode, stage) in HELD:
            held_ok = (bool(obs["grasped"][k]) and abs(q1 - q0) <= 0.01) if mode == "open" else q1 <= 0.015
        rec["idle"] = dict(in_stage=not terminal and ev._last["label"] == stage, return_=round(total, 5),
                           failed=ev._last["failed"], chunks=chunk + 1, q0=round(q0, 4), q1=round(q1, 4),
                           held_ok=held_ok)
        rec["valid"] = bool(rec["idle"]["in_stage"] and held_ok and abs(total) < 0.05 and rec["build_cpu_s"] <= 10)
    else:
        rec["valid"] = False
        rec["target"] = next((l["target"] for l in log if "target" in l), None)
    return start, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("open", "close"))
    ap.add_argument("--stage", required=True)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print("CHECKOUT", abc_sim.__file__, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, recs = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, 8)) as pool:
        for row, rec in pool.map(one, [(args.mode, args.stage, i) for i in range(args.count)]):
            if row and rec["valid"]:
                rows.append(row)
            recs.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "log"}), flush=True)
    built = [r for r in recs if r["built"]]
    kept = sum(r["idle"]["in_stage"] and r["idle"]["held_ok"] for r in built)
    per_target = {d: dict(tried=0, valid=0) for d in DRAWERS}
    for r in recs:
        if "target" in r:
            per_target[r["target"]]["tried"] += 1
            per_target[r["target"]]["valid"] += int(r["valid"])
    summary = dict(mode=args.mode, stage=args.stage, tried=args.count, built=len(built), valid=len(rows),
                   idle_kept=kept, idle_fraction=round(kept / len(built), 3) if built else 0.0,
                   mean_build_cpu_s=round(sum(r["build_cpu_s"] for r in recs) / len(recs), 3),
                   max_valid_build_cpu_s=max((r["build_cpu_s"] for r in recs if r["valid"]), default=0),
                   over_budget=sum(r["build_cpu_s"] > 10 for r in recs), per_target=per_target,
                   fail_reasons=sorted({(r["log"][-1].get("why", "") if r["log"] else "") for r in recs
                                        if not r["built"]}))
    held = (args.mode, args.stage) in HELD
    summary["G3"] = bool(len(rows) >= 80 and summary["max_valid_build_cpu_s"] <= 10
                         and (not held or summary["idle_fraction"] >= 0.90))

    def ragged(key):
        out = np.empty(len(rows), dtype=object)
        out[:] = [r[key] for r in rows]
        return out
    name = f"{args.mode}_{args.stage}"
    np.savez_compressed(args.out / f"{name}.npz", seed=[r["seed"] for r in rows], stage=args.stage,
                        target=[r["target"] for r in rows], arm=[r["arm"] for r in rows],
                        state=ragged("state"), rest_height=ragged("rest_height"), labels=ragged("labels"))
    (args.out / f"{name}.json").write_text(json.dumps(dict(summary=summary, per_seed=recs), indent=1, default=str))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
