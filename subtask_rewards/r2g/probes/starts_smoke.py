"""Build a few starts per stage and hold each idle for 10 chunks of 32 (G3 smoke; the full run is tools/build_starts.py).
  python starts_smoke.py task1,task2 [n=6]"""
import collections, sys, time
import numpy as np
import abc_sim
from abc_sim.task_eval.bottles_starts import idle_action
from abc_sim.task_eval.r2g_starts import SEED_BASE

tasks, n = sys.argv[1].split(","), int(sys.argv[2]) if len(sys.argv) > 2 else 6
for t in tasks:
    for st in ("reach", "pick", "carry", "place"):
        env = abc_sim.make_env(task=f"r2g_{t}__{st}", render_cameras=False)
        ok = idle_ok = 0
        why, tg = collections.Counter(), collections.Counter()
        t0 = time.time()
        for i in range(n):
            seed = SEED_BASE + i
            start, log = env.subtask.build_start(env, seed)
            if start is None:
                why[str(log[-1].get("why", "?"))[:70]] += 1
                continue
            ok += 1
            tg[env.subtask.scene.names[start["target"]]] += 1
            env.reset_to_stage(seed, start)
            ev = env.subtask
            a, tot, d = idle_action(env), 0.0, False
            for c in range(10):
                for _ in range(32):
                    env.step(a, render_obs=False)
                r, d = ev.chunk_reward()
                tot += r
                if d:
                    break
            good = (not d) and abs(tot) < 0.05 and ev.last["label"] == st
            idle_ok += good
            if not good:
                why[f"idle: {ev.failure or ev.last['label']} r={tot:.2f}"] += 1
        print(t[:24], st, f"built {ok}/{n} idle-stable {idle_ok}/{ok} {time.time() - t0:.0f}s", dict(tg), dict(why), flush=True)
        env.close()
