"""Phase 0: reset audit for put_plastic_bottles_in_bin (scores, counts, crashes, geometry)."""
import collections, sys, time
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
env = abc_sim.make_env(task="put_plastic_bottles_in_bin", render_cameras=False)
cnt = collections.Counter(); bad = []; crash = []; t0 = time.time()
for s in range(N):
    try:
        env.reset(seed=s); r = env.evaluate_task()
    except Exception as e:
        crash.append((s, type(e).__name__, str(e)[:80])); env = abc_sim.make_env(task="put_plastic_bottles_in_bin", render_cameras=False); continue
    ev = env._task_evaluator
    cnt[len(ev.bottle_names)] += 1
    if bool(np.asarray(r.success).ravel()[0]) or float(np.asarray(r.reward).ravel()[0]) > 0: bad.append((s, r.metrics["num_bottles_in_bin"]))
    if s < 3:
        m = env.model; d = env.data
        print("seed", s, "bottles", ev.bottle_names, "bin cone (bot y, top y, r bot, r top)",
              np.round([ev._bin_bottom_y, ev._bin_top_y, ev._bin_radius_bottom, ev._bin_radius_top], 3),
              "bin pos", np.round(d.qpos[ev._bin_qpos_adr:ev._bin_qpos_adr+3], 3), "reaches", np.round(ev._bottle_reaches, 3))
        for n, a in zip(ev.bottle_names, ev._bottle_qpos_addrs):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            print("  ", n, "pos", np.round(d.qpos[a:a+3], 3), "com", np.round(d.subtree_com[b], 3), "mass", round(float(m.body_subtreemass[b]), 3))
print(f"{len(bad)}/{N} resets score >0: {bad[:10]}; crashes {len(crash)}: {crash[:5]}; counts {dict(cnt)}; {time.time()-t0:.0f}s")
