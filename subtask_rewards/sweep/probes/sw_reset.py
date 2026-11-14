"""Phase 0: reset audit for sweep_away_paper_scraps_from_table (scores, crashes, scene geometry)."""
import collections, sys, time
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TASK = "sweep_away_paper_scraps_from_table"
env = abc_sim.make_env(task=TASK, render_cameras=False)
cnt = collections.Counter(); bad = []; crash = []; t0 = time.time()
bin_xy, bin_top, bin_up, scales = [], [], [], collections.defaultdict(list)
for s in range(N):
    try:
        env.reset(seed=s); r = env.evaluate_task()
    except Exception as e:
        crash.append((s, type(e).__name__, str(e)[:80])); env = abc_sim.make_env(task=TASK, render_cameras=False); continue
    ev = env._task_evaluator; m = env.model; d = env.data
    act = r.metrics["active_scraps"][0]; cnt[len(act)] += 1
    if bool(np.asarray(r.success).ravel()[0]) or float(np.asarray(r.reward).ravel()[0]) > 0: bad.append((s, r.metrics["num_scraps_in_bin"]))
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
    R = d.xmat[b].reshape(3, 3); bin_up.append(R[:, 1]); bin_xy.append(d.xpos[b][:2].copy())
    bin_top.append(float(d.xpos[b][2] + ev._bin_top_y)); 
    for k, v in env._last_randomization.scale_states.items(): scales[k].append(v)
    if s < 3:
        print("seed", s, "active", act, "bin local r_bot/r_top/y_bot/y_top", np.round([ev._bin_radius_bottom, ev._bin_radius_top, ev._bin_bottom_y, ev._bin_top_y], 3),
              "bin pos", np.round(d.xpos[b], 3), "bin up(world of local y)", np.round(R[:, 1], 3))
        for n in ["brush", "dustpan"] + act:
            bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            print("  ", n, "pos", np.round(d.xpos[bb], 3), "mass", round(float(m.body_subtreemass[bb]), 4))
bin_xy = np.array(bin_xy)
print("bin xy range", np.round(bin_xy.min(0), 3), np.round(bin_xy.max(0), 3), "rim z", np.round([min(bin_top), max(bin_top)], 3),
      "min up_z", round(float(np.min(np.array(bin_up)[:, 2])), 4))
print("scales", {k: (round(min(v), 3), round(max(v), 3)) for k, v in scales.items()})
print(f"{len(bad)}/{N} resets score >0: {bad[:10]}; crashes {len(crash)}: {crash[:5]}; counts {dict(cnt)}; {time.time()-t0:.0f}s")
