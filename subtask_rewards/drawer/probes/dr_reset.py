"""Phase 0: reset audit for open_drawer / close_drawer (OpenWAM overlay): score at reset, prompt, target spread,
drawer start openings, handle positions, idle drift over one 32-action chunk.

  python dr_reset.py [N=40] [seed0=0]
"""
import collections, json, sys, time
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
S0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
J = ("joint_top_drawer_low", "joint_middle_drawer_low", "joint_bottom_drawer_low")
SITES = ("top_handle", "mid_handle", "bot_handle")
out = {}
for task in ("open_drawer", "close_drawer"):
    env = abc_sim.make_env(task=task, render_cameras=False)
    m = env.model
    print(task, "dt", m.opt.timestep, "decimation", env._control_decimation, "nq", m.nq)
    adr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in J]
    sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in SITES]
    cnt = collections.Counter(); bad = []; crash = []; prompts = collections.Counter(); starts = []; drift = []; hx = []
    bad_idle = []; t0 = time.time()
    for s in range(S0, S0 + N):
        try:
            env.reset(seed=s); r = env.evaluate_task()
        except Exception as e:
            crash.append((s, type(e).__name__, str(e)[:80])); env = abc_sim.make_env(task=task, render_cameras=False); continue
        md = env._last_randomization.metadata
        cnt[md["target_drawer"]] += 1; prompts[env.prompt] += 1
        q0 = env.data.qpos[adr].copy(); starts.append(float(q0[J.index(md["target_joint"])]))
        hx.append(np.concatenate([env.data.site_xpos[i] for i in sid]).tolist())
        if bool(np.asarray(r.success).ravel()[0]) or float(np.asarray(r.reward).ravel()[0]) > 0:
            bad.append((s, float(np.asarray(r.reward).ravel()[0])))
        if env.prompt != md["prompt"] or md["target_drawer"] not in env.prompt:
            bad.append((s, "prompt mismatch " + env.prompt))
        act = env.get_init_q().astype(np.float32)
        for _ in range(32):
            env.step(act, render_obs=False)
        r2 = env.evaluate_task()
        drift.append(float(np.abs(env.data.qpos[adr] - q0).max()))
        if bool(np.asarray(r2.success).ravel()[0]) or float(np.asarray(r2.reward).ravel()[0]) > 1e-3:
            bad_idle.append((s, float(np.asarray(r2.reward).ravel()[0])))
    hx = np.array(hx).reshape(-1, 3, 3)
    res = dict(n=N, crashes=crash, nonzero_at_reset=bad, nonzero_after_idle_chunk=bad_idle, targets=dict(cnt),
               prompts=dict(prompts), target_start_m=[round(min(starts), 4), round(max(starts), 4)],
               max_idle_drift_m=max(drift),
               handle_xyz_min=np.round(hx.min(0), 3).tolist(), handle_xyz_max=np.round(hx.max(0), 3).tolist(),
               secs=round(time.time() - t0))
    print(json.dumps(res, indent=1))
    out[task] = res
json.dump(out, open(sys.argv[3] if len(sys.argv) > 3 else "/dev/null", "w"), indent=1)
