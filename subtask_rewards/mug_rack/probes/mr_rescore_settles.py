"""Re-score every settled pose (settle/cf_seed*.json + classes5k) with the current scorer; table of
geometric class x old verdict x new verdict x physical label (hangs_by_handle). Also lists hole sizes per model."""
import json, glob, time, collections, numpy as np, abc_sim, mujoco
env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
t=time.time(); env.reset(seed=0); print("reset s", round(time.time()-t,2))
for v in range(19):
    env.reset(seed=0, options={"randomization": {"mug_variants": [f"mug_{v}"], "mug_count": 1, "randomize_scales": False}})
    print(f"mug_{v}", len(env._task_evaluator._hole(0)), end="; ")
print()
tab=collections.Counter()
for f in sorted(glob.glob("../simulation/outputs/subtask_rewards/mug_rack/settle/cf_seed*.json")):
    rows=json.load(open(f))["rows"]
    cls={r["k"]:r for r in json.load(open(f.replace("cf_","classes5k_")))["rows"]}
    env.reset(seed=int(rows[0]["seed"])); q0=env.data.qpos.copy(); ev=env._task_evaluator; adr=ev._mug_qpos_addrs[0]
    for r in rows:
        env.data.qpos[:]=q0; env.data.qpos[adr:adr+7]=cls[r["k"]]["qpos"]; mujoco.mj_forward(env.model, env.data)
        res=env.evaluate_task(); new=float(np.asarray(res.reward).ravel()[0])>0
        tab[(r["class"], "old_acc" if r["accepted"] else "old_rej", "new_acc" if new else "new_rej", "hangs" if r["hangs_by_handle"] else "not")]+=1
for k in sorted(tab): print(k, tab[k])
