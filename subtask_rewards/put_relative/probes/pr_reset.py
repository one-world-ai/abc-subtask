import numpy as np, abc_sim, collections, sys
env = abc_sim.make_env(task="put_relative", render_cameras=False)
N=int(sys.argv[1]); bad=[]; crash=[]; c=collections.Counter()
for s in range(N):
    try:
        env.reset(seed=s); r=env.evaluate_task()
    except RuntimeError as e:
        crash.append(s); env=abc_sim.make_env(task="put_relative", render_cameras=False); continue
    c[env._task_evaluator._goal["direction"]]+=1
    if bool(np.asarray(r.success).ravel()[0]): bad.append((s, env.prompt))
print(f"{len(bad)}/{N} resets already successful; {len(crash)} resets raised: {crash[:10]}"); print(bad[:10]); print(c)
