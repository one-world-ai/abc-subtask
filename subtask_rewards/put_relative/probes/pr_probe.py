import numpy as np, mujoco, abc_sim, collections
env = abc_sim.make_env(task="put_relative", render_cameras=False)
rows=[]; cnt=collections.Counter()
for s in range(30):
    env.reset(seed=s); ev=env._task_evaluator; m,d=env.model,env.data
    r=env.evaluate_task(); rew=float(np.asarray(r.reward).ravel()[0]); suc=bool(np.asarray(r.success).ravel()[0])
    g=ev._goal; md=env._last_randomization.metadata
    mb=m.jnt_bodyid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,g["mover"]+"_joint")]
    # mover size from geom aabb
    gs=[i for i in range(m.ngeom) if m.geom_bodyid[i]==mb or m.body_parentid[m.geom_bodyid[i]]==mb]
    mass=m.body_subtreemass[mb]
    objs=[o for o in md.get("objects",[])] if isinstance(md.get("objects"),list) else []
    cnt[g["direction"]]+=1
    print(s, f"rew {rew:.2f} succ {suc}", repr(env.prompt), "| n_obj", len(objs) or "?", "| mover z", round(float(d.qpos[g['mover_adr']+2]),3), "spawn_z", g["spawn_z"] and round(g["spawn_z"],3), "| mass", round(float(mass),3), "| along/perp", np.round(r.metrics["mover_along_m"],3), np.round(r.metrics["mover_perp_m"],3))
print(cnt)
print("metadata keys", list(md.keys()))
