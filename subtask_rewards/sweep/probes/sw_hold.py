"""Phase 3 pilot: active scraps at reset, body origin vs CoM, and a top-down hold of the nearest scrap (seeds 970M+i)."""
import sys, numpy as np, mujoco, abc_sim
from abc_sim.task_eval.stagekit import ObjectScene, free_joints_matching, hold_object, set_object_pose, GRIP_MAX
from abc_sim.task_eval.put_relative_starts import topdown_grasp
print(abc_sim.__file__, "GRIP_MAX", GRIP_MAX)
env = abc_sim.make_env(task="sweep_away_paper_scraps_from_table", render_cameras=False)
for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 6):
    seed = 970_000_000 + i
    env.forget_arm_state(); env.reset(seed=seed, randomize=True)
    m, d = env.model, env.data
    ev = env._task_evaluator
    sc = ObjectScene(m, free_joints_matching(m, r"trash_(\d+)_jnt"))
    act = ev._active_trash_mask[0]
    com = d.subtree_com[sc.bodies]; org = np.array([d.qpos[a:a+3] for a in sc.qpos_adr])
    c = sc.contact_obs(d)
    k = int(np.flatnonzero(act)[np.argmin(c["dist"][act])])
    binp = d.qpos[ev._bin_qpos_adr:ev._bin_qpos_adr+3]
    print(f"seed {seed}: active {act.sum()} names {[n for n,a in zip(sc.names,act) if a]} target {sc.names[k]} xy {np.round(com[k,:2],3)} z {com[k,2]:.3f} "
          f"origin-com {np.linalg.norm(org[k]-com[k])*1000:.1f} mm  bin {np.round(binp,3)} dist {c['dist'][k]:.3f}")
    arm = 0 if com[k,1] > 0 else 1
    for yaw in (0.0, np.pi/2):
        env.forget_arm_state(); env.reset(seed=seed, randomize=True)
        g = topdown_grasp(env.data, sc, k, yaw, arm, below_com_m=0.0)
        a = sc.qpos_adr[k]; p = env.data.qpos[a:a+3].copy(); p[2] += 0.08
        set_object_pose(env, sc, k, p, env.data.qpos[a+3:a+7].copy())
        ok, info = hold_object(env, sc, k, g)
        info.pop("q", None)
        f = [env.data.qpos[x] for x in __import__("abc_sim.task_eval.stagekit", fromlist=["arm_ids"]).arm_ids(env.model, ["left","right"][arm]).finger_qadr] if False else None
        print("   yaw", round(yaw,2), ok, info)
