"""Tree pieces of the mug tree: which are pegs (axis, tip, radius), per reset seed."""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco, numpy as np
import abc_sim
print(abc_sim.__file__)
env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
for seed in [1_200_000_000 + i for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 3)]:
    t = time.time(); env.forget_arm_state(); env.reset(seed=seed); env.evaluate_task(); dt = time.time() - t
    m, d, ev = env.model, env.data, env._task_evaluator
    print("seed", seed, "reset_s", round(dt, 2), "mugs", ev.mug_names, "axis", ev._axis_xy.round(3), "band", round(ev._hang_z_lo, 3), round(ev._hang_z_hi, 3))
    for g in ev._tree_hook_geoms:
        mid = int(m.geom_dataid[g])
        v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        c = w.mean(0); ax = np.linalg.svd(w - c)[2][0]
        if np.linalg.norm((c + ax * .02)[:2] - ev._axis_xy) < np.linalg.norm((c - ax * .02)[:2] - ev._axis_xy): ax = -ax
        t_ = (w - c) @ ax; rad = np.linalg.norm((w - c) - np.outer(t_, ax), axis=1)
        print(f"  g{g} c={c.round(3)} ax={ax.round(2)} elev={np.degrees(np.arcsin(ax[2])):.0f} len={np.ptp(t_):.3f} r_med={np.median(rad):.4f} tip={(c+ax*t_.max()).round(3)}")
    for k, b in enumerate(ev._mug_bodies):
        from abc_sim.task_eval.stagekit import object_variant
        a = ev._mug_qpos_addrs[k]
        print("  mug", k, object_variant(m, int(b[0])), d.qpos[a:a+3].round(3))
for arm in ("left", "right"):
    print(arm, "base", d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_1")].round(3) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_1")>=0 else None)
