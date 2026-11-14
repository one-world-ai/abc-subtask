"""Phase 0/1: drawer geometry — pull direction, handle bar size/position, drawer masses, arm bases, grasp-site
distance to each handle at home (reach span)."""
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__)
env = abc_sim.make_env(task="open_drawer", render_cameras=False)
for seed in (0, 1, 2):
    env.reset(seed=seed); m, d = env.model, env.data
    print("seed", seed, "target", env._last_randomization.metadata["target_drawer"])
    for lk, st, jn in (("link_top_drawer_low", "top_handle", "joint_top_drawer_low"), ("link_middle_drawer_low", "mid_handle", "joint_middle_drawer_low"), ("link_bottom_drawer_low", "bot_handle", "joint_bottom_drawer_low")):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, lk); j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        pull = d.xmat[b].reshape(3, 3) @ m.jnt_axis[j]
        s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, st)
        gs = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{a}_grasp_site") for a in ("left", "right")]
        dist = [float(np.linalg.norm(d.site_xpos[g] - d.site_xpos[s])) for g in gs]
        print(f"  {lk}: subtree mass {m.body_subtreemass[b]:.3f} kg, pull dir {np.round(pull,3)}, handle {np.round(d.site_xpos[s],3)}, grasp-site dist L/R {np.round(dist,3)}, damping {m.dof_damping[m.jnt_dofadr[j]]}, frictionloss {m.dof_frictionloss[m.jnt_dofadr[j]]}")
    for a in ("left", "right"):
        print("  ", a, "arm base", np.round(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{a}_arm')], 3) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'{a}_arm') >= 0 else "?")
