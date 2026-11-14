"""Phase 0: a bottle held above the bin opening (gripper leaning 30-60 deg: straight-down poses above the rim are out
of reach, bt_reach.py) — does the stock scorer count it as in the bin?"""
import numpy as np, mujoco, abc_sim
from abc_sim.task_eval.stagekit import (ObjectScene, axis_angle_quat, free_joints_matching, hold_object, qmul,
                                        set_object_pose)
from abc_sim.task_eval.put_relative_starts import topdown_grasp
print(abc_sim.__file__)
env = abc_sim.make_env(task="put_plastic_bottles_in_bin", render_cameras=False)
BASE = {0: np.array([0.252, 0.31]), 1: np.array([0.252, -0.31])}
def bottle_axis(m, d, name):
    heavy = max((b for b in range(m.nbody) if m.body_mass[b] > 0 and m.body(b).name.startswith(name + "_")), key=lambda b: m.body_mass[b])
    return d.ximat[heavy].reshape(3, 3)[:, int(np.argmin(m.body_inertia[heavy]))]
out = []
for seed in range(10):
    for above in (0.03, 0.08):
        ok = False
        for tilt in (40, 50, 60, 30):
            for sign in (1, -1):
                env.reset(seed=seed); m, d = env.model, env.data; ev = env._task_evaluator; ba = ev._bin_qpos_adr
                scene = ObjectScene(m, free_joints_matching(m, r"bottle_(\d+)_joint")); k = 0
                R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[ba+3:ba+7]); rim = (d.qpos[ba:ba+3] + R.reshape(3, 3) @ [0, ev._bin_top_y, 0])[2]
                ax = bottle_axis(m, d, scene.names[k]); yaw = float(np.arctan2(ax[1], ax[0]) + np.pi / 2)
                arm = 0 if d.qpos[ba+1] > 0 else 1
                g = topdown_grasp(d, scene, k, yaw, arm)  # T_obj_link6 fixed; rotate bottle + gripper rigidly
                a = scene.qpos_adr[k]; com = d.subtree_com[scene.bodies[k]].copy()
                to_bin = d.qpos[ba:ba+2] - BASE[arm]; u = np.array([-to_bin[1], to_bin[0], 0.0]); u /= np.linalg.norm(u)
                qr = axis_angle_quat(u * sign, np.deg2rad(tilt)); Rr = np.zeros(9); mujoco.mju_quat2Mat(Rr, qr); Rr = Rr.reshape(3, 3)
                com_t = np.array([d.qpos[ba], d.qpos[ba+1], rim + above])
                set_object_pose(env, scene, k, com_t + Rr @ (d.qpos[a:a+3] - com), qmul(qr, d.qpos[a+3:a+7]))
                ok, info = hold_object(env, scene, k, g)
                if ok: break
            if ok: break
        if not ok: out.append((seed, above, "no hold", info.get("why"))); continue
        r = env.evaluate_task(); c = d.subtree_com[scene.bodies[k]]
        out.append((seed, above, f"tilt {sign*tilt}", "held, com above rim %.3f" % (c[2] - rim), "stock in-bin:", bool(r.metrics["bottle_in_bin_mask"][0, k])))
for x in out: print(*x)
