"""Phase 0: can a gripper reach over the bin? IK of the grasp site at the bin centre, per height and tilt."""
import numpy as np, mujoco, abc_sim
from abc_sim.task_eval.stagekit import ik_link6, actuator_limits, arm_ids
env = abc_sim.make_env(task="put_plastic_bottles_in_bin", render_cameras=False)
env.reset(seed=0); m, d = env.model, env.data; ev = env._task_evaluator; ba = ev._bin_qpos_adr
for arm in ("left", "right"):
    print(arm, "link_1 base", np.round(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_1")], 3) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_1") >= 0 else "?")
R_DOWN = np.diag([1.0, -1.0, -1.0])
def Rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
def Rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
c = d.qpos[ba:ba+3].copy()
for arm in ("left",):
    lim = actuator_limits(m, arm)
    for z in (0.85, 0.95, 1.05, 1.10, 1.15):
        row = []
        for tilt in (0, 20, 40, 60):
            best = 9
            for yaw in np.linspace(0, np.pi, 4, endpoint=False):
                # tilt about the axis perpendicular to the arm->bin direction: link6 z leans away from the arm
                Rw = Rz(yaw) @ R_DOWN
                Rw = (Rz(-np.pi/2) @ Rx(np.deg2rad(tilt)) @ Rz(np.pi/2)) @ Rw if arm == "left" else Rw
                site = np.array([c[0], c[1], z]); p6 = site - Rw[:, 2] * 0.1347
                T6 = np.eye(4); T6[:3, :3] = Rw; T6[:3, 3] = p6
                q, pe, re = ik_link6(m, d, arm, T6, limits=lim)
                best = min(best, pe + 0.05 * re)
            row.append(f"tilt{tilt}:{best*100:.1f}")
        print(arm, "site z", z, *row)
