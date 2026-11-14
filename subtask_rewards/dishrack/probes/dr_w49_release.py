"""Phase 0: exp12 world 49 (the only upright-in-slot plate) — the success check fired while the gripper still held
the plate (contact flicker). From the saved final state, open the grippers without moving the arms (300 steps), then
raise the arm's wrist by stepping joint 2/3 targets slowly: does the plate stay in the slot and score?"""
import json, sys
import mujoco, numpy as np, abc_sim
sys.path.insert(0, "subtask_rewards/dishrack/probes")
from dr_replay_final import plate_info, score, SPEC
O = "../simulation/outputs/subtask_rewards/dishrack/replay_final"
env = abc_sim.make_env(task="load_plates_into_dish_rack", render_cameras=False)
env.forget_arm_state(); env.reset(seed=720949075, randomize=True); env.evaluate_task()
m, d = env.model, env.data
for name in ("final", "hold"):
    s = np.load(f"{O}/world-49_{name}_state.npy"); mujoco.mj_setState(m, d, s, SPEC); mujoco.mj_forward(m, d)
    print(name, "ctrl", np.round(d.ctrl, 3).tolist())
    base = np.array([d.ctrl[i] for i in env._ctrl_indices], np.float32)
    act = base.copy()
    for i in env._gripper_set: act[i] = 1.0
    out = []
    for k in range(1, 401):
        a = act.copy()
        if k > 150:  # lift: slowly move both arms toward home
            f = min(1.0, (k - 150) / 250)
            home = np.asarray(env.get_init_q(), np.float32)
            for i in range(14):
                if i not in env._gripper_set: a[i] = (1 - f) * base[i] + f * home[i]
        env.step(a, render_obs=False)
        if k % 50 == 0:
            p = plate_info(env)[0]
            out.append((k, score(env), p["pos_rack"], p["normal_rack"], p["speed"], p["n_touch"]))
    for x in out: print(name, *x)
    s1 = np.empty_like(s); mujoco.mj_getState(m, d, s1, SPEC); np.save(f"{O}/world-49_{name}_release_state.npy", s1)
