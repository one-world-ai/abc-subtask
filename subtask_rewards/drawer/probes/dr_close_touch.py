"""Phase 0: contacts with the target drawer at the first stock success of the overlay close script (seeds 0, 1, 4):
the contact flag read at control-step boundaries chatters while pushing at 8 cm/s."""
import sys, numpy as np, mujoco, abc_sim
sys.path.insert(0, "subtask_rewards/drawer/probes")
import dr_scripted as D
for seed in (0, 1, 4):
    env = abc_sim.make_env(task="close_drawer", render_cameras=False); env.reset(seed=seed); md = env._last_randomization.metadata
    ro = D.Logged(env, md["target_drawer"]); m = env.model
    orig = ro.step
    def step(n=1, every=2):
        for _ in range(n):
            orig(1)
            t = ro.trace[-1]
            if t["success"] and not getattr(ro, "done", False):
                ro.done = True; d = env.data
                cs = []
                for i in range(d.ncon):
                    c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
                    if g1 in ro.dg[md["target_drawer"]] or g2 in ro.dg[md["target_drawer"]]:
                        o = g2 if g1 in ro.dg[md["target_drawer"]] else g1
                        cs.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[o])), round(float(c.dist), 4)))
                print(seed, md["target_drawer"], "first success step", len(ro.trace), "q", round(t["q"], 4), "qd", round(t["qd"], 3), "contacts", cs[:8], "prev touch", [x["touch_target"] for x in ro.trace[-6:]])
    ro.step = step
    D.sr.script_drawer(ro, md, "close", True)
