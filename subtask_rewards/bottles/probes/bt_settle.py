"""Phase 0: drop one bottle from random poses over the bin, let it settle, compare the stock in-bin flag with
physics categories. Also: one held bottle above the bin (does held count?)."""
import sys, json, collections
import numpy as np, mujoco, abc_sim
from abc_sim.task_eval.stagekit import ObjectScene, quat_of, subtree
print(abc_sim.__file__, flush=True)
seed, tries = int(sys.argv[1]), int(sys.argv[2])
env = abc_sim.make_env(task="put_plastic_bottles_in_bin", render_cameras=False)
env.reset(seed=seed); m, d = env.model, env.data; ev = env._task_evaluator
base = d.qpos.copy(); bq = d.qvel.copy() * 0
k = 0; name = ev.bottle_names[k]; adr = int(ev._bottle_qpos_addrs[k])
jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"); dadr = int(m.jnt_dofadr[jid])
bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name); bot = subtree(m, bb)
binb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bin_container"); bins = subtree(m, binb)
table = {i for i in range(m.nbody) if "table" in (m.body(i).name or "")}
ba = ev._bin_qpos_adr
def bin_frame():
    R = np.zeros(9); mujoco.mju_quat2Mat(R, d.qpos[ba+3:ba+7]); return d.qpos[ba:ba+3].copy(), R.reshape(3, 3)
p0, R0 = bin_frame(); rim = (p0 + R0 @ [0, ev._bin_top_y, 0])[2]; floor = (p0 + R0 @ [0, ev._bin_bottom_y, 0])[2]
print("bin rim z", round(rim, 3), "floor z", round(floor, 3), "r_top", round(ev._bin_radius_top, 3), "bottle reach", round(float(ev._bottle_reaches[k]), 3))
def touching():
    t = collections.Counter()
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for x, y in ((b1, b2), (b2, b1)):
            if x in bot and y not in bot: t["bin" if y in bins else "table" if y == 0 else "other"] += 1
    return t
rng = np.random.default_rng(seed); rows = []
for i in range(tries):
    d.qpos[:] = base; d.qvel[:] = 0
    # park other bottles far away? keep them: they are on the table, outside the bin
    pos = np.array([p0[0] + rng.normal(0, 0.06), p0[1] + rng.normal(0, 0.06), rim + rng.uniform(0.0, 0.12)])
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    d.qpos[adr:adr+3] = pos; d.qpos[adr+3:adr+7] = q
    mujoco.mj_forward(m, d)
    if any(c.dist < -0.001 for c in d.contact[:d.ncon] if m.geom_bodyid[c.geom1] in bot or m.geom_bodyid[c.geom2] in bot):
        continue
    ctrl = d.ctrl.copy()
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    v = np.linalg.norm(d.qvel[dadr:dadr+3])
    if v > 0.01: continue
    r = ev.evaluate_qpos_batch(d.qpos[None].copy()); flag = bool(r.metrics["bottle_in_bin_mask"][0, k])
    p, R = bin_frame(); com = d.subtree_com[bb]
    # lowest/highest point of the bottle's collision geoms (AABB of rbound spheres is too coarse): use geom xpos ± rbound
    gz = [d.geom_xpos[g][2] for g in range(m.ngeom) if m.geom_bodyid[g] in bot and (m.geom_contype[g] or m.geom_conaffinity[g])]
    t = touching(); tipped = float((R @ [0, 1, 0])[2]) < 0.9
    cat = ("bin_tipped" if tipped else "on_table" if t["table"] and not t["bin"] else "rim_and_table" if t["table"] else
           "inside" if com[2] < rim else "on_rim" if t["bin"] else "other")
    rows.append(dict(cat=cat, flag=flag, com_above_rim=float(com[2] - rim), qpos=d.qpos.tolist()))
c = collections.Counter((r["cat"], r["flag"]) for r in rows)
for key in sorted(c): print(key, c[key])
onrim = [r["com_above_rim"] for r in rows if r["cat"] == "on_rim"]
if onrim: print("on_rim com above rim: min/median/max", np.round([min(onrim), np.median(onrim), max(onrim)], 3))
json.dump(rows, open(f"subtask_rewards/bottles/probes/settle_seed{seed}.json", "w"))
