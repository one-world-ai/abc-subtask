"""Phase 0: random physical settles of one scrap around an upright or tipped bin.

Each try drops trash_1 from a random pose within --radius of the bin axis, lets it come to rest, and classifies it
physically: `inside` = touches only bin geoms and its centre is inside the inner wall (no margin) between floor and
rim; `on_bin_outside` = touches only the bin but not inside; `table` = touches the table/floor (or anything else).
Reports per class how many the old cylinder rule (rim radius at every height, no tilt check) and the current
evaluator accept.

    sw_settle.py --seed 0 --bin upright|tipped --tries 300 --out file.json
"""
import argparse, collections, json, time
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__, flush=True)
p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--bin", default="upright")
p.add_argument("--tries", type=int, default=300); p.add_argument("--radius", type=float, default=0.15)
p.add_argument("--out"); a = p.parse_args()
env = abc_sim.make_env(task="sweep_away_paper_scraps_from_table", render_cameras=False)
env.reset(seed=a.seed); m, d = env.model, env.data; ev = env._task_evaluator
rng = np.random.default_rng(1000 + a.seed)
bb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trash_1")
badr = ev._bin_qpos_adr; tadr = int(m.jnt_qposadr[m.body_jntadr[tb]]); tdof = int(m.jnt_dofadr[m.body_jntadr[tb]])
bin_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bb}
trash_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == tb}
# park the other scraps far away so only trash_1 is near the bin
for k, n in enumerate(ev.trash_names[1:]):
    adr = ev._trash_qpos_addrs[k + 1]; d.qpos[adr:adr + 3] = [0.35 + 0.05 * k, -0.55, 0.77]
if a.bin == "tipped":
    q = d.qpos[badr + 3:badr + 7].copy(); rot = np.empty(4); mujoco.mju_axisAngle2Quat(rot, np.array([1.0, 0, 0]), np.pi / 2)
    out = np.empty(4); mujoco.mju_mulQuat(out, rot, q); d.qpos[badr + 3:badr + 7] = out; d.qpos[badr + 2] = 0.75 + 0.10
d.qvel[:] = 0; mujoco.mj_forward(m, d)
for _ in range(int(1.5 / m.opt.timestep)): mujoco.mj_step(m, d)
d.qvel[:] = 0; base_q = d.qpos.copy()
up_z0 = float(d.xmat[bb].reshape(3, 3)[2, 1]); print("bin at rest: pos", np.round(d.xpos[bb], 3), "up_z", round(up_z0, 3), flush=True)
r_bot, r_top, y_bot, y_top = ev._bin_radius_bottom, ev._bin_radius_top, ev._bin_bottom_y, ev._bin_top_y
def local(): R = d.xmat[bb].reshape(3, 3); v = R.T @ (d.xpos[tb] - d.xpos[bb]); return float(np.hypot(v[0], v[2])), float(v[1])
def old_rule():
    rad, y = local(); return rad <= ev._bin_radius + 0.005 and y_bot - 0.005 <= y <= y_top + 0.005
def geo_inside():
    rad, y = local(); f = np.clip((y - y_bot) / (y_top - y_bot), 0, 1); return rad <= r_bot + f * (r_top - r_bot) and y_bot <= y <= y_top
counts = collections.Counter(); kept = []; t0 = time.time()
for i in range(a.tries):
    d.qpos[:] = base_q; d.qvel[:] = 0
    c = d.xpos[bb]; ang = rng.uniform(0, 2 * np.pi); rr = a.radius * np.sqrt(rng.uniform())
    d.qpos[tadr:tadr + 3] = [c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang), rng.uniform(0.78, 1.05)]
    qq = rng.normal(size=4); d.qpos[tadr + 3:tadr + 7] = qq / np.linalg.norm(qq)
    mujoco.mj_forward(m, d)
    if any(d.contact[j].dist < -0.001 and (d.contact[j].geom1 in trash_geoms or d.contact[j].geom2 in trash_geoms) for j in range(d.ncon)):
        counts["skip_penetrating"] += 1; continue
    rest = False
    for k in range(int(4.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if k * m.opt.timestep > 1.0 and k % 50 == 0 and np.linalg.norm(d.qvel[tdof:tdof + 3]) < 0.01: rest = True; break
    if not rest: counts["skip_moving"] += 1; continue
    others = set()
    for j in range(d.ncon):
        g1, g2 = d.contact[j].geom1, d.contact[j].geom2
        if g1 in trash_geoms: others.add(g2)
        elif g2 in trash_geoms: others.add(g1)
    only_bin = bool(others) and others <= bin_geoms
    cls = ("inside" if geo_inside() else "on_bin_outside") if only_bin else "table"
    r = env.evaluate_task(); new = "trash_1" in r.metrics["scraps_in_bin"][0]; old = old_rule()
    counts[(cls, "old" if old else "-", "new" if new else "-")] += 1
    rad, y = local(); kept.append(dict(cls=cls, old=old, new=new, rad=round(rad, 4), y=round(y, 4), up_z=round(float(r.metrics["bin_up_z"][0]), 3),
                                    qpos=np.round(d.qpos[tadr:tadr + 7], 6).tolist()))
print(f"seed {a.seed} bin {a.bin} up_z {up_z0:.3f}: {len(kept)} kept in {time.time() - t0:.0f}s", flush=True)
for k, v in sorted(counts.items(), key=str): print("  ", k, v)
if a.out: json.dump(dict(seed=a.seed, bin=a.bin, bin_up_z=up_z0, bin_qpos=base_q[badr:badr + 7].tolist(), counts={str(k): v for k, v in counts.items()}, kept=kept), open(a.out, "w"))
