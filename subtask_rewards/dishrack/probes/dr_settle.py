"""Phase 0 audit: random physical settles. One plate per scene, dropped from a random pose (random orientation, centre
within the rack footprint + 8 cm, 2-12 cm above the rack top), 2 s of physics, arms idle. Each rest pose is
classified geometrically: upright (normal_z <= 0.55) / tilted / flat; touches rack / table; centre and lowest point
over the rack footprint. Counts how many the stock rule (old) and the footprint rule (new) accept per class, and
saves the accepted ones' states for rendering.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python subtask_rewards/dishrack/probes/dr_settle.py \
        --racks 0,1,2 --tries 150 --out ../simulation/outputs/subtask_rewards/dishrack/settle
"""
import argparse, json
from collections import Counter
from pathlib import Path
import mujoco, numpy as np, abc_sim
from abc_sim.task_eval.dishrack import _geom_points_world
SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
print("abc_sim from", abc_sim.__file__, flush=True)
p = argparse.ArgumentParser(); p.add_argument("--racks", required=True); p.add_argument("--tries", type=int, default=150)
p.add_argument("--seed", type=int, default=1); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
env = abc_sim.make_env(task="load_plates_into_dish_rack", render_cameras=False)
for rv in a.racks.split(","):
    env.forget_arm_state()
    env.reset(seed=a.seed, randomize=True, options={"randomization": {"plate_count": 1, "dish_rack_variant": f"dish_rack_{rv}", "randomize_variants": True}})
    m, d = env.model, env.data; ev = env._task_evaluator; ev.evaluate_model_data(m, d)
    rack = ev._rack_body_id; rp, Rr = d.xpos[rack].copy(), d.xmat[rack].reshape(3, 3).copy()
    lo, hi = ev._rack_footprint(d)
    top = max(float(((_geom_points_world(m, d, g) - rp) @ Rr)[:, 2].max()) for g in ev._rack_collision_geom_ids)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint"); qa, da = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate"); pg = set(ev._plate_collision_geom_ids[ev.plate_indices[0]])
    s0 = np.empty(mujoco.mj_stateSize(m, SPEC)); mujoco.mj_getState(m, d, s0, SPEC)
    rng = np.random.default_rng(int(rv)); cnt = Counter(); kept = []
    for i in range(a.tries):
        mujoco.mj_setState(m, d, s0, SPEC)
        c = np.array([rng.uniform(lo[0] - 0.08, hi[0] + 0.08), rng.uniform(lo[1] - 0.08, hi[1] + 0.08), top + rng.uniform(0.1, 0.2)])
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        d.qpos[qa:qa + 3] = rp + Rr @ c; d.qpos[qa + 3:qa + 7] = q; d.qvel[da:da + 6] = 0
        mujoco.mj_forward(m, d)
        for _ in range(1000): mujoco.mj_step(m, d)
        r = ev.evaluate_model_data(m, d); mt = r.metrics
        nz = float(mt["plate_normal_abs_z"][0][0]); rackc = bool(mt["plate_rack_contact_mask"][0][0])
        table = any(((int(x.geom1) in pg) != (int(x.geom2) in pg)) and int(m.geom_bodyid[int(x.geom2) if int(x.geom1) in pg else int(x.geom1)]) in
                    {b for b in range(m.nbody) if m.body(b).name in ("play_table", "table", "world")} for x in d.contact[:d.ncon])
        over = bool(mt["plate_over_rack_footprint_mask"][0][0]); speed = float(np.linalg.norm(d.qvel[da:da + 3]))
        old = (bool(mt["plate_spatially_in_rack_mask"][0][0]) and rackc and nz <= ev.upright_normal_z_max)
        cls = ("upright" if nz <= 0.55 else "flat" if nz >= 0.75 else "tilted") + ("+rack" if rackc else "") + ("+table" if table else "") + ("+over" if over else "+outside")
        cnt[(cls, "old" if old else "-", "new" if bool(r.success[0]) else "-")] += 1
        if old or r.success[0]:
            s = np.empty_like(s0); mujoco.mj_getState(m, d, s, SPEC); np.save(a.out / f"rack{rv}_{i:03d}.npy", s)
            kept.append({"i": i, "cls": cls, "old": bool(old), "new": bool(r.success[0]), "nz": round(nz, 3), "speed": round(speed, 3),
                         "centre": np.round(Rr.T @ (d.xpos[pb] - rp), 3).tolist(), "bottom": np.round(mt["plate_bottom_rack_xy"][0][0], 3).tolist()})
    out = {"rack": rv, "seed": a.seed, "footprint": [lo.tolist(), hi.tolist()], "counts": [[*k, v] for k, v in sorted(cnt.items())], "kept": kept}
    (a.out / f"rack{rv}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"rack": rv, "counts": out["counts"]}), flush=True)
