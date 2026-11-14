"""Phase 0: re-create the dr_slots.py releases the stock scorer accepts but the footprint rule rejects, report where
the plate's centre and lowest point are relative to the rack footprint, and save each state (for dr_render.py)."""
import glob, json, sys
from pathlib import Path
import mujoco, numpy as np, abc_sim
from abc_sim.task_eval.dishrack import _geom_points_world
SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
env = abc_sim.make_env(task="load_plates_into_dish_rack", render_cameras=False)
for f in sorted(src.glob("seed*_rack*.json")):
    res = json.loads(f.read_text()); rv = res["rack_variant"].split("_")[-1]
    for key, ax in res["axes"].items():
        for row in ax["rows"]:
            if not (row.get("old_success") and not row.get("success")):
                continue
            axis, frac = int(key.split(":")[0]), float(key.split(":")[1])
            env.forget_arm_state()
            env.reset(seed=res["seed"], randomize=True, options={"randomization": {
                "plate_count": 1, "dish_rack_variant": res["rack_variant"], "randomize_variants": True}})
            m, d = env.model, env.data; ev = env._task_evaluator; ev.evaluate_model_data(m, d)
            rack = ev._rack_body_id; rp, Rr = d.xpos[rack].copy(), d.xmat[rack].reshape(3, 3).copy()
            lo, hi = ev._rack_footprint(d); other = 1 - axis
            u = (lo[other] + hi[other]) / 2 + frac * (hi[other] - lo[other])
            pidx = ev.plate_indices[0]; jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
            qa, da = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]); pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")
            pts = np.concatenate([_geom_points_world(m, d, g) for g in ev._plate_collision_geom_ids[pidx]])
            ploc = (pts - d.xpos[pb]) @ d.xmat[pb].reshape(3, 3); pc = (ploc.min(0) + ploc.max(0)) / 2
            radius = float(np.linalg.norm(ploc[:, :2] - pc[:2], axis=1).max())
            ez, ex = Rr[:, axis], Rr[:, 2]; Rp = np.stack([ex, np.cross(ez, ex), ez], 1); qp = np.empty(4); mujoco.mju_mat2Quat(qp, Rp.reshape(9))
            c = np.zeros(3); c[axis], c[other], c[2] = row["t"], u, row["zlow"] + radius
            d.qpos[qa:qa + 3] = rp + Rr @ c - Rp @ pc; d.qpos[qa + 3:qa + 7] = qp; d.qvel[da:da + 6] = 0
            mujoco.mj_forward(m, d)
            for _ in range(750): mujoco.mj_step(m, d)
            r = ev.evaluate_model_data(m, d); mt = r.metrics
            name = f"rack{rv}_{key.replace(':', '_')}_t{row['t']:+.4f}"
            s = np.empty(mujoco.mj_stateSize(m, SPEC)); mujoco.mj_getState(m, d, s, SPEC); np.save(out / f"{name}.npy", s)
            cen = Rr.T @ (d.xpos[pb] - rp)
            print(json.dumps({"case": name, "seed": res["seed"], "success": bool(r.success[0]),
                              "centre_rack": np.round(cen, 3).tolist(), "bottom_rack": np.round(mt["plate_bottom_rack_xy"][0][0], 3).tolist(),
                              "footprint": np.round(mt["rack_footprint_xy"][0], 3).tolist(), "normal_z": round(float(mt["plate_normal_abs_z"][0][0]), 3)}), flush=True)
