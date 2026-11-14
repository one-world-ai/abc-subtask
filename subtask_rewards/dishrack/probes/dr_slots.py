"""Phase 0/1: where can an upright plate go into this rack, and does the scorer accept it there?

For a reset scene with one plate (one scene per rack variant), is stood upright (flat normal along rack axis ``a``, a = rack x or y) at positions t along
``a`` (5 mm grid) over the rack's footprint centre line, and lowered kinematically in 5 mm steps until it would
penetrate a rack collision geom by > 1 mm. The deepest collision-free height of its lowest point above the rack
base is the insertion depth. "Inserted" = lowest point <= 3 cm above the rack base (it is down between the tines).
Each inserted pose is then released (zero velocity, arms idle) for 1.5 s and classified: upright in rack
(normal_z <= 0.55, touches rack, bottom over the footprint) or not, with the stock evaluator's success.
Three lines per axis (footprint centre line and +-25 % of the width). Slots = maximal runs of t where the released
plate scores; run length = slot play for this plate.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python subtask_rewards/dishrack/probes/dr_slots.py \
        --seeds 0 --out ../simulation/outputs/subtask_rewards/dishrack/slots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.dishrack import _geom_points_world

print("abc_sim from", abc_sim.__file__, flush=True)


def quat_from_R(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, np.float64).reshape(9))
    return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True)
    p.add_argument("--racks", default="0,1,2,3,4,5,6,7,8,9,11,12", help="dish_rack_<n> variants, one scene each")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--step", type=float, default=0.005)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    env = abc_sim.make_env(task="load_plates_into_dish_rack", render_cameras=False)
    seeds = list(map(int, a.seeds.split(",")))
    for seed, rv in [(s, r) for s in seeds for r in a.racks.split(",")]:
        env.forget_arm_state()
        env.reset(seed=seed, randomize=True, options={"randomization": {
            "plate_count": 1, "dish_rack_variant": f"dish_rack_{rv}", "randomize_variants": True}})
        m, d = env.model, env.data
        ev = env._task_evaluator
        ev.evaluate_model_data(m, d)
        meta = env._last_randomization.metadata
        rack = ev._rack_body_id
        rp, Rr = d.xpos[rack].copy(), d.xmat[rack].reshape(3, 3).copy()
        lo, hi = ev._rack_footprint(d)
        rack_pts = np.concatenate([(_geom_points_world(m, d, g) - rp) @ Rr for g in ev._rack_collision_geom_ids])
        rack_top = float(rack_pts[:, 2].max())
        pidx = ev.plate_indices[0]
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
        qa, da = int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid])
        pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")
        pgeoms = ev._plate_collision_geom_ids[pidx]
        pgset, rgset = set(pgeoms), set(ev._rack_collision_geom_ids)
        # plate geometry in its own frame
        pts = np.concatenate([_geom_points_world(m, d, g) for g in pgeoms])
        ploc = (pts - d.xpos[pb]) @ d.xmat[pb].reshape(3, 3)
        pc = (ploc.min(0) + ploc.max(0)) / 2
        depth = float(ploc[:, 2].max() - ploc[:, 2].min())
        radius = float(np.linalg.norm(ploc[:, :2] - pc[:2], axis=1).max())
        s0 = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(m, d, s0, mujoco.mjtState.mjSTATE_INTEGRATION)
        res = {"seed": seed, "rack_variant": meta["dish_rack_variant"], "plate_variant": meta["plate_variants"][0],
               "scales": env._last_randomization.scale_states, "footprint": [lo.tolist(), hi.tolist()],
               "rack_top": rack_top, "plate_depth": depth, "plate_radius": radius, "axes": {}}
        for axis, frac in [(ax, f) for ax in (0, 1) for f in (-0.25, 0.0, 0.25)]:
            other = 1 - axis
            u = (lo[other] + hi[other]) / 2 + frac * (hi[other] - lo[other])
            # plate frame: z (flat normal) -> rack axis, x -> rack z (up), y completes
            ez = Rr[:, axis]
            ex = Rr[:, 2]
            ey = np.cross(ez, ex)
            Rp = np.stack([ex, ey, ez], 1)
            qp = quat_from_R(Rp)
            rows = []
            for t in np.arange(lo[axis] + 0.01, hi[axis] - 0.01 + 1e-9, a.step):
                mujoco.mj_setState(m, d, s0, mujoco.mjtState.mjSTATE_INTEGRATION)
                c_rack = np.zeros(3)
                c_rack[axis], c_rack[other] = t, u
                best = None
                # lowest point of the upright plate is at centre - radius along world z
                for zlow in np.arange(rack_top + 0.01, 0.002, -a.step):
                    c_rack[2] = zlow + radius
                    d.qpos[qa:qa + 3] = rp + Rr @ c_rack - Rp @ pc
                    d.qpos[qa + 3:qa + 7] = qp
                    d.qvel[da:da + 6] = 0
                    mujoco.mj_forward(m, d)
                    hit = False
                    for c in d.contact[: d.ncon]:
                        g1, g2 = int(c.geom1), int(c.geom2)
                        if ((g1 in pgset and g2 in rgset) or (g2 in pgset and g1 in rgset)) and c.dist < -0.001:
                            hit = True
                            break
                    if hit:
                        break
                    best = zlow
                row = {"t": round(float(t), 4), "zlow": None if best is None else round(float(best), 4)}
                if best is not None and best <= rack_top - 0.03:  # sank between rack parts: release it there
                    c_rack[2] = best + radius
                    d.qpos[qa:qa + 3] = rp + Rr @ c_rack - Rp @ pc
                    d.qpos[qa + 3:qa + 7] = qp
                    d.qvel[da:da + 6] = 0
                    mujoco.mj_forward(m, d)
                    for _ in range(750):
                        mujoco.mj_step(m, d)
                    r = ev.evaluate_model_data(m, d)
                    mt = r.metrics
                    nz = float(mt["plate_normal_abs_z"][0][0])
                    old = (bool(mt["plate_spatially_in_rack_mask"][0][0]) and bool(mt["plate_rack_contact_mask"][0][0])
                           and not bool(mt["plate_gripper_contact_mask"][0][0]) and nz <= ev.upright_normal_z_max)
                    row.update({
                        "success": bool(r.success[0]), "old_success": bool(old),
                        "normal_z": round(nz, 3),
                        "rack_contact": bool(mt["plate_rack_contact_mask"][0][0]),
                        "over_footprint": bool(mt["plate_over_rack_footprint_mask"][0][0]),
                        "speed": round(float(np.linalg.norm(d.qvel[da:da + 3])), 4),
                    })
                rows.append(row)
            runs, cur = [], []
            for r in rows:
                if r.get("success"):
                    cur.append(r["t"])
                elif cur:
                    runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            key = f"{axis}:{frac:+.2f}"
            res["axes"][key] = {
                "positions": len(rows), "released": sum("success" in r for r in rows),
                "released_success": sum(r.get("success", False) for r in rows),
                "released_old_success": sum(r.get("old_success", False) for r in rows),
                "old_not_new": sum(r.get("old_success", False) and not r.get("success", False) for r in rows),
                "slots": len(runs),
                "slot_play_mm": [round(1000 * (x[-1] - x[0])) for x in runs],
                "slot_centres": [round((x[0] + x[-1]) / 2, 4) for x in runs],
                "rows": rows,
            }
            print(json.dumps({"seed": seed, "rack": res["rack_variant"], "plate": res["plate_variant"],
                              "depth": round(depth, 4), "radius": round(radius, 4), "rack_top": round(rack_top, 3),
                              "line": key, **{k: v for k, v in res["axes"][key].items() if k != "rows"}}),
                  flush=True)
        (a.out / f"seed{seed}_rack{rv}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
