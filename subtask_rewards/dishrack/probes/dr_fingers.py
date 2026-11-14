"""Gripper geometry in the link_6 frame: fingertip pads (collision points with z > 0.12) open and closed — where
the jaws meet (the true grasp centre), the maximum opening and the tip length beyond the stock grasp site."""
import json
import mujoco, numpy as np, abc_sim
from abc_sim.task_eval.dishrack import _geom_points_world
from abc_sim.task_eval.stagekit import GRIP_MAX, arm_ids
env = abc_sim.make_env(task="load_plates_into_dish_rack", render_cameras=False)
env.reset(seed=0); m, d = env.model, env.data
for arm in ("left", "right"):
    ids = arm_ids(m, arm)
    for name, f in (("closed", 0.0), ("open", GRIP_MAX)):
        d.qpos[ids.finger_qadr[0]], d.qpos[ids.finger_qadr[1]] = f, -f
        mujoco.mj_forward(m, d)
        p6, R6 = d.xpos[ids.link6], d.xmat[ids.link6].reshape(3, 3)
        rep = {}
        for side in ("left", "right"):
            tag = "lf" if side == "left" else "rf"
            bodies = {b for b in range(m.nbody) if m.body(b).name in (f"{arm}_{tag}_rot", f"{arm}_{tag}_down")}
            gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
            pts = (np.concatenate([_geom_points_world(m, d, g) for g in gs]) - p6) @ R6
            tip = pts[pts[:, 2] > 0.12]
            rep[side] = {"tip_lo": np.round(tip.min(0), 4).tolist(), "tip_hi": np.round(tip.max(0), 4).tolist(), "n_geoms": len(gs)}
        print(arm, name, json.dumps(rep))
