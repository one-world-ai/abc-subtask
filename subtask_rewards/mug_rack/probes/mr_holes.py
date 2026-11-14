"""Handle-hole size of all 19 mug models (collision mesh, scale 1): inscribed circle diameter of the hole in the
handle's mid-plane (3 slices, 2-mm gaps between convex pieces closed). Compare with the peg diameter (~13.5 mm).

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_holes.py --out ../simulation/outputs/subtask_rewards/mug_rack/holes.json
"""
import argparse, json, os, sys
os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import abc_sim  # noqa: E402
from mr_classify import hole_cells  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
res = {}
for v in range(19):
    env.reset(seed=0, options={"randomization": {"mug_variants": [f"mug_{v}"], "mug_count": 1, "randomize_scales": False}})
    cells, _ = hole_cells(env)
    info = env._task_evaluator._mug_handles[0]
    res[f"mug_{v}"] = {"cells": int(len(cells)), "inscribed_d_mm": round(1000 * hole_cells.last["inscribed_d"], 1),
                       "handle_t_offset_mm": round(1000 * hole_cells.last["t0"], 1),
                       "wall_d_mm": round(2000 * info["wall"], 1)}
    print(f"mug_{v}", res[f"mug_{v}"], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
