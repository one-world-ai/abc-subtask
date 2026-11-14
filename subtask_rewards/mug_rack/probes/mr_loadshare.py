"""Candidate scorer rule: share of the upward contact force on the mug carried by its handle *pieces* (collision
pieces found by mr_counterfactual.handle_pieces), from mj_forward at the scored qpos (qvel 0). Compared with the
physical label hangs_by_handle (mr_counterfactual) on every settled pose.

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_loadshare.py <cf_seed*.json> ... --out <file>
"""
import argparse, json, os, sys
os.environ.setdefault("MUJOCO_GL", "disable")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import abc_sim  # noqa: E402
from mr_counterfactual import handle_pieces  # noqa: E402


def share(m, d, mug_geoms, hp):
    f6 = np.zeros(6)
    up_h = up = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if (a in mug_geoms) == (b in mug_geoms):
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        fz = (c.frame.reshape(3, 3).T @ f6[:3])[2]
        fz = fz if b in mug_geoms else -fz
        own = b if b in mug_geoms else a
        up += max(fz, 0.0)
        if own in hp:
            up_h += max(fz, 0.0)
    return up_h / up if up > 1e-9 else 0.0


ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
a = ap.parse_args()
env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
classes = {}
out = []
for f in a.files:
    rows = json.load(open(f))["rows"]
    cls = {r["k"]: r for r in json.load(open(f.replace("cf_", "classes5k_")))["rows"]}
    env.reset(seed=int(rows[0]["seed"]))
    m, d, ev = env.model, env.data, env._task_evaluator
    hp = set(handle_pieces(env).tolist())
    mg = set(ev._mug_geoms[0].tolist())
    q0 = d.qpos.copy()
    adr = ev._mug_qpos_addrs[0]
    for r in rows:
        d.qpos[:] = q0
        d.qpos[adr:adr + 7] = cls[r["k"]]["qpos"]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        out.append({**r, "piece_share": round(share(m, d, mg, hp), 3)})
for lab in (True, False):
    s = sorted(x["piece_share"] for x in out if x["hangs_by_handle"] == lab)
    print("hangs" if lab else "not  ", len(s), "share quantiles", [s[int(q * (len(s) - 1))] for q in (0, .05, .25, .5, .75, .95, 1)])
for t in (0.3, 0.5, 0.7):
    tp = sum(x["piece_share"] >= t and x["hangs_by_handle"] for x in out)
    fp = sum(x["piece_share"] >= t and not x["hangs_by_handle"] for x in out)
    fn = sum(x["piece_share"] < t and x["hangs_by_handle"] for x in out)
    acc_fp = sum(x["piece_share"] >= t and x["accepted"] and not x["hangs_by_handle"] for x in out)
    acc_tp = sum(x["piece_share"] >= t and x["accepted"] and x["hangs_by_handle"] for x in out)
    print(f"t={t}: share>=t hangs {tp}, not {fp}, missed {fn}; with stock rule: accepted&hangs {acc_tp}, accepted&not {acc_fp}")
json.dump(out, open(a.out, "w"), indent=1)
