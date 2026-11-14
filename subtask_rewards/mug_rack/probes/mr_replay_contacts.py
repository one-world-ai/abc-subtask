"""Replay an exp12 hang-mug world bit-exactly and list, at chosen steps, what every mug touches.

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_replay_contacts.py <job_dir> --steps 1500,2450 [--save-state <npz>]

Prints per mug: in-band / tree contact / handle-touch / other-contact masks of the scorer, the bodies it is in
contact with (MuJoCo contacts), its height above its reset height and its tilt. --save-state stores the full
integration state at each listed step (for renders / start-state experiments).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import abc_sim  # noqa: E402

print(f"abc_sim from {abc_sim.__file__}", flush=True)


def root_name(m, b):
    while int(m.body_parentid[b]) > 0 and int(m.body_parentid[b]) != 0:
        p = int(m.body_parentid[b])
        if p == 0:
            break
        b = p
    return m.body(b).name


def report(env, rest_z):
    m, d, ev = env.model, env.data, env._task_evaluator
    r = env.evaluate_task()
    met = r.metrics
    out = []
    for k, name in enumerate(ev.mug_names):
        geoms = set(ev._mug_geoms[k].tolist())
        touching = set()
        for c in d.contact[: d.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            if (a in geoms) == (b in geoms):
                continue
            o = b if a in geoms else a
            touching.add(m.body(int(m.geom_bodyid[o])).name)
        adr = ev._mug_qpos_addrs[k]
        R = d.xmat[int(ev._mug_bodies[k][0])].reshape(3, 3)
        out.append({"mug": name, "in_band": met["mug_in_band_mask"][0][k], "tree": met["mug_tree_contact_mask"][0][k],
                    "handle": met["mug_held_by_handle_mask"][0][k], "other": met["mug_other_contact_mask"][0][k],
                    "lift_cm": round(100 * float(d.qpos[adr + 2] - rest_z[k]), 1),
                    "tilt_deg": round(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))), 1),
                    "touching": sorted(touching)})
    return float(np.asarray(r.reward).ravel()[0]), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--save-state", default=None)
    a = ap.parse_args()
    want = sorted(int(s) for s in a.steps.split(","))
    trial = next(a.job.glob("trial-*"))
    ep = json.loads((trial / "episode.json").read_text())
    calls = [c for c in map(json.loads, (a.job / "server-trace.jsonl").open()) if "actions" in c]
    execute = int(ep["policy_metadata"]["execute_chunk_dim"])
    env = abc_sim.make_env(task="hang_mug_on_mug_rack", prompt=ep["prompt"], render_cameras=False)
    env.reset(seed=int(ep["seed"]), randomize=True)
    ev = env._task_evaluator
    rest_z = [float(env.data.qpos[adr + 2]) for adr in ev._mug_qpos_addrs]
    print("variants", env._last_randomization.metadata["mug_variants"])
    steps, saved = 0, {}
    for i, call in enumerate(calls):
        st = np.asarray(env.get_state(), dtype=np.float32)
        drift = float(np.abs(st - np.asarray(call["state"], dtype=np.float32)).max())
        assert drift == 0, (i, drift)
        for act in np.asarray(call["actions"], dtype=np.float32)[:execute]:
            env.step(act, render_obs=False)
            steps += 1
            if steps in want:
                rew, rows = report(env, rest_z)
                print(json.dumps({"step": steps, "reward": rew, "mugs": rows}), flush=True)
                if a.save_state:
                    s = np.zeros(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
                    mujoco.mj_getState(env.model, env.data, s, mujoco.mjtState.mjSTATE_INTEGRATION)
                    saved[f"step_{steps}"] = s
            if steps >= want[-1]:
                break
        if steps >= want[-1]:
            break
    if a.save_state:
        np.savez(a.save_state, seed=int(ep["seed"]), **saved)


if __name__ == "__main__":
    main()
