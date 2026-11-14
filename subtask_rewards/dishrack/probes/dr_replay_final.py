"""Phase 0: replay an exp12 plate world bit-exactly to its recorded success step, save the state, then keep simulating
to see whether the "success" survives: (hold) the last recorded action for 300 steps, (retreat) both arms back home
with open grippers for 300 steps. Reports per check: success, plate speed, plate pose in the rack frame and which
bodies the plate touches.

    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python subtask_rewards/dishrack/probes/dr_replay_final.py \
        --out ../simulation/outputs/subtask_rewards/dishrack/replay_final <job_dir> [...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import abc_sim

print("abc_sim from", abc_sim.__file__, flush=True)
SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def plate_info(env) -> dict:
    m, d = env.model, env.data
    ev = env._task_evaluator
    rack = ev._rack_body_id
    Rr = d.xmat[rack].reshape(3, 3)
    out = []
    for i in ev.plate_indices:
        name = "plate" if i == 0 else f"plate_{i}"
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint" if i == 0 else f"plate_joint_{i}")
        dof = int(m.jnt_dofadr[j])
        pg = {g for g in range(m.ngeom) if _root_is(m, int(m.geom_bodyid[g]), b)}
        touch = {}
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in pg) == (g2 in pg):
                continue
            other = g2 if g1 in pg else g1
            ob = int(m.geom_bodyid[other])
            while int(m.body_parentid[ob]) not in (0,) and m.body(ob).name not in ("dishrack", "table") \
                    and not m.body(ob).name.startswith(("left_", "right_", "plate")):
                ob = int(m.body_parentid[ob])
            key = m.body(ob).name or f"body{ob}"
            if other in ev._rack_geom_ids:
                key = "rack"
            elif other in ev._gripper_geom_ids:
                key = "gripper"
            loc = Rr.T @ (np.asarray(c.pos) - d.xpos[rack])
            touch.setdefault(key, []).append(np.round(loc, 3).tolist())
        Rp = d.xmat[b].reshape(3, 3)
        out.append({
            "plate": name,
            "pos_rack": np.round(Rr.T @ (d.xpos[b] - d.xpos[rack]), 4).tolist(),
            "normal_rack": np.round(Rr.T @ Rp[:, 2], 3).tolist(),
            "speed": round(float(np.linalg.norm(d.qvel[dof:dof + 3])), 4),
            "angspeed": round(float(np.linalg.norm(d.qvel[dof + 3:dof + 6])), 3),
            "touch": {k: v[:6] for k, v in touch.items()},
            "n_touch": {k: len(v) for k, v in touch.items()},
        })
    return out


def _root_is(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return False


def score(env) -> tuple[bool, float]:
    r = env.evaluate_task()
    return bool(np.asarray(r.success).ravel()[0]), float(np.asarray(r.reward).ravel()[0])


def run(job: Path, out: Path) -> dict:
    trial = next(job.glob("trial-*"))
    ep = json.loads((trial / "episode.json").read_text())
    calls = [c for c in map(json.loads, (job / "server-trace.jsonl").open()) if "actions" in c]
    execute = int(ep["policy_metadata"]["execute_chunk_dim"])
    total = int(ep["steps_executed"])
    env = abc_sim.make_env(task="load_plates_into_dish_rack", prompt=ep["prompt"], render_cameras=False)
    env.reset(seed=int(ep["seed"]), randomize=True)
    meta = getattr(env._last_randomization, "metadata", {})
    steps, last = 0, None
    for idx, call in enumerate(calls):
        if steps >= total:
            break
        st = np.asarray(env.get_state(), np.float32)
        drift = float(np.abs(st - np.asarray(call["state"], np.float32)).max())
        assert drift == 0, f"drift {drift} at call {idx}"
        for a in np.asarray(call["actions"], np.float32)[:execute]:
            if steps >= total:
                break
            env.step(a, render_obs=False)
            last = a
            steps += 1
            if steps % 50 == 0 and steps >= total - 200:
                score(env)  # the evaluator only accumulates at checks; keep the recorded cadence near the end
    s0 = np.empty(mujoco.mj_stateSize(env.model, SPEC))
    mujoco.mj_getState(env.model, env.data, s0, SPEC)
    res = {"world": job.name, "seed": ep["seed"], "steps": steps, "dt": float(env.model.opt.timestep),
           "variants": {k: meta.get(k) for k in ("dish_rack_variant", "plate_variants")},
           "scales": getattr(env._last_randomization, "scale_states", {}),
           "final": {"success": score(env), "plates": plate_info(env)}}
    np.save(out / f"{job.name}_final_state.npy", s0)
    for mode in ("hold", "retreat"):
        mujoco.mj_setState(env.model, env.data, s0, SPEC)
        mujoco.mj_forward(env.model, env.data)
        act = last.copy()
        if mode == "retreat":
            act = np.asarray(env.get_init_q(), np.float32).copy()
            act[6] = act[13] = 1.0
        rows = []
        for k in range(1, 301):
            if mode == "retreat":  # first open the grippers in place for 40 steps, then go home
                a = last.copy() if k <= 40 else act
                if k <= 40:
                    a[6] = a[13] = 1.0
            else:
                a = act
            env.step(a, render_obs=False)
            if k % 50 == 0:
                rows.append({"k": k, "success": score(env), "plates": plate_info(env)})
        res[mode] = rows
        s1 = np.empty_like(s0)
        mujoco.mj_getState(env.model, env.data, s1, SPEC)
        np.save(out / f"{job.name}_{mode}_state.npy", s1)
    env.close()
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jobs", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for job in a.jobs:
        r = run(job, a.out)
        (a.out / f"{job.name}.json").write_text(json.dumps(r, indent=1, default=float))
        print(json.dumps({"world": r["world"], "variants": r["variants"], "final": r["final"]["success"],
                          "hold": [x["success"] for x in r["hold"]], "retreat": [x["success"] for x in r["retreat"]]}),
              flush=True)


if __name__ == "__main__":
    main()
