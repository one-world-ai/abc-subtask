"""Reproducible LEGO scorer probes. Run from the ABC checkout with MUJOCO_GL=disable."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import abc_sim
import mujoco
import numpy as np

COLORS = ("red", "yellow", "blue")


def rotation(quat):
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, quat)
    return out.reshape(3, 3)


def score(env):
    r = env.evaluate_task()
    return {"reward": float(np.asarray(r.reward).ravel()[0]),
            "success": bool(np.asarray(r.success).ravel()[0]),
            "correct": int(r.metrics["num_correct"][0]),
            "wrong": int(r.metrics["num_wrong_bin"][0])}


def make_env(seed):
    env = abc_sim.make_env(task="sort_lego_blocks", render_cameras=False)
    env.forget_arm_state()
    env.reset(seed=seed)
    return env


def reset_probe():
    env = make_env(0)
    rows = []
    for seed in range(40):
        env.forget_arm_state()
        env.reset(seed=seed)
        ev = env._task_evaluator
        row = {"seed": seed, **score(env), "counts": {c: len(ev.block_names[c]) for c in COLORS}}
        rows.append(row)
    env.close()
    return {"rows": rows, "nonzero": sum(r["reward"] != 0 for r in rows),
            "successes": sum(r["success"] for r in rows)}


def handmade_probe():
    rows = []
    for seed in range(3):
        env = make_env(seed)
        m, d, ev = env.model, env.data, env._task_evaluator
        q0 = d.qpos.copy()
        for color in COLORS:
            a = int(ev._block_qpos_addrs[color][0])
            ba = ev._bin_qpos_addrs[color]
            center, size = ev._goal_regions[color]
            for label in ("matching_center_floating", "matching_floor", "outside_side", "wrong_color_bin", "bin_tipped_90", "bin_inverted"):
                d.qpos[:] = q0
                d.qvel[:] = 0
                if label == "wrong_color_bin":
                    other = COLORS[(COLORS.index(color)+1) % 3]
                    dest = ev._bin_qpos_addrs[other]
                    local = ev._goal_regions[other][0]
                else:
                    dest, local = ba, center.copy()
                if label == "matching_floor":
                    local[2] = center[2] - size[2] + 0.01
                elif label == "outside_side":
                    local[0] = center[0] + size[0] + 0.02
                elif label in ("bin_tipped_90", "bin_inverted"):
                    theta = np.pi / 2 if label == "bin_tipped_90" else np.pi
                    d.qpos[ba+3:ba+7] = [np.cos(theta/2), np.sin(theta/2), 0, 0]
                    d.qpos[ba+2] = 0.82 if label == "bin_inverted" else 0.81
                rot = rotation(d.qpos[dest+3:dest+7])
                d.qpos[a:a+3] = d.qpos[dest:dest+3] + rot @ local
                d.qpos[a+3:a+7] = [1, 0, 0, 0]
                mujoco.mj_forward(m, d)
                rows.append({"seed": seed, "color": color, "case": label, **score(env),
                             "object_qpos": d.qpos[a:a+7].tolist(), "bin_qpos": d.qpos[dest:dest+7].tolist()})
        env.close()
    return rows


def settle_probe(tries=60):
    rows = []
    for seed in range(3):
        env = make_env(seed)
        m, d, ev = env.model, env.data, env._task_evaluator
        q0 = d.qpos.copy()
        rng = np.random.default_rng(930000000 + seed)
        color = COLORS[seed]
        a = int(ev._block_qpos_addrs[color][0])
        joint = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ev.block_names[color][0] + "_joint"))
        da = int(m.jnt_dofadr[joint])
        body = int(m.jnt_bodyid[joint])
        ba = ev._bin_qpos_addrs[color]
        center, size = ev._goal_regions[color]
        for trial in range(tries):
            d.qpos[:] = q0
            d.qvel[:] = 0
            rot = rotation(d.qpos[ba+3:ba+7])
            local = center.copy()
            local[:2] += rng.uniform(-1.8, 1.8, 2) * size[:2]
            local[2] = center[2] + size[2] + rng.uniform(0.02, 0.08)
            d.qpos[a:a+3] = d.qpos[ba:ba+3] + rot @ local
            quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
            d.qpos[a+3:a+7] = quat
            mujoco.mj_forward(m, d)
            for _ in range(int(1.5/m.opt.timestep)):
                mujoco.mj_step(m, d)
            binrot = rotation(d.qpos[ba+3:ba+7])
            local = binrot.T @ (d.qpos[a:a+3] - d.qpos[ba:ba+3])
            speed = float(np.linalg.norm(d.qvel[da:da+3]))
            # Independently classify the rest point in the bin's local frame.
            in_xy = bool(np.all(np.abs(local[:2]-center[:2]) <= size[:2]))
            cls = "interior" if in_xy and local[2] <= center[2]+size[2] else "outside"
            if speed > 0.02: cls = "moving"
            contacts = []
            for con in d.contact[:d.ncon]:
                gs = (int(con.geom1), int(con.geom2))
                bs = [int(m.geom_bodyid[g]) for g in gs]
                if body in bs:
                    contacts.append([m.body(b).name for b in bs])
            rows.append({"seed": seed, "trial": trial, "color": color, "class": cls, "speed": speed,
                         "local": local.tolist(), "contacts": contacts, **score(env),
                         "object_qpos": d.qpos[a:a+7].tolist(), "bin_qpos": d.qpos[ba:ba+7].tolist()})
        env.close()
    counts = Counter((r["class"], r["correct"] > 0) for r in rows)
    return {"counts": {str(k): v for k, v in counts.items()}, "rows": rows}


def held_probe():
    from abc_sim.task_eval.stagekit import ObjectScene, free_joints_matching, hold_object
    from abc_sim.task_eval.put_relative_starts import topdown_grasp
    rows = []
    for seed in range(6):
        env = make_env(seed)
        m,d,ev = env.model,env.data,env._task_evaluator
        entries = free_joints_matching(m, r"(?:red|yellow|blue)_lego_(\d+)_joint")
        scene = ObjectScene(m,entries)
        color = COLORS[seed % 3]
        name = ev.block_names[color][0]
        k = scene.names.index(name)
        a = int(scene.qpos_adr[k]); ba = ev._bin_qpos_addrs[color]
        center,size = ev._goal_regions[color]
        local = center.copy(); local[2] = center[2] + size[2] - 0.014
        d.qpos[a:a+3] = d.qpos[ba:ba+3] + rotation(d.qpos[ba+3:ba+7]) @ local
        mujoco.mj_forward(m,d)
        arm = 0 if d.qpos[a+1] > 0 else 1
        row = {"seed":seed,"color":color,"held":False}
        for yaw in np.linspace(0,np.pi,6,endpoint=False):
            grasp = topdown_grasp(d,scene,k,float(yaw),arm,below_com_m=0)
            ok,info = hold_object(env,scene,k,grasp,rng=np.random.default_rng(seed))
            if ok:
                row.update(held=True,**score(env)); break
            row['last_failure'] = info.get('why')
        rows.append(row)
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("reset", "handmade", "settle", "held"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tries", type=int, default=60)
    args = p.parse_args()
    print("abc_sim", abc_sim.__file__, flush=True)
    result = {"reset": reset_probe, "handmade": handmade_probe, "held": held_probe,
              "settle": lambda: settle_probe(args.tries)}[args.mode]()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}) if isinstance(result,dict)
          else f"{len(result)} handmade cases", flush=True)


if __name__ == "__main__": main()
