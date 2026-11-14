"""Render mug_1 poses on the tree (CPU, Mesa EGL) from 3 views, mug painted orange, tree blue.

    E=../envs/mesa-cpu; __EGL_VENDOR_LIBRARY_FILENAMES=$E/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl \
    EGL_PLATFORM=surfaceless LD_LIBRARY_PATH=$E/lib PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_render.py --seed 0 --qpos x,y,z,qw,qx,qy,qz --out a.png
    ... or --classes classes.json --pick 'accepted' --outdir dir   (renders every accepted row)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import abc_sim

print(f"abc_sim from {abc_sim.__file__}", flush=True)


def render(env, qpos, path, title=""):
    m, d, ev = env.model, env.data, env._task_evaluator
    adr = ev._mug_qpos_addrs[0]
    if qpos is not None:
        d.qpos[adr: adr + 7] = qpos
    mujoco.mj_forward(m, d)
    for g in ev._mug_geoms[0].tolist() + [g for g in range(m.ngeom) if m.geom_bodyid[g] in ev._mug_bodies[0]]:
        m.geom_rgba[g] = [1.0, 0.45, 0.0, 1.0]
        m.geom_matid[g] = -1
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mug_tree")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == tid:
            m.geom_rgba[g] = [0.2, 0.4, 1.0, 1.0]
            m.geom_matid[g] = -1
    r = mujoco.Renderer(m, 360, 360)
    target = d.xpos[int(ev._mug_bodies[0][0])].copy()
    imgs = []
    for az in (0, 90, 180, 270):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = target
        cam.distance = 0.32
        cam.azimuth = az
        cam.elevation = -20
        r.update_scene(d, camera=cam)
        imgs.append(r.render())
    img = np.concatenate(imgs, axis=1)
    Image.fromarray(img).save(path)
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--qpos")
    ap.add_argument("--out")
    ap.add_argument("--classes")
    ap.add_argument("--pick", default="accepted")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--outdir")
    a = ap.parse_args()
    env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
    if a.qpos:
        env.reset(seed=a.seed)
        render(env, np.array([float(v) for v in a.qpos.split(",")]), a.out)
        return
    rows = json.load(open(a.classes))["rows"]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    count = {}
    for row in rows:
        key = "accepted" if row["accepted"] else row["class"]
        suspect = row["accepted"] != (row["class"] == "threaded")  # scorer and geometry disagree
        if a.pick == "suspect":
            if not suspect:
                continue
            key = "suspect"
        elif a.pick not in ("all", key):
            continue
        count[key] = count.get(key, 0) + 1
        if count[key] > a.max:
            continue
        env.reset(seed=int(row["seed"]))
        name = f"{key}_s{row['seed']}_{row['variant']}_k{row['k']}_{row['class']}.png"
        render(env, np.array(row["qpos"]), out / name)
        print(name, flush=True)


if __name__ == "__main__":
    main()
