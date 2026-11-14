"""Render saved dish-rack states on CPU (Mesa EGL), plates painted orange, rack painted light blue: four views around
the rack (along both rack axes, oblique, top). Usage (see abc-subtask-reward references/gotchas.md for the env):

    E=../envs/mesa-cpu; __EGL_VENDOR_LIBRARY_FILENAMES=$E/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl \
    EGL_PLATFORM=surfaceless LD_LIBRARY_PATH=$E/lib PYTHONPATH=$PWD ../simulation/.venv/bin/python \
    subtask_rewards/dishrack/probes/dr_render.py --seed 1928549573 --state <x.npy> --out <x.png>

``--state`` may be omitted to render the reset scene of ``--seed``.
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
from PIL import Image

import abc_sim

SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def render(env, out, size=480, dist=0.75, focus="rack"):
    m, d = env.model, env.data
    ev = env._task_evaluator
    rack = ev._rack_body_id
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        root = b
        while int(m.body_parentid[root]) != 0:
            root = int(m.body_parentid[root])
        rn = m.body(root).name
        if m.geom_group[g] == 3:
            continue
        if b == 0:
            m.geom_rgba[g, 3] = 0.0  # room walls block the views from the far side
            m.geom_matid[g] = -1
        if rn.startswith(("left_", "right_")) and not any(k in m.body(b).name for k in ("finger", "link_6")):
            m.geom_rgba[g, 3] = 0.15  # arms see-through, fingers solid
        if rn.startswith("plate"):
            m.geom_rgba[g] = (1.0, 0.45, 0.0, 1.0)
            m.geom_matid[g] = -1
        elif rn == "dishrack":
            m.geom_rgba[g] = (0.55, 0.75, 1.0, 1.0)
            m.geom_matid[g] = -1
    r = mujoco.Renderer(m, size, size)
    yaw = float(np.degrees(np.arctan2(d.xmat[rack][3], d.xmat[rack][0])))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = d.xpos[rack] + np.array([0, 0, 0.08])
    if focus == "plate":
        cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")]
    cam.distance = dist
    tiles = []
    for az, el in ((yaw + 180, -12), (yaw + 90, -12), (yaw + 135, -40), (yaw + 90, -89)):
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, cam)
        tiles.append(r.render())
    img = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    Image.fromarray(img).save(out)
    r.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--state")
    p.add_argument("--out", required=True)
    p.add_argument("--dist", type=float, default=0.75)
    p.add_argument("--focus", default="rack", choices=("rack", "plate"))
    p.add_argument("--one-plate", action="store_true", help="one-plate scene, random rack (dr_rim_grasp.py)")
    p.add_argument("--rack", default=None, help="one-plate scene with dish_rack_<n> (as dr_slots.py / dr_rim_grasp.py)")
    a = p.parse_args()
    env = abc_sim.make_env(task="load_plates_into_dish_rack", render_cameras=False)
    env.forget_arm_state()
    opts = None
    if a.rack is not None:
        opts = {"randomization": {"plate_count": 1, "dish_rack_variant": f"dish_rack_{a.rack}", "randomize_variants": True}}
    elif a.one_plate:
        opts = {"randomization": {"plate_count": 1, "randomize_variants": True}}
    env.reset(seed=a.seed, randomize=True, options=opts)
    if a.state:
        s = np.load(a.state)
        mujoco.mj_setState(env.model, env.data, s, SPEC)
    mujoco.mj_forward(env.model, env.data)
    render(env, a.out, dist=a.dist, focus=a.focus)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
