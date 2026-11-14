"""CPU Mesa review renders for the physical-settle classes and inverted-bin probe."""
import json
from pathlib import Path

import abc_sim
import mujoco
import numpy as np
from PIL import Image

root = Path(__file__).parent
settles = json.loads((root/'settle.json').read_text())['rows']
handmade = json.loads((root/'handmade_before.json').read_text())
chosen = [next(r for r in settles if r['class']==kind) for kind in ('interior','outside')]
chosen.append(next(r for r in handmade if r['case']=='bin_inverted'))
chosen.extend(next(r for r in settles if r['seed']==seed and r['trial']==trial) for seed,trial in [(0,59),(1,35),(0,1)])
for label,row in zip(('interior','outside','inverted','edge0','edge1','rim'),chosen):
    env = abc_sim.make_env(task='sort_lego_blocks',render_cameras=False)
    env.forget_arm_state(); env.reset(seed=row['seed'])
    m,d,ev = env.model,env.data,env._task_evaluator
    color=row['color']; a=int(ev._block_qpos_addrs[color][0]); ba=ev._bin_qpos_addrs[color]
    d.qpos[a:a+7]=row['object_qpos']; d.qpos[ba:ba+7]=row['bin_qpos']
    mujoco.mj_forward(m,d)
    body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ev.block_names[color][0])
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g])==body:
            m.geom_matid[g]=-1; m.geom_rgba[g]=[0,1,0.8,1]
    cam=mujoco.MjvCamera(); cam.lookat=d.qpos[ba:ba+3]+[0,0,.03]
    cam.distance=.42; cam.azimuth=20 if label.startswith('edge') else 110; cam.elevation=-85 if label.startswith('edge') else -45
    renderer=mujoco.Renderer(m,height=480,width=640)
    renderer.update_scene(d,camera=cam)
    Image.fromarray(renderer.render()).save(root/f'{label}.png')
    renderer.close(); env.close()
    print(label,flush=True)
