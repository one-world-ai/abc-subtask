"""Render saved audit configurations using Mesa EGL (see skill gotchas)."""
from pathlib import Path
import os
os.environ['MUJOCO_GL']='egl'
import abc_sim,mujoco,numpy as np
from PIL import Image
P=Path(__file__).parent
for source in sorted(P.glob('*.npz')):
 z=np.load(source);seed=int(z['seed']);e=abc_sim.make_env(task='sort_nuts_and_bolts',render_cameras=False);e.reset(seed=seed)
 e.data.qpos[:]=z['qpos'];mujoco.mj_forward(e.model,e.data)
 for g in range(e.model.ngeom):
  b=e.model.body(int(e.model.geom_bodyid[g])).name
  if e.model.geom_bodyid[g]==0 or b in ('gate_collision','top_camera_d405'):
   e.model.geom_rgba[g,3]=0
  if b and (b.startswith('nut_') or b.startswith('bolt_')) and int(e.model.geom_group[g])==2:
   e.model.geom_matid[g]=-1;e.model.geom_rgba[g]=[1,.05,.7,1] if b.startswith('nut_') else [.1,.7,1,1]
 cam=mujoco.MjvCamera();cam.lookat[:]=[.61,0,.79];cam.distance=1.10;cam.azimuth=90;cam.elevation=-70
 r=mujoco.Renderer(e.model,height=480,width=640);r.update_scene(e.data,camera=cam)
 Image.fromarray(r.render()).save(source.with_suffix('.png'));r.close();e.close();print(source.name,flush=True)
