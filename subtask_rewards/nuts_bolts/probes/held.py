"""Try actual two-finger held poses inside the stock region; no reward-stage code."""
from pathlib import Path
import json,os
os.environ.setdefault('MUJOCO_GL','disable')
import abc_sim,mujoco,numpy as np
from abc_sim.task_eval.stagekit.scene import ObjectScene,free_joints_matching
from abc_sim.task_eval.stagekit.holding import hold_object,set_object_pose
from abc_sim.task_eval.put_relative_starts import topdown_grasp
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
P=Path(__file__).parent;rows=[]
for kind in ('nut','bolt'):
 for attempt in range(8):
  e=abc_sim.make_env(task='sort_nuts_and_bolts',render_cameras=False);e.reset(seed=0);v=e._task_evaluator
  objs=free_joints_matching(e.model,r'nut_(\d+)_joint')+free_joints_matching(e.model,r'bolt_(\d+)_joint')
  scene=ObjectScene(e.model,objs);k=scene.names.index((v.nut_names if kind=='nut' else v.bolt_names)[0])
  b=v._nuts_bin_qpos_addr if kind=='nut' else v._bolts_bin_qpos_addr
  c=v._nuts_goal_local_center if kind=='nut' else v._bolts_goal_local_center
  qb=e.data.qpos[b:b+7];R=_quat_to_rotmat_batch(qb[3:][None])[0]
  pos=qb[:3]+R@(c+np.array([0,0,.015]))
  arm=0 if pos[1]>0 else 1
  a=scene.qpos_adr[k];q=e.data.qpos[a+3:a+7].copy();set_object_pose(e,scene,k,pos,q)
  grasp=topdown_grasp(e.data,scene,k,attempt*np.pi/8,arm,below_com_m=0)
  ok,info=hold_object(e,scene,k,grasp,rng=np.random.default_rng(attempt))
  r=e.evaluate_task();o=scene.contact_obs(e.data)
  row={'kind':kind,'attempt':attempt,'held':bool(ok),'fingers':int(o['fingers'][k]),'stock_reward':float(r.reward[0]),'info':{k:v for k,v in info.items() if k!='q'}}
  rows.append(row);print(row,flush=True)
  if ok:np.savez_compressed(P/f'held_{kind}.npz',qpos=e.data.qpos,seed=0)
  e.close()
  if ok:break
(P/'held.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))
