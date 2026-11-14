"""Compare rejected floor-rest points with the physical bin's collision-wall boundaries."""
import json
from pathlib import Path

import abc_sim
import mujoco
import numpy as np

from subtask_rewards.lego.probes.audit import rotation

root=Path(__file__).parent
rows=json.loads((root/'settle.json').read_text())['rows']
results=[]
for seed,trial in [(0,59),(1,35)]:
    row=next(r for r in rows if r['seed']==seed and r['trial']==trial)
    env=abc_sim.make_env(task='sort_lego_blocks',render_cameras=False)
    env.forget_arm_state(); env.reset(seed=seed)
    m,d,ev=env.model,env.data,env._task_evaluator
    color=row['color']; a=int(ev._block_qpos_addrs[color][0]); ba=ev._bin_qpos_addrs[color]
    d.qpos[a:a+7]=row['object_qpos']; d.qpos[ba:ba+7]=row['bin_qpos']
    mujoco.mj_forward(m,d)
    R=rotation(d.qpos[ba+3:ba+7])
    p=d.qpos[ba:ba+3]+R@np.array([0,0,row['local'][2]])
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,ev.block_names[color][0])
    groups=np.array([0,0,0,1,0,0],dtype=np.uint8)
    # mj_ray ignores fully transparent geoms: expose collision group 3 for this measurement only.
    m.geom_rgba[:,3]=1
    hits=[]
    for v in ([1,0,0],[-1,0,0],[0,1,0],[0,-1,0]):
        gid=np.zeros(1,np.int32)
        distance=mujoco.mj_ray(m,d,p,R@np.array(v,dtype=float),groups,1,bid,gid)
        hits.append({'direction':v,'distance':float(distance),
                     'geom':m.geom(int(gid[0])).name if gid[0]>=0 else None})
    center,size=ev._goal_regions[color]
    results.append({'seed':seed,'trial':trial,'local':row['local'],
                    'goal_center':center.tolist(),'goal_halfsize':size.tolist(),'wall_hits':hits})
    env.close()
(root/'edge_geometry.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
