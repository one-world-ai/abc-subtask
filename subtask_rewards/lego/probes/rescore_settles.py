"""Score the exact saved settle poses after a scorer repair, without replay drift."""
import json
from collections import Counter
from pathlib import Path

import abc_sim
import mujoco

root=Path(__file__).parent
rows=json.loads((root/'settle.json').read_text())['rows']
out=[]
for seed in range(3):
    env=abc_sim.make_env(task='sort_lego_blocks',render_cameras=False)
    env.forget_arm_state(); env.reset(seed=seed)
    m,d,ev=env.model,env.data,env._task_evaluator
    q0=d.qpos.copy()
    for row in rows:
        if row['seed']!=seed: continue
        color=row['color']; a=int(ev._block_qpos_addrs[color][0]); ba=ev._bin_qpos_addrs[color]
        d.qpos[:]=q0; d.qvel[:]=0
        d.qpos[a:a+7]=row['object_qpos']; d.qpos[ba:ba+7]=row['bin_qpos']
        mujoco.mj_forward(m,d)
        r=env.evaluate_task()
        on_bin=any(any('sorting_bin' in str(b) for b in pair) for pair in row['contacts'])
        cls='moving' if row['speed']>.02 else 'bin_supported' if on_bin else 'table_or_other'
        out.append({'seed':seed,'trial':row['trial'],'class':cls,
                    'old_correct':row['correct'],'new_correct':int(r.metrics['num_correct'][0])})
    env.close()
counts=Counter((r['class'],r['old_correct']>0,r['new_correct']>0) for r in out)
result={'counts':{str(k):v for k,v in counts.items()},'rows':out}
(root/'settle_rescored.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result['counts']),flush=True)
