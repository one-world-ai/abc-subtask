"""100 fixed-seed clutter candidates, exact restores and ten-chunk idle validation."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault('MUJOCO_GL','disable');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import abc_sim
from abc_sim.task_eval.clutter_starts import SEED_BASE,build_start,idle_action
_ENV=None

def one(job):
 global _ENV
 stage,i,tries=job
 if _ENV is None:_ENV=abc_sim.make_env(task='grab_specific_object_from_clutter__'+stage,render_cameras=False)
 seed=SEED_BASE+i;begin=time.monotonic();cpu=time.process_time()
 try:
  start,log=build_start(_ENV,seed,stage,tries=tries)
  record=dict(seed=seed,built=start is not None,build_s=round(time.monotonic()-begin,3),build_cpu_s=round(time.process_time()-cpu,3),log=log,skipped=bool(log and log[-1].get('skipped')),source=(log[-1].get('source') if log else None))
  record['coverage_exclusion']=not start and any(r.get('coverage_exclusion') for r in log)
  if start:
   _ENV.reset_to_stage(seed,start);ev=_ENV._task_evaluator;a=idle_action(_ENV);total=0;terminal=False
   for chunk in range(10):
    for _ in range(32):_ENV.step(a,render_obs=False)
    r,terminal=ev.chunk_reward();total+=r
    if terminal:break
   record['idle']=dict(in_stage=not terminal and ev._last['label']==stage,return_=float(total),failed=ev._last['failed'],chunks=chunk+1)
   record['valid']=record['idle']['in_stage'] and abs(total)<.05
  else:record['valid']=False
  return start,record
 except Exception as ex:return None,dict(seed=seed,built=False,valid=False,build_s=round(time.monotonic()-begin,3),error=repr(ex),coverage_exclusion=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('reach','pick','handoff','carry','place'));p.add_argument('--count',type=int,default=100);p.add_argument('--workers',type=int,default=6);p.add_argument('--tries',type=int,default=8);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-seeds',type=int,default=None,help='scan this many seeds and keep the first --count eligible (not skipped) ones');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 print('CHECKOUT',abc_sim.__file__,flush=True);rows=[];records=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  scanned=[]
  for row,r in pool.map(one,[(a.stage,i,a.tries) for i in range(a.max_seeds or a.count)]):
   scanned.append((row,r));print(json.dumps(r),flush=True)
 strict=[r for _,r in scanned[:a.count]]
 eligible=[(row,r) for row,r in scanned if not r.get('skipped')][:a.count]
 for row,r in eligible:
  records.append(r)
  if row and r['valid']:rows.append(row)
 built=sum(r['built'] for r in records);kept=sum(r.get('idle',{}).get('in_stage',False) for r in records)
 summary=dict(stage=a.stage,tried=len(records),built=built,valid=len(rows),idle_kept=kept,idle_fraction=kept/built if built else 0,mean_build_s=sum(r['build_s'] for r in records)/len(records),mean_build_cpu_s=sum(r.get('build_cpu_s',0) for r in records)/len(records),coverage_exclusions=sum(r['coverage_exclusion'] for r in records),
  strict_first=dict(seeds=len(strict),eligible=sum(not r.get('skipped') for r in strict),valid=sum(r['valid'] for r in strict)),
  sources={k:sum(r.get('source')==k for r in records if r['valid']) for k in sorted({str(r.get('source')) for r in records if r['valid']})})
 summary['G3']=len(rows)>=80 and summary['mean_build_cpu_s']<=10 and (a.stage not in ('handoff','carry','place') or summary['idle_fraction']>=.9)
 def ragged(k):
  x=np.empty(len(rows),object);x[:]=[r[k] for r in rows];return x
 np.savez_compressed(a.out/f'{a.stage}.npz',seed=[r['seed'] for r in rows],stage=a.stage,target=[r['target'] for r in rows],arm=[r['arm'] for r in rows],state=ragged('state'),rest_height=ragged('rest_height'),labels=ragged('labels'),handoff_scene=[('receiver' in r['info']) for r in rows],source=[str(r['info'].get('source','')) for r in rows])
 (a.out/f'{a.stage}.json').write_text(json.dumps(dict(summary=summary,per_seed=records,scanned=[r for _,r in scanned]),indent=2,default=str));print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
