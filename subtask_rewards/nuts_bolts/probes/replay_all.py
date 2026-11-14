"""Run independent recorded worlds in CPU processes; keep every exit status and log."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,subprocess,sys
R=Path(__file__).resolve().parents[3];P=Path(__file__).parent/'replay';P.mkdir(exist_ok=True)
BASE=R.parent/'simulation/outputs/openwam/experiment12/step-57000/tasks/sort_nuts_and_bolts'
def run(job):
 output=P/(job.name+'.json')
 if output.exists():return {'world':job.name,'exit':0,'already_complete':True}
 cmd=[sys.executable,str(R/'.claude/skills/abc-scorer-audit/scripts/rescore_replay.py'),'--task','sort_nuts_and_bolts','--out',str(P),'--metrics',str(job)]
 with (P/(job.name+'.log')).open('w') as f:
  result=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
 row={'world':job.name,'exit':result.returncode};print(json.dumps(row),flush=True);return row
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=9) as ex:rows=list(ex.map(run,sorted(BASE.glob('world-*'))))
 (P/'exits.json').write_text(json.dumps(rows,indent=2))
 print(json.dumps(rows),flush=True)
