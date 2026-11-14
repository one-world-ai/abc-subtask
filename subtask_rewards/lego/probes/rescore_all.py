"""Re-score the ten exp12 LEGO worlds in three independent CPU workers."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'replay_cavity')
parser.add_argument('--workers',type=int,default=3)
args=parser.parse_args()
out=args.out.resolve()
out.mkdir(parents=True,exist_ok=True)
source_hash=hashlib.sha256((root/'abc_sim/task_eval/lego_blocks.py').read_bytes()).hexdigest()
hash_file=out/'scorer_source.sha256'
if hash_file.exists() and hash_file.read_text().strip()!=source_hash:
    raise SystemExit('Scorer changed: choose a fresh --out directory rather than reusing stale replay results')
hash_file.write_text(source_hash+'\n')
jobs = sorted((root.parent/'simulation/outputs/openwam/experiment12/step-57000/tasks/sort_lego_blocks').glob('world-*'))
script = root/'.claude/skills/abc-scorer-audit/scripts/rescore_replay.py'

def run(job):
    if (out/f'{job.name}.json').exists():
        return {'world':job.name,'returncode':0,'reused':True}
    with (out/f'{job.name}.log').open('w') as log:
        result = subprocess.run([sys.executable,'-u',str(script),'--task','sort_lego_blocks',
                                 '--out',str(out),str(job)],stdout=log,stderr=subprocess.STDOUT,cwd=root)
    row = {'world':job.name,'returncode':result.returncode}
    print(json.dumps(row),flush=True)
    return row

with ThreadPoolExecutor(max_workers=args.workers) as pool:
    results=list(pool.map(run,jobs))
(out/'run_status.json').write_text(json.dumps(results,indent=2))
raise SystemExit(any(r['returncode'] for r in results))
