"""Summarize complete bit-exact final-scorer replays; refuse partial runs."""
import hashlib
import json
from pathlib import Path
from statistics import mean

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent/'replay_cavity'
status=json.loads((out/'run_status.json').read_text())
assert len(status)==10 and not any(r['returncode'] for r in status), status
rows=[json.loads(p.read_text())['summary'] for p in sorted(out.glob('world-*.json'))]
assert len(rows)==10, len(rows)
source_hash=hashlib.sha256((root/'abc_sim/task_eval/lego_blocks.py').read_bytes()).hexdigest()
result={'worlds':10,'all_replays_bit_exact':True,
        'scorer_source_sha256':source_hash,
        'old_successes':sum(r['recorded_success'] for r in rows),
        'new_successes':sum(r['new_success'] for r in rows),
        'old_mean_final':mean(r['old_final'] for r in rows),
        'new_mean_final':mean(r['new_final'] for r in rows),
        'old_mean_max':mean(r['old_max'] for r in rows),
        'new_mean_max':mean(r['new_max'] for r in rows),
        'worlds_summary':rows}
(out/'scorer_source.sha256').write_text(source_hash+'\n')
(out/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='worlds_summary'},indent=2))
