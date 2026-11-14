"""Derive compact summaries from raw audit evidence and completed recorded-world replays."""
from pathlib import Path
import json
P=Path(__file__).parent
x=json.loads((P/'audit.json').read_text())
valid=[r for r in x['settles'] if r['valid']]
rest=[r for r in valid if r['speed']<.02]
summary={'resets':{'tried':len(x['reset']),'errors':sum('error' in r for r in x['reset']),'nonzero':sum(r.get('reward',0)>0 for r in x['reset']),'successes':sum(r.get('success',False) for r in x['reset']),'parts_min':min(r['nuts']+r['bolts'] for r in x['reset']),'parts_max':max(r['nuts']+r['bolts'] for r in x['reset'])},'handmade':[{k:r[k] for k in ['case','reward','success']} for r in x['handmade']],'settles':{'attempted':len(x['settles']),'numerically_valid':len(valid),'excluded_numerical':len(x['settles'])-len(valid),'at_rest':len(rest),'valid_still_moving':len(valid)-len(rest),'classes':[]},'stock_scorer_changed':False}
for kind in ('nut','bolt'):
    for cls in ('floor','wall_or_rim','outside'):
        selected=[r for r in rest if r['kind']==kind and (('floor' if kind+'s_bin_floor' in r['contacts'] else 'wall_or_rim' if r['support']=='bin' else 'outside')==cls)]
        summary['settles']['classes'].append({'kind':kind,'class':cls,'n':len(selected),'accepted':sum(r['accepted'] for r in selected)})
summary['held_probes']={}
for key,path in [('naive','held.json'),('wrong_frame','held_offset.json'),('native_frame','held_native_pad.json')]:
    rows=json.loads((P/path).read_text())
    summary['held_probes'][key]={'tried':len(rows),'held':sum(r['held'] for r in rows),'is_G3_validation':False}
replays=[]
for p in sorted((P/'replay').glob('world-*.json')):
    r=json.loads(p.read_text());s=r['summary']
    replays.append({**s,'checks':len(r['rows']),'score_differences':sum(abs(v['old']-v['new'])>1e-6 for v in r['rows'])})
replay_summary={'complete':len(replays)==10,'worlds_completed':len(replays),'new_success_count':sum(r['new_success'] for r in replays),'checks':sum(r['checks'] for r in replays),'score_differences':sum(r['score_differences'] for r in replays),'mean_final':sum(r['new_final'] for r in replays)/max(1,len(replays)),'mean_max':sum(r['new_max'] for r in replays)/max(1,len(replays)),'worlds':replays}
replay_summary['numerical_warning_worlds']=[p.stem for p in sorted((P/'replay').glob('world-*.log')) if 'WARNING: Nan, Inf or huge value in QACC' in p.read_text()]
summary['replay']=replay_summary
(P/'summary.json').write_text(json.dumps(summary,indent=2))
(P/'replay/summary.json').write_text(json.dumps(replay_summary,indent=2))
print(json.dumps({k:v for k,v in replay_summary.items() if k!='worlds'}),flush=True)
