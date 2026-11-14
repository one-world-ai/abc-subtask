"""Print the G4/G5 table of every scripted summary_<mode>_<stage>.json in a directory (no sim imports)."""
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
for f in sorted(d.glob("summary_*.json")):
    S = json.loads(f.read_text())["summary"]
    if not S:
        continue
    print(f"== {f.stem[8:]}  G4={S['G4']} G5={S.get('G5')} margin={S.get('margin')}")
    for name, v in S.items():
        if isinstance(v, dict):
            pt = " ".join(f"{t[0]}:{x['plus']}/{x['tried']}" for t, x in v["per_target"].items())
            print(f"  {name:22s} {v['kind']:6s} planned {v['planned']:2d}/{v['tried']:2d}  +1 {v['plus']:2d}  -1 {v['minus']:2d}"
                  f"  0 {v['zero']:2d}  as_expected {v['as_expected']:2d}  mean_return {v['mean_return']:+.3f}  [{pt}]")
