"""Gate table (G3 starts, G4 solver, G5 reward order) from the start-set and rollout summaries.

  python gates.py STARTS_ROOT ROLLOUTS_DIR [--json validation.json]
Thresholds: .claude/skills/abc-subtask-reward/references/gates.md.
"""
import argparse, json
from pathlib import Path

STAGES = ("reach", "pick", "carry", "place")
ap = argparse.ArgumentParser()
ap.add_argument("starts", type=Path); ap.add_argument("rollouts", type=Path); ap.add_argument("--json", type=Path)
a = ap.parse_args()
out = {}
print("| task | stage | starts kept (G3 >= 80) | CPU s mean / max | solver +1 of starts / planned (G4) | hacks as expected of planned (G5) | "
      "solver - best hack return (G5 >= 0.5) | hack +1 | verdict |")
print("|---|---|---|---|---|---|---|---|---|")
for d in sorted(p for p in a.starts.iterdir() if p.is_dir() and p.name.startswith("r2g_")):
    task = d.name[4:]
    for stage in STAGES:
        sp, rp = d / f"{stage}.json", a.rollouts / f"{task}__{stage}.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        g3 = s["kept"] * 100 / s["seeds"] >= 80 and s["cpu_s_max"] <= 10
        row = dict(starts=f"{s['kept']}/{s['seeds']}", cpu=f"{s['cpu_s_mean']} / {s['cpu_s_max']}", G3=g3, dropped=s["dropped"],
                   targets=s["targets"])
        cells = [task, stage, row["starts"], row["cpu"]]
        if rp.exists():
            r = json.loads(rp.read_text())["summary"]
            sol = r["solver"]
            g4 = sol["starts"] >= 12 and sol["plus"] >= 0.4 * sol["starts"] and sol["planned"] and sol["plus"] >= 0.7 * sol["planned"]
            hacks = {k: v for k, v in r.items() if k != "solver"}
            exp_ok = all(v["planned"] >= min(10, v["starts"] / 2) and v["as_expected"] >= 0.8 * v["planned"] for v in hacks.values())
            best = max((v["mean_return"] for v in hacks.values() if v["mean_return"] is not None and v["expect"] != "any"),
                       default=0.0)
            gap = (sol["mean_return"] or 0.0) - best
            plus = {k: f"{v['plus']}/{v['starts']}" for k, v in hacks.items() if v["plus"]}
            leak = any(v["plus"] > 0.1 * v["starts"] for k, v in hacks.items() if v["expect"] != "any")
            g5 = exp_ok and gap >= 0.5 and not leak
            row.update(G4=bool(g4), G5=bool(g5), solver=f"{sol['plus']}/{sol['starts']} / {sol['plus']}/{sol['planned']}",
                       hacks={k: f"{v['as_expected']}/{v['planned']} (of {v['starts']})" for k, v in hacks.items()},
                       gap=round(gap, 2), hack_plus=plus, summary=r)
            cells += [row["solver"], "; ".join(f"{k} {v}" for k, v in row["hacks"].items()), f"{gap:.2f}",
                      ", ".join(f"{k} {v}" for k, v in plus.items()) or "none",
                      "PASS" if g3 and g4 and g5 else "FAIL " + " ".join(n for n, ok in (("G3", g3), ("G4", g4), ("G5", g5)) if not ok)]
        else:
            cells += ["", "", "", "", "G3 " + ("pass" if g3 else "FAIL")]
        out[f"{task}__{stage}"] = row
        print("| " + " | ".join(str(c) for c in cells) + " |")
if a.json:
    a.json.write_text(json.dumps(out, indent=1))
