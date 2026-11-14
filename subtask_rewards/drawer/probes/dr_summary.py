"""Aggregate dr_scripted.py outputs: python dr_summary.py scripted_s0.json scripted_s10.json"""
import collections, json, sys
R = {}
for f in sys.argv[1:]:
    for t, d in json.load(open(f)).items():
        r = R.setdefault(t, collections.defaultdict(list))
        for k, v in d.items(): r[k] += v if isinstance(v, list) else [v]
for t, r in R.items():
    print("==", t)
    for k in ("success_script", "failure_script", "hook", "partial"):
        rows = r[k]
        if not rows: continue
        by = collections.defaultdict(lambda: [0, 0])
        for x in rows: by[x["target"]][0] += x["ever_success"]; by[x["target"]][1] += 1
        ev = sum(x["ever_success"] for x in rows); fin = sum(x["final_success"] for x in rows)
        lost = [x["seed"] for x in rows if x["ever_success"] and not x["final_success"]]
        print(f"{k:15s} ever {ev}/{len(rows)}, final {fin}, lost after success {lost}, per target (succ, n) {dict(by)}, "
              f"max success flips {max(x['success_flips'] for x in rows)}, max other-drawer q {max(x['max_others'] for x in rows):.4f}, "
              f"hand on target at first success {sum(bool(x['touch_at_first_success']) for x in rows if x['ever_success'])}/{ev}, "
              f"final q {[x['final_q'] for x in rows]}")
    for x in r["fling"]: print("fling", {k: x[k] for k in ("seed", "v", "max_q", "min_q", "final_q", "ever_success")})
    print("handmade", r["handmade"][:len(r["handmade"]) // len(sys.argv[1:])])
