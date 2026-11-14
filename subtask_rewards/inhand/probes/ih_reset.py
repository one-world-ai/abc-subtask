"""Phase 0: reset audit for inhand_transfer_item_to_other_side.
Per seed: reset, score; idle N_IDLE actions (object settles) while tracking the peak height, score again.
Also the deepest object contact penetration at reset (spawn overlap) and the object geometry.
usage: ih_reset.py first last [out.json]"""
import collections, json, sys, time
import numpy as np
from ih_common import make, reset, score, obj_pos, obj_extent

N_IDLE = 300
first, last = int(sys.argv[1]), int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else None
env = make()
rows, crash = [], []
t0 = time.time()
for s in range(first, last):
    try:
        scene, md = reset(env, s)
        d, m = env.data, env.model
        lo0, hi0 = obj_extent(env, scene)
        pen = 0.0
        for c in d.contact[:d.ncon]:
            if scene.owner[c.geom1][0] == 0 or scene.owner[c.geom2][0] == 0:
                pen = min(pen, float(c.dist))
        r0, s0, _ = score(env)
        a = env.get_state().astype(np.float32)
        zmax, ever = 0.0, False
        for i in range(N_IDLE):
            env.step(a, render_obs=False)
            zmax = max(zmax, float(obj_pos(env, scene)[2]))
            if i % 10 == 9:
                ever |= score(env)[1]
        r1, s1, m1 = score(env)
    except Exception as e:  # noqa: BLE001
        crash.append((s, type(e).__name__, str(e)[:100])); env = make(); continue
    p = obj_pos(env, scene); lo, hi = obj_extent(env, scene)
    spawn = env._last_randomization.object_states["task_object_joint"]["pos"]
    rows.append(dict(seed=s, cat=md["category"], var=md["variant"], side=md["side"], spawn=np.round(spawn, 3).tolist(),
                     pos=np.round(p, 3).tolist(), ext=np.round(hi - lo, 3).tolist(), zmin0=round(float(lo0[2]), 4),
                     pen0=round(pen, 4), zmax=round(zmax, 3), mass=round(float(m.body_subtreemass[scene.bodies[0]]), 3),
                     r_reset=r0, s_reset=s0, r_settled=r1, s_settled=s1, s_ever=bool(ever), cfg=bool(m1.get("goal_configured"))))
cats = collections.Counter(r["cat"] for r in rows)
print("categories", dict(cats), "sides", dict(collections.Counter(r["side"] for r in rows)))
launched = [r for r in rows if r["zmax"] > 1.0]
moved = [r for r in rows if abs(r["pos"][1] - r["spawn"][1]) > 0.05]
succ = [r for r in rows if r["s_reset"] or r["s_settled"] or r["s_ever"]]
print(f"seeds {first}-{last}: {len(rows)} ok, {len(crash)} crashes {crash[:3]}")
print(f"success at reset/settled/ever: {sum(r['s_reset'] for r in rows)}/{sum(r['s_settled'] for r in rows)}/{sum(r['s_ever'] for r in rows)} -> {[(r['seed'], r['cat']) for r in succ]}")
print(f"reward > 0.05 after settle: {[(r['seed'], round(r['r_settled'], 3)) for r in rows if r['r_settled'] > 0.05]}")
print(f"launched (peak z > 1 m): {[(r['seed'], r['cat'], r['zmax'], r['pen0']) for r in launched]}")
print(f"moved > 5 cm in y: {[(r['seed'], r['cat'], r['spawn'][1], r['pos'][1]) for r in moved]}")
print(f"unconfigured: {[r['seed'] for r in rows if not r['cfg']]}; {time.time()-t0:.0f}s")
if out:
    json.dump(dict(rows=rows, crash=crash), open(out, "w"))
