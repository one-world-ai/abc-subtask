"""Phase 1: how did a recorded episode move the scraps? Bit-exact replay of one trial; per scrap, which bodies it
touched (gripper fingers, brush, dustpan, bin, table) and its max height; brush/dustpan displacement and lift.

    sw_tooluse.py <job_dir> [<job_dir> ...]   (prints one JSON line per job)
"""
import collections, json, sys
from pathlib import Path
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__, flush=True)
for job in map(Path, sys.argv[1:]):
    trial = next(job.glob("trial-*")); ep = json.loads((trial / "episode.json").read_text())
    calls = [c for c in map(json.loads, (job / "server-trace.jsonl").open()) if "actions" in c]
    ex = int(ep["policy_metadata"]["execute_chunk_dim"]); total = int(ep["steps_executed"])
    env = abc_sim.make_env(task="sweep_away_paper_scraps_from_table", prompt=ep["prompt"], render_cameras=False)
    env.reset(seed=int(ep["seed"]), randomize=True); m, d = env.model, env.data
    act = env.evaluate_task().metrics["active_scraps"][0]
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in act + ["brush", "dustpan", "bin_container"]}
    def root_name(g):
        b = int(m.geom_bodyid[g]); n = m.body(b).name
        if n.startswith(("left_", "right_")): return "robot:" + n
        return n or ("world" if b == 0 else str(b))
    touch = {n: collections.Counter() for n in act}; zmax = {n: 0.0 for n in act}
    tool0 = {t: d.xpos[bid[t]].copy() for t in ("brush", "dustpan", "bin_container")}
    tool_dmax = {t: 0.0 for t in tool0}; tool_zmax = {t: 0.0 for t in tool0}
    robot_tool = collections.Counter(); steps = 0
    for c in calls:
        if steps >= total: break
        for a in np.asarray(c["actions"], np.float32)[:ex]:
            if steps >= total: break
            env.step(a, render_obs=False); steps += 1
            if steps % 10: continue
            for n in act: zmax[n] = max(zmax[n], float(d.xpos[bid[n]][2]))
            for t in tool0:
                tool_dmax[t] = max(tool_dmax[t], float(np.linalg.norm(d.xpos[bid[t]][:2] - tool0[t][:2])))
                tool_zmax[t] = max(tool_zmax[t], float(d.xpos[bid[t]][2] - tool0[t][2]))
            for j in range(d.ncon):
                b1, b2 = root_name(d.contact[j].geom1), root_name(d.contact[j].geom2)
                for x, y in ((b1, b2), (b2, b1)):
                    if x in touch: touch[x][y.split(":")[0] if y.startswith("robot") else y] += 1
                    if x in ("brush", "dustpan") and y.startswith("robot"): robot_tool[x] += 1
    r = env.evaluate_task()
    print(json.dumps(dict(world=job.name, seed=ep["seed"], steps=steps, success=bool(r.success[0]),
        scraps={n: dict(zmax=round(zmax[n], 3), touched={k: v for k, v in touch[n].items() if k not in n}) for n in act},
        tool_moved_xy=dict((k, round(v, 3)) for k, v in tool_dmax.items()), tool_lift=dict((k, round(v, 3)) for k, v in tool_zmax.items()),
        robot_touch_tool=dict(robot_tool))), flush=True)
