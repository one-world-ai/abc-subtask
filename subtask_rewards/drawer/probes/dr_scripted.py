"""Phase 0: scorer audit of open_drawer / close_drawer with the overlay's scripted IK rollouts (imported read-only
from ../openwam-sim-eval/abc_tasks/tools/scripted_rollouts.py, NO_RENDER) plus free-drawer flings.

Per seed and task, logs every control step: target q, max other q, stock success, finger contact with the target
drawer / any drawer, gripper command. Reports where the stock success first fires, whether a hand was on the drawer
then, and whether success still holds after release + retreat + 1 s settle.

  python dr_scripted.py OUT.json seed0 nseeds
"""
import importlib.util, json, os, sys, time
os.environ["NO_RENDER"] = "1"
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__)
HERE = os.path.dirname(os.path.abspath(__file__))
SR = os.path.normpath(os.path.join(HERE, "../../../../openwam-sim-eval/abc_tasks/tools/scripted_rollouts.py"))
spec = importlib.util.spec_from_file_location("owam_scripted", SR); sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)

J = {"top": "joint_top_drawer_low", "middle": "joint_middle_drawer_low", "bottom": "joint_bottom_drawer_low"}
LINK = {"top": "link_top_drawer_low", "middle": "link_middle_drawer_low", "bottom": "link_bottom_drawer_low"}


def subtree_geoms(m, body):
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body); bs = {root}
    for b in range(m.nbody):
        if m.body_parentid[b] in bs and b: bs.add(b)
    return {g for g in range(m.ngeom) if m.geom_bodyid[g] in bs}


def robot_geoms(m):
    out = set()
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if n.startswith("left_") or n.startswith("right_"): out.add(g)
    return out


class Logged(sr.Rollout):
    """Rollout that logs drawer state + contacts every control step (record() is a no-op)."""
    def __init__(self, env, target):
        self.target = target
        m = env.model
        self.adr = {k: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for k, j in J.items()}
        self.dg = {k: subtree_geoms(m, b) for k, b in LINK.items()}
        self.rg = robot_geoms(m)
        self.trace = []
        # sr.Rollout.__init__ without the renderer/fonts
        self.env, self.robot = env, sr.Robot(env)
        self.cams, self.view, self.label, self.frames, self.log = [], None, "", [], []
        self.renderer, self.ik_err, self.n = None, 0.0, 0

    def record(self):
        pass

    def step(self, n=1, every=2):
        for _ in range(n):
            self.env.step(self.robot.action.astype(np.float32), render_obs=False)
            self.n += 1
            d = self.env.data; r = self.env.evaluate_task()
            touch = {k: False for k in J}
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist > 1e-3: continue
                g1, g2 = int(c.geom1), int(c.geom2)
                for k, gs in self.dg.items():
                    if (g1 in gs and g2 in self.rg) or (g2 in gs and g1 in self.rg): touch[k] = True
            q = {k: float(d.qpos[a]) for k, a in self.adr.items()}
            self.trace.append(dict(q=q[self.target], others=max(v for k, v in q.items() if k != self.target),
                                   success=bool(np.asarray(r.success).ravel()[0]), reward=float(np.asarray(r.reward).ravel()[0]),
                                   touch_target=touch[self.target], touch_other=any(v for k, v in touch.items() if k != self.target),
                                   qd=float(d.qvel[self.env.model.jnt_dofadr[mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, J[self.target])]])))


def summarize(tr):
    s = [t["success"] for t in tr]
    first = next((i for i, x in enumerate(s) if x), None)
    return dict(steps=len(tr), ever_success=any(s), first_success_step=first,
                touch_at_first_success=(tr[first]["touch_target"] if first is not None else None),
                final_success=s[-1], final_q=round(tr[-1]["q"], 4), max_q=round(max(t["q"] for t in tr), 4),
                min_q=round(min(t["q"] for t in tr), 4), max_others=round(max(t["others"] for t in tr), 4),
                touched_other=any(t["touch_other"] for t in tr), success_flips=int(sum(a != b for a, b in zip(s, s[1:]))))


def scripted(task, seed, success):
    env = abc_sim.make_env(task=task, render_cameras=False)
    env.reset(seed=seed); md = env._last_randomization.metadata
    ro = Logged(env, md["target_drawer"])
    t0 = time.time()
    sr.script_drawer(ro, md, "open" if task == "open_drawer" else "close", success)
    # retreat is in the script; settle 1 s more with the action held
    ro.hold(1.0)
    out = summarize(ro.trace); out.update(target=md["target_drawer"], ik_err=round(ro.ik_err, 4), secs=round(time.time() - t0, 1))
    return out


def hook_open(env, ro, md):
    """No-grasp open: closed fingertips enter from the front beside the crossbar's end (bar half-width 2.5 cm, stem
    0.5 cm, gap behind the bar ~3 cm deep), slide sideways behind the bar, then pull 17 cm, retreat."""
    bar, pull, q0 = sr.drawer_geom(env, md["target_drawer"])
    arm = "right" if bar[1] < 0 else "left"
    side = np.cross([0, 0, 1], pull); side /= np.linalg.norm(side)
    if arm == "left":
        side = -side                                   # enter from the side nearer the arm's base? (keep wrist outboard)
    a = -pull * np.cos(np.radians(15)) + sr.DOWN * np.sin(np.radians(15))
    R = [sr.frame(a, [0, 0, 1]), sr.frame(a, side)]
    tip_in = bar - pull * 0.021 + side * 0.045          # beside the bar end, at the gap depth
    tip_hk = bar - pull * 0.021 + side * 0.015          # behind the bar, beside the stem
    R = ro.move(arm, tip_in - a * 0.019 - a * 0.08, R, grip=0.0, speed=0.3)
    ro.move(arm, tip_in - a * 0.019, R, speed=0.05)
    ro.move(arm, tip_hk - a * 0.019, R, speed=0.03)
    ro.move(arm, tip_hk - a * 0.019 + pull * 0.17, R, speed=0.08)
    ro.move(arm, tip_hk - a * 0.019 + pull * 0.17 + side * 0.04, R, speed=0.1)
    ro.move(arm, tip_hk - a * 0.019 + pull * 0.17 + side * 0.04 - a * 0.10, R, speed=0.2)


def pull_partial(env, ro, md, frac=0.6):
    """Overlay success grasp, but pull only 0.6 x 17 cm (~10 cm), release, retreat."""
    bar, pull, q0 = sr.drawer_geom(env, md["target_drawer"])
    arm = "right" if bar[1] < 0 else "left"
    approach = lambda deg: -pull * np.cos(np.radians(deg)) + sr.DOWN * np.sin(np.radians(deg))
    cands = [(approach(t), sr.frame(approach(t), [0, 0, 1])) for t in (15, 30, 22)]
    R = ro.move(arm, bar - cands[0][0] * 0.10, [c[1] for c in cands], grip=1.0, speed=0.3)
    a = next(c[0] for c in cands if np.allclose(c[1], R))
    ro.move(arm, bar - a * 0.004, R, speed=0.06); ro.grip(arm, 0.0, 0.7)
    ro.move(arm, bar - a * 0.004 + pull * 0.17 * frac, R, speed=0.10); ro.grip(arm, 1.0, 0.4)
    ro.move(arm, bar + pull * 0.17 * frac - a * 0.10, R, speed=0.2)


def custom(task, seed, fn):
    env = abc_sim.make_env(task=task, render_cameras=False)
    env.reset(seed=seed); md = env._last_randomization.metadata
    ro = Logged(env, md["target_drawer"])
    fn(env, ro, md); ro.hold(1.0)
    out = summarize(ro.trace); out.update(target=md["target_drawer"], ik_err=round(ro.ik_err, 4))
    return out


def fling(task, seed, v):
    """No robot: give the target drawer a velocity and let it coast (bounce at the travel limits?)."""
    env = abc_sim.make_env(task=task, render_cameras=False)
    env.reset(seed=seed); md = env._last_randomization.metadata
    ro = Logged(env, md["target_drawer"])
    m = env.model; j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, md["target_joint"])
    env.data.qvel[m.jnt_dofadr[j]] = v
    ro.step(60)
    return dict(v=v, **summarize(ro.trace))


def handmade(task, seed):
    """Set qpos directly (mj_forward) around each threshold and read the stock evaluator."""
    env = abc_sim.make_env(task=task, render_cameras=False)
    env.reset(seed=seed); md = env._last_randomization.metadata
    m, d = env.model, env.data
    adr = {k: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]) for k, jn in J.items()}
    other = next(k for k in J if k != md["target_drawer"])
    q0 = d.qpos.copy(); rows = []
    cases = ([(0.119, 0.0), (0.121, 0.0), (0.2, 0.0), (0.121, 0.049), (0.121, 0.051), (0.0, 0.2)] if task == "open_drawer"
             else [(0.016, 0.0), (0.014, 0.0), (0.0, 0.0), (0.014, 0.049), (0.014, 0.051)])
    for qt, qo in cases:
        d.qpos[:] = q0; d.qpos[adr[md["target_drawer"]]] = qt; d.qpos[adr[other]] = qo; d.qvel[:] = 0
        mujoco.mj_forward(m, d); r = env.evaluate_task()
        rows.append(dict(target_q=qt, other_q=qo, success=bool(np.asarray(r.success).ravel()[0]), reward=round(float(np.asarray(r.reward).ravel()[0]), 3)))
    return rows


if __name__ == "__main__":
    out_path, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    res = {}
    for task in ("open_drawer", "close_drawer"):
        R = res[task] = dict(success_script=[], failure_script=[], hook=[], partial=[], fling=[], handmade=handmade(task, s0))
        for seed in range(s0, s0 + n):
            R["success_script"].append(dict(seed=seed, **scripted(task, seed, True)))
            R["failure_script"].append(dict(seed=seed, **scripted(task, seed, False)))
            print(task, seed, "S", R["success_script"][-1], "\n   F", R["failure_script"][-1], flush=True)
            if task == "open_drawer":
                R["hook"].append(dict(seed=seed, **custom(task, seed, hook_open)))
                R["partial"].append(dict(seed=seed, **custom(task, seed, pull_partial)))
                print("   H", R["hook"][-1], "\n   P", R["partial"][-1], flush=True)
        vs = (0.3, 0.6, 1.0, 2.0) if task == "open_drawer" else (-0.3, -0.6, -1.0, -2.0)
        for v in vs:
            R["fling"].append(dict(seed=s0, **fling(task, s0, v)))
            print(task, "fling", R["fling"][-1], flush=True)
        print(task, "handmade", R["handmade"], flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
