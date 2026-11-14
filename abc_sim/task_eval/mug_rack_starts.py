"""Start states for the mug-rack subtasks (see ``mug_rack_stages``), built in the simulator for any seed.

    reach    the reset scene (seeds whose only mugs are mug_1/7/10 are skipped: approved decision 2)
    pick     one gripper open around the rim of the target mug, which still rests where it spawned (top-down rim
             pinch away from the handle); the open fingers do not touch it and it does not move while settling
    align    chained from the pick solver: the pick start, close, lift link_6 by 8-15 cm (physical moves only)
    insert   the target in a real hang pose on a free peg (settled under gravity from a hole-on-peg placement,
             stock scorer says hanging), pulled out along the peg so the hole centre is 0.5-3.5 cm beyond the tip,
             held there by a rim pinch synthesized in that pose (kit ``hold_object``)
    release  the same real hang pose, lifted 3 mm, held by a rim pinch (threaded >= 2 cm deep)

Grasps are synthesized (source 3 of the skill): the mug-flip library grasps rarely suit the hang orientation
(phase 0 pilot: 198 of 224 failed IK or the hold). A rim pinch closes link_6's y axis radially across the cup wall
just below the rim, link_6 z pointing into the cup (tilted outwards by ``tilt``), at a rim angle away from the
handle, so the handle stays free (approved decision 4) and backing off along -z leaves the mug in place.
Each start is a dict that ``env.reset_to_stage(seed, start)`` restores exactly.
"""
from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.mug_rack_stages import A, I, R, RL, STAGES, StageConfig, classify
from abc_sim.task_eval.stagekit import (ARMS, GRIP_MAX, STATE_SPEC, T, actuator_limits, arm_ids, hold_action,
                                       hold_object, ik_link6, inv, place_gripper, set_object_pose)

SEED_BASE = 1_200_000_000  # 950M is nuts_bolts (subtask_rewards/README.md)
SITE_Z = 0.1347  # grasp site along link_6 z
RIM_DEPTH_M = 0.015  # grasp site this far below the rim (table pick)
HANG_RIM_DEPTH_M = 0.025  # deeper for the hang orientation (weight across the pinch)
HANDLE_CLEAR_DEG = 60.0  # rim angle at least this far from the handle direction


# -- geometry --------------------------------------------------------------------------------------------------------

def mug_geometry(scene, k: int) -> dict:
    """Cup frame of mug k in its body frame: axis centre (xy), handle angle, rim height, outer radius per angle."""
    info = scene.geometry._mug_handles[k]
    local = info["_hole_args"][0]
    centre = np.asarray(info["centre"], float)
    hs = info["handle_sectors"]
    handle_angle = float(np.angle(np.mean(np.exp(1j * (-np.pi + (hs + 0.5) * np.pi / 12))))) if hs.size else 0.0
    off = local[:, :2] - centre
    ang = np.arctan2(off[:, 1], off[:, 0])
    rad = np.linalg.norm(off, axis=1)
    body = ~np.isin(scene.geometry._sector(off), info["allowed"])
    z_rim = float(local[body, 2].max())
    z_base = float(local[:, 2].min())
    return dict(centre=centre, handle_angle=handle_angle, z_rim=z_rim, z_base=z_base, wall=float(info["wall"]),
                local=local, ang=ang, rad=rad, body=body)


def rim_radius(geo: dict, theta: float, band_m: float = 0.012) -> float:
    """Outer radius of the cup wall near the rim at angle theta (body frame)."""
    d = np.abs(np.angle(np.exp(1j * (geo["ang"] - theta))))
    sel = geo["body"] & (d < np.deg2rad(15)) & (geo["local"][:, 2] > geo["z_rim"] - band_m)
    return float(geo["rad"][sel].max()) if sel.any() else geo["wall"]


def rim_pinch(geo: dict, theta: float, tilt: float, arm: int, roll: float = 0.0, depth: float = RIM_DEPTH_M) -> dict:
    """Grasp (kit format, body frame): pinch the wall at rim angle theta, link_6 z into the cup tilted ``tilt``
    (rad) outwards; ``roll`` flips the closing axis sign (0 or pi), which mirrors the wrist."""
    e = np.array([np.cos(theta), np.sin(theta), 0.0])
    zc = np.array([0.0, 0.0, 1.0])
    r_out = rim_radius(geo, theta)
    site = np.array([*(geo["centre"] + (r_out - 0.003) * e[:2]), geo["z_rim"] - depth])
    z6 = -np.cos(tilt) * zc + np.sin(tilt) * e  # tilted so the wrist sits outside the cup
    y6 = e - (e @ z6) * z6
    y6 /= np.linalg.norm(y6)
    if roll:
        y6 = -y6
    x6 = np.cross(y6, z6)
    T6 = np.eye(4)
    T6[:3, :3] = np.column_stack([x6, y6, z6])
    T6[:3, 3] = site - SITE_Z * z6
    return {"arm": int(arm), "T_obj_link6": T6, "fingers": np.array([GRIP_MAX, GRIP_MAX]),
            "theta": float(theta), "tilt": float(tilt), "roll": float(roll)}


def candidate_thetas(geo: dict, toward: float, n: int = 12) -> list[float]:
    """Rim angles >= HANDLE_CLEAR_DEG from the handle, sorted by closeness to ``toward`` (body-frame angle)."""
    out = []
    for th in np.linspace(-np.pi, np.pi, n, endpoint=False):
        if abs(np.angle(np.exp(1j * (th - geo["handle_angle"])))) >= np.deg2rad(HANDLE_CLEAR_DEG):
            out.append(float(th))
    return sorted(out, key=lambda th: abs(np.angle(np.exp(1j * (th - toward)))))


def arm_base(arm: int) -> np.ndarray:
    return np.array([0.252, 0.31 if arm == 0 else -0.31, 0.823])


# -- motion helpers (physical, never teleport a held object) --------------------------------------------------------

def idle_action(env) -> np.ndarray:
    action = np.zeros(14, np.float32)
    for a, arm in enumerate(ARMS):
        action[a * 7:a * 7 + 6] = env.data.qpos[arm_ids(env.model, arm).qadr]
        action[a * 7 + 6] = env.data.ctrl[env._ctrl_indices[a * 7 + 6]] / GRIP_MAX
    return action


def plan_path(env, arm: int, target: np.ndarray, *, n: int = 8, max_pos=0.008, max_rot=0.2, q_start=None):
    """IK waypoints for a straight link_6 move (slerp orientation); None, why if a waypoint is unreachable."""
    model, data = env.model, env.data
    ids = arm_ids(model, ARMS[arm])
    planning = mujoco.MjData(model)
    planning.qpos[:] = data.qpos
    if q_start is not None:
        planning.qpos[ids.qadr] = q_start
    mujoco.mj_kinematics(model, planning)
    start = T(planning.xpos[ids.link6], planning.xquat[ids.link6])
    qa, qb = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(qa, start[:3, :3].ravel())
    mujoco.mju_mat2Quat(qb, np.ascontiguousarray(target[:3, :3]).ravel())
    if qa @ qb < 0:
        qb = -qb
    lim = actuator_limits(model, ARMS[arm])
    out = []
    for f in np.linspace(0, 1, n + 1)[1:]:
        quat = (1 - f) * qa + f * qb
        quat /= np.linalg.norm(quat)
        pose = np.eye(4)
        m9 = np.empty(9)
        mujoco.mju_quat2Mat(m9, quat)
        pose[:3, :3] = m9.reshape(3, 3)
        pose[:3, 3] = (1 - f) * start[:3, 3] + f * target[:3, 3]
        q, pe, re = ik_link6(model, planning, ARMS[arm], pose, limits=lim, seeds=3)
        if q is None or pe > max_pos or re > max_rot:
            return None, f"IK path unreachable (position {pe:.4f}, rotation {re:.3f})"
        out.append(q)
        planning.qpos[ids.qadr] = q
    return out, ""


def run_path(env, arm: int, waypoints, *, actions: int = 64, grip: float = 0.0):
    other = idle_action(env)
    for q in waypoints:
        action = other.copy()
        action[arm * 7:arm * 7 + 6] = q
        action[arm * 7 + 6] = grip
        for _ in range(max(1, actions // len(waypoints))):
            env.step(action, render_obs=False)


def move_link(env, arm: int, target: np.ndarray, *, actions: int = 64, grip: float = 0.0, n: int = 8):
    wp, why = plan_path(env, arm, target, n=n)
    if wp is None:
        return False, why
    run_path(env, arm, wp, actions=actions, grip=grip)
    return True, ""


def link6(env, arm: int) -> np.ndarray:
    ids = arm_ids(env.model, ARMS[arm])
    return T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])


def hold_steps(env, n: int, arm: int | None = None, grip: float | None = None):
    action = idle_action(env)
    if grip is not None:
        action[arm * 7 + 6] = grip
    for _ in range(n):
        env.step(action, render_obs=False)


# -- hang poses ------------------------------------------------------------------------------------------------------

def _hang_rotation(scene, k: int, peg: int, sign: float, roll: float = 0.0) -> np.ndarray:
    """Body rotation putting mug k's hole axis along ``sign`` x peg axis, the handle pointing up (as it hangs)."""
    geo = mug_geometry(scene, k)
    _, _, hax = scene.hole(k)
    out_dir = np.array([np.cos(geo["handle_angle"]), np.sin(geo["handle_angle"]), 0.0])
    out_dir = out_dir - (out_dir @ hax) * hax
    out_dir /= np.linalg.norm(out_dir)
    third = np.cross(hax, out_dir)
    u = sign * scene.peg_axis[peg]
    up = np.array([0.0, 0.0, 1.0])
    w_out = up - (up @ u) * u
    w_out /= np.linalg.norm(w_out)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        w_out = c * w_out + s * np.cross(u, w_out)
    w_third = np.cross(u, w_out)
    return np.column_stack([u, w_out, w_third]) @ np.column_stack([hax, out_dir, third]).T


def settle_hang(env, scene, k: int, peg: int, sign: float, *, depth: float, settle: int = 150) -> tuple[bool, dict]:
    """Put mug k with its hole centre on ``peg``'s axis ``depth`` inside the tip, handle up; let it settle with the
    arms at home. -> (stock scorer says hanging and threaded on that peg at rest, info)."""
    model, data = env.model, env.data
    Rm = _hang_rotation(scene, k, peg, sign)
    _, c, _ = scene.hole(k)
    p_hole = scene.peg_tip[peg] - depth * scene.peg_axis[peg] + np.array([0, 0, 0.002])
    pos = p_hole - Rm @ c
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, Rm.ravel())
    set_object_pose(env, scene, k, pos, q)
    other, _ = scene.other_contact(data)
    if other[k]:
        return False, {"why": "hang placement collides"}
    act = hold_action(env, 0, env.get_init_q()[0:6], 1.0)
    act[7:13], act[13] = env.get_init_q()[7:13], 1.0
    for _ in range(settle):
        env.step(act, render_obs=False)
    obs = scene.observe(data, scene.heights(data))
    ok = bool(obs["hanging"][k] and obs["speed"][k] < 0.01)
    rel = scene.peg_relations(data, k)
    on_peg = bool(rel["threaded"][peg])
    return ok and on_peg, {"hanging": bool(obs["hanging"][k]), "on_peg": on_peg, "speed": float(obs["speed"][k]),
                           "depth": float(-rel["axial"][peg])}


def hang_grasps(scene, data, k: int, arms=(0, 1)):
    """Rim pinches (both arms, every rim angle away from the handle, 4 tilts, both wrist mirrors) for mug k."""
    geo = mug_geometry(scene, k)
    # roll 0 first: with the hang orientation the mirrored wrist (roll pi) closed past the wall in 6 of 7 holds
    # (diagnosis 2026-11-14, seed 6); the deeper pinch (2.5 cm) held more often than 1.5 cm.
    # round 3 (time budget): 8 rim angles x 3 tilts instead of 16 x 4; IK was ~75 % of the build time
    for roll in (0.0, np.pi):
        for arm in arms:
            for th in candidate_thetas(geo, 0.0, n=8):
                for tilt in (0.0, -0.35, 0.35):
                    yield rim_pinch(geo, th, tilt, arm, roll, depth=HANG_RIM_DEPTH_M)


def _gripper_geoms(model, arm: int) -> np.ndarray:
    from abc_sim.task_eval.stagekit import geoms_of, subtree
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{ARMS[arm]}_link_6")
    return geoms_of(model, subtree(model, b))


def feasible_grasps(env, scene, k: int, arms=(0, 1), rng=None, limit: int = 10):
    """Grasps of ``hang_grasps`` with an IK solution whose open gripper (kinematics only) touches nothing but
    mug k's surroundings-free space: no contact of the gripper with the tree, table or other mugs. Easiest first."""
    model, data = env.model, env.data
    out = []
    Tk = scene.object_T(data, k)
    scratch = mujoco.MjData(model)
    mine = set(np.flatnonzero((scene.owner[:, 0] == 0) & (scene.owner[:, 1] == k)).tolist())
    grip = {a: set(_gripper_geoms(model, a).tolist()) for a in arms}
    for g in hang_grasps(scene, data, k, arms):
        Tg = Tk @ g["T_obj_link6"]
        if np.linalg.norm(Tg[:3, 3] - arm_base(g["arm"])) > 0.60 or Tg[2, 3] < 0.80:
            continue
        q, pe, re = ik_link6(model, data, ARMS[g["arm"]], Tg, seeds=3, iters=50, rng=rng)
        if q is None or pe > 3e-3 or re > 0.05:
            continue
        ids = arm_ids(model, ARMS[g["arm"]])
        scratch.qpos[:] = data.qpos
        scratch.qpos[ids.qadr] = q
        scratch.qpos[ids.finger_qadr[0]], scratch.qpos[ids.finger_qadr[1]] = GRIP_MAX, -GRIP_MAX
        mujoco.mj_forward(model, scratch)
        hit = False
        for i in range(int(scratch.ncon)):
            c = scratch.contact[i]
            a, b = int(c.geom1), int(c.geom2)
            if (a in grip[g["arm"]] and b not in mine) or (b in grip[g["arm"]] and a not in mine):
                hit = True
                break
        if not hit:
            out.append((pe + 0.05 * re, g))
            if len(out) >= limit:
                break
    return [g for _, g in out]  # generation order (roll 0 first)


def clear_pose(env, scene, k: int, peg: int, hang: np.ndarray, out_m: float):
    """Mug k's settled hang pose ``hang`` moved ``out_m`` (+ up to 1 cm more) out along ``peg`` and lifted to the
    middle of the contact-free lift range (0-15 mm), so the peg passes the hole with clearance all round.
    -> qpos[7] or None."""
    data = env.data
    for extra in (0.0, 0.005, 0.01):
        free = []
        for lift in np.arange(0.0, 0.0155, 0.001):
            pose = hang.copy()
            pose[:3] += (out_m + extra) * scene.peg_axis[peg] + np.array([0.0, 0.0, lift])
            set_object_pose(env, scene, k, pose[:3], pose[3:])
            ok = not scene.other_contact(data)[0][k]
            if ok:
                free.append(pose)
            elif free:
                break
        if free:
            pose = free[len(free) // 2]
            set_object_pose(env, scene, k, pose[:3], pose[3:])
            return pose
    return None


def hang_hold(env, scene, k: int, rng, *, out_m: float, max_holds: int = 8, arms=(0, 1)):
    """Mug k is in a settled hang pose on a peg: move it clear (``clear_pose``) and hold it there with a rim pinch
    (IK- and kinematically collision-free ones first, at most ``max_holds`` physical hold checks). -> (ok, info)."""
    model, data = env.model, env.data
    a = int(scene.qpos_adr[k])
    peg = int(np.argmax(scene.peg_relations(data, k)["threaded"]))
    hang = data.qpos[a:a + 7].copy()
    saved = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, saved, STATE_SPEC)
    goal = clear_pose(env, scene, k, peg, hang, out_m)
    if goal is None:
        mujoco.mj_setState(model, data, saved, STATE_SPEC)
        mujoco.mj_forward(model, data)
        return False, dict(why="no contact-free pose near the hang")
    grasps = feasible_grasps(env, scene, k, arms, rng)
    whys = []
    for grasp in grasps[:max_holds]:
        mujoco.mj_setState(model, data, saved, STATE_SPEC)
        set_object_pose(env, scene, k, goal[:3], goal[3:])
        ok, info = hold_object(env, scene, k, grasp, rng=rng)
        if ok:  # G3 repair round 1: the 30-action hold missed slow slips (insert idle: 17 of 67 left the stage)
            why = _creeps(env, scene, k, int(grasp["arm"]))
            if why:
                ok, info["why"] = False, why
        if ok:
            info.update(peg=peg, theta=grasp["theta"], tilt=grasp["tilt"], roll=grasp["roll"], out_m=out_m,
                        arm=int(grasp["arm"]), feasible=len(grasps))
            return True, info
        whys.append(info.get("why", "?"))
    mujoco.mj_setState(model, data, saved, STATE_SPEC)
    mujoco.mj_forward(model, data)
    return False, dict(why=("; ".join(sorted(set(w.split(" (")[0] for w in whys))) or "no feasible grasp")[:300],
                       feasible=len(grasps))


def hang_start(env, scene, k: int, stage: int, rng, *, options: int = 12):
    """Insert / release start for mug k: for each (peg, side) nearest an arm first, settle a real hang, then hold
    it clear of the peg (release: in place, threaded; insert: hole centre 0.5-3.5 cm beyond the tip)."""
    model, data = env.model, env.data
    saved = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, saved, STATE_SPEC)
    order = np.argsort([min(np.linalg.norm(t - arm_base(0)), np.linalg.norm(t - arm_base(1))) for t in scene.peg_tip])
    opts = [(int(p), sg) for p in order for sg in (1.0, -1.0)]
    first = int(rng.integers(2))  # which side first: both kinds of start appear in a start set
    opts = [opts[i ^ first] for i in range(len(opts))]
    tried = []
    import time
    t0 = time.process_time()
    for peg, sign in opts[:options]:
        if time.process_time() - t0 > 8.0:  # G3 budget is 10 s per start (reset and hole measurement included)
            tried.append("out of time")
            break
        mujoco.mj_setState(model, data, saved, STATE_SPEC)
        mujoco.mj_forward(model, data)
        ok, hinfo = settle_hang(env, scene, k, peg, sign, depth=float(rng.uniform(0.03, 0.045)))
        if not ok:
            tried.append(f"peg{peg}{'+' if sign > 0 else '-'}: " + (hinfo.get("why") or "not hanging"))
            continue
        out = hinfo["depth"] + float(rng.uniform(0.005, 0.035)) if stage == I else 0.0
        ok, info = hang_hold(env, scene, k, rng, out_m=out)
        if ok:
            return True, dict(info, hang_peg=peg, sign=sign, hang_depth=round(hinfo["depth"], 4), tried=tried)
        tried.append(f"peg{peg}{'+' if sign > 0 else '-'}: " + info["why"])
    mujoco.mj_setState(model, data, saved, STATE_SPEC)
    mujoco.mj_forward(model, data)
    return False, dict(why=" | ".join(tried)[:500])


# -- builders --------------------------------------------------------------------------------------------------------

class _no_eval:
    """Step the physics without running the subtask evaluator (it only reads the state; the stock threading test
    on a hanging mug costs ~20 ms per action, 10x the physics). The built state is the same either way."""

    def __init__(self, env):
        self.env = env

    def __enter__(self):
        self.ev, self.env._task_evaluator = self.env._task_evaluator, None

    def __exit__(self, *exc):
        self.env._task_evaluator = self.ev


def _creeps(env, scene, k: int, arm: int, steps: int = 32, max_move_m: float = 0.002) -> str:
    """Hold one more chunk; '' if mug k stays put in ``arm``'s pinch, else why."""
    data = env.data
    a = int(scene.qpos_adr[k])
    p0 = data.qpos[a:a + 3].copy()
    hold_steps(env, steps)
    moved = float(np.linalg.norm(data.qpos[a:a + 3] - p0))
    if moved > max_move_m or not scene.contact_obs(data)["touched"][k]:
        return f"creeps in the pinch ({moved:.4f} m per chunk)"
    return ""

def _reset(env, seed):
    env.forget_arm_state()
    env.reset(seed=seed, randomize=True)
    env.evaluate_task()
    ev = env._task_evaluator
    return ev, ev.scene


def _snapshot(env, seed, stage, k, arm, rest, labels, info):
    state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
    return dict(seed=int(seed), stage=stage, target=int(k), arm=int(arm), state=state, rest_height=rest,
                labels=np.asarray(labels), info=info)


def _label(scene, data, rest, cfg, k, start_label=R):
    obs = scene.observe(data, rest)
    labels = np.full(len(scene.names), start_label)
    for _ in range(3):
        labels = classify(obs, labels, cfg)
    return labels, obs


def pick_setup(env, scene, k: int, arm: int, rest, rng, *, tries: int = 36, skip: int = 0,
               depth: float = RIM_DEPTH_M):
    """Open gripper around the rim of resting mug k (top-down rim pinch). -> (ok, info)."""
    model, data = env.model, env.data
    geo = mug_geometry(scene, k)
    root = scene.bodies[k]
    a = int(scene.qpos_adr[k])
    pose = data.qpos[a:a + 7].copy()
    Rm = data.xmat[root].reshape(3, 3)
    to_arm = Rm.T @ (arm_base(arm) - data.xpos[root])
    toward = float(np.arctan2(to_arm[1], to_arm[0]))
    thetas = candidate_thetas(geo, toward, n=16)[:6]
    thetas = thetas[skip % len(thetas):] + thetas[:skip % len(thetas)]  # a later attempt starts at another angle
    saved = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, saved, STATE_SPEC)
    why = []
    for i in range(tries):
        th = thetas[i % len(thetas)] + rng.uniform(-0.1, 0.1)
        tilt = (0.0, 0.3, 0.5, 0.15)[(i // (2 * len(thetas))) % 4]
        grasp = rim_pinch(geo, th, tilt, arm, roll=np.pi * ((i // len(thetas)) % 2), depth=depth)
        mujoco.mj_setState(model, data, saved, STATE_SPEC)
        mujoco.mj_forward(model, data)
        ok, gi = place_gripper(env, scene, k, grasp, open_extra=0.0, rng=rng)
        if not ok:
            why.append(gi["why"])
            continue
        data.qpos[a:a + 7] = pose
        mujoco.mj_forward(model, data)
        obs = scene.contact_obs(data)
        if obs["touched"][k] or _gripper_touches(scene, data, arm):
            why.append("open gripper touches the mug or the scene")
            continue
        act = hold_action(env, arm, gi["q"], 1.0)
        for _ in range(10):
            env.step(act, render_obs=False)
        moved = float(np.linalg.norm(data.qpos[a:a + 3] - pose[:3]))
        if moved > 0.01 or scene.contact_obs(data)["touched"][k]:  # mug-flip pick-start rule
            why.append(f"mug moved {moved:.3f} m while settling")
            continue
        return True, dict(theta=round(th, 3), tilt=tilt, tries=i + 1, grasp=grasp)
    return False, dict(why="; ".join(sorted(set(why)))[:300], tries=tries)


def _gripper_touches(scene, data, arm: int) -> bool:
    """A finger of ``arm`` touches anything (table, tree, other mug) — the open gripper must hang free."""
    own = {arm * 2, arm * 2 + 1}
    for i in range(int(data.ncon)):
        c = data.contact[i]
        o1, o2 = scene.owner[int(c.geom1)], scene.owner[int(c.geom2)]
        if (o1[0] == 1 and int(o1[1]) in own) or (o2[0] == 1 and int(o2[1]) in own):
            return True
    return False


def pick_solver(env, scene, k: int, arm: int, lift: float = 0.10):
    """Close on the rim, then lift link_6 straight up by ``lift`` (physical; 60 % of it if that is unreachable)."""
    hold_steps(env, 32, arm, 0.0)
    why = ""
    for frac in (1.0, 0.6):
        target = link6(env, arm)
        target[2, 3] += frac * lift
        wp, why = plan_path(env, arm, target)
        if wp is not None:
            run_path(env, arm, wp, actions=96, grip=0.0)
            return True, ""
    return False, why


def build_start(env, seed: int, stage: str, *, rng: np.random.Generator | None = None, tries: int = 3,
                cfg: StageConfig | None = None):
    """Make a ``stage`` start for ``seed`` on an env made for ``hang_mug_on_mug_rack__<stage>``.
    -> (start or None, log of attempts)."""
    cfg = cfg or StageConfig()
    rng = rng or np.random.default_rng(seed)
    si = STAGES.index(stage)
    log = []
    for attempt in range(1 if si in (R, I, RL) else tries + 3 * (si == A)):  # hang starts already loop over pegs
        try:
            ev, scene = _reset(env, seed)
        except RuntimeError as exc:  # no threadable mug: skipped seed (decision 2)
            return None, log + [dict(attempt=attempt, why=f"skipped: {exc}")]
        data = env.data
        rest = scene.heights(data)
        k = ev.target
        arm = 0 if data.qpos[scene.qpos_adr[k] + 1] > 0 else 1
        info = dict(attempt=attempt, target=k, variant=scene.variants[k], arm=arm)
        if si == R:
            labels, _ = _label(scene, data, rest, cfg, k)
            if labels[k] != R:
                return None, [dict(info, why=f"reset scene is {STAGES[labels[k]]}")]
            return _snapshot(env, seed, stage, k, -1, rest, labels, info), [dict(info, ok=True)]
        with _no_eval(env):
            ok, info, arm = _setup(env, scene, k, arm, si, rest, rng, attempt, info)
        if not ok:
            log.append(info)
            continue
        labels, obs = _label(scene, data, rest, cfg, k)
        info.update(label=STAGES[labels[k]] if labels[k] < len(STAGES) else "fallen", dist=round(float(obs["dist"][k]), 4),
                    lift=round(float(obs["lift"][k]), 4))
        if labels[k] != si:
            log.append(dict(info, why=f"classified as {info['label']}"))
            continue
        return _snapshot(env, seed, stage, k, arm, rest, labels, info), log + [dict(info, ok=True)]
    return None, log


def _setup(env, scene, k, arm, si, rest, rng, attempt, info):
    """The stage's scripted setup on the reset scene. -> (ok, info, arm)."""
    data = env.data
    if True:
        if si in (1, 2):
            # align: the chain closes a deeper pinch (2.5 cm, as the hang starts) -- G3 repair round 1: the 1.5 cm
            # pinch slipped out of 5 of 78 align starts within the 10 idle chunks
            # round 3: back to the 1.5 cm pinch for the align chain too (2.5 cm built fewer: 69 vs 78 of 100; the
            # creep check now catches the slow slips the 1.5 cm pinch had)
            ok, detail = pick_setup(env, scene, k, arm, rest, rng, skip=2 * attempt)
            info.update({key: v for key, v in detail.items() if key != "grasp"})
            if ok and si == 2:
                ok, why = pick_solver(env, scene, k, arm, lift=float(rng.uniform(0.08, 0.15)))
                hold_steps(env, 16)
                if not ok:
                    info["why"] = "pick chain: " + why
                else:  # creep check: one more chunk held, the mug must not sink in the pinch
                    a = int(scene.qpos_adr[k])
                    z0 = float(data.qpos[a + 2])
                    hold_steps(env, 32)
                    sink = z0 - float(data.qpos[a + 2])
                    # round 2: finger contacts flicker on a still mug, so require a finger, not both, here
                    if sink > 0.002 or not scene.contact_obs(data)["touched"][k]:
                        ok, info["why"] = False, f"creeping in the pinch ({sink:.4f} m per chunk)"
        else:
            ok, detail = hang_start(env, scene, k, si, rng)
            info.update({key: v for key, v in detail.items() if key != "q"})
            arm = int(detail.get("arm", arm))
    return ok, info, arm
