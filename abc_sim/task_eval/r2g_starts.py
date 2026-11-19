"""Start states of the Real2Gym "in a container" stages (subtask_rewards/r2g/stage_table.md).

  reach  the scene's own reset
  pick   the named arm teleported open around the resting target (top-down, fingers across a narrow axis)
  carry  the target held by the named arm 5-15 cm above its spot (stagekit.hold_object)
  place  chained from a carry hold: a physical move until the target is over its container

A start is kept only if it classifies as its stage, the pinch the final check needs has been accumulated (carry,
place) and the scene holds still. Starts whose container the named arm cannot reach from above are not built
(stage table decision 5).
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from abc_sim.task_eval.bottles_starts import idle_action, move_link
from abc_sim.task_eval.stagekit.holding import touches_other
from abc_sim.task_eval.r2g_stages import C, PINCH, R, STAGES, classify
from abc_sim.task_eval.stagekit import (ARMS, GRIP_MAX, STATE_SPEC, T, arm_ids, hold_action, hold_object, ik_link6,
                                       inv, place_gripper, set_object_pose)

SEED_BASE = 930_000_000  # start-set seeds: SEED_BASE + i
WEDGE_SLACK_M = 0.015    # a pinch closes to the object's width; 5 screwdriver holds that never released were 6 cm wider
GRIP_OPEN_M = 0.085      # widest object the open fingers go around (2 x 4.75 cm travel, minus clearance)
MAX_START_S = 10.0       # G3: CPU seconds per start


def _state(env) -> np.ndarray:
    st = np.zeros(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, st, STATE_SPEC)
    return st


def labels_now(scene, data, rest, cfg) -> np.ndarray:
    """Stage labels of a state judged on its own (no history)."""
    obs = scene.observe(data, rest)
    lab = np.full(len(scene.names), R, dtype=np.int64)
    for _ in range(3):
        lab = classify(obs, lab, cfg)
    return lab


def reset_scene(env, seed: int, target: int | None = None) -> tuple[bool, str]:
    """reset(seed) as a fresh episode on the plain scene and begin the evaluator's episode there."""
    env.forget_arm_state()
    try:
        env.reset(seed=seed, randomize=True)
        env.evaluate_task()
    except RuntimeError as exc:
        return False, f"reset raised: {exc}"
    ev = env._task_evaluator
    if target is not None and int(target) != ev.target:
        if int(target) not in ev.candidates():
            return False, f"object {target} is not a valid target in this layout"
        ev.begin(env.model, env.data, int(target))
    return True, ""


SIDE_TILTS_DEG = (80.0, 60.0, 40.0, 20.0)  # holds for tall containers (see hold_tilts)
TALL_M = 0.10
TILTS_DEG = (0.0, 15.0, 30.0, 40.0)  # approach tilted away from the arm base (the scripted probes' set)


def pinch_grasp(env, scene, k: int, yaw: float, arm: int, raise_m: float = 0.0, tilt_deg: float = 0.0) -> dict:
    """Grasp from above with the finger-pad centre (not the grasp site: the pads sit 4.4 cm to its side) on object
    k's centre of mass, ``raise_m`` above it; fingers open, closing along the horizontal direction ``yaw``; the
    approach is tilted ``tilt_deg`` from the vertical, away from the arm's base (reaches further out)."""
    model, data = env.model, env.data
    ids = arm_ids(model, ARMS[arm])
    s = scene.sites[arm]
    T6 = T(data.xpos[ids.link6], data.xquat[ids.link6])
    Ts = np.eye(4)
    Ts[:3, :3], Ts[:3, 3] = data.site_xmat[s].reshape(3, 3), data.site_xpos[s]
    T6_site = inv(T6) @ Ts  # the grasp site in link_6's frame (rigid)
    com = np.asarray(data.xipos[scene.bodies[k]], dtype=np.float64)
    base = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{ARMS[arm]}_arm")]
    out = com[:2] - base[:2]
    out = out / max(float(np.linalg.norm(out)), 1e-9)
    t = np.radians(tilt_deg)
    z = np.array([out[0] * np.sin(t), out[1] * np.sin(t), -np.cos(t)])  # site z: the approach direction
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])                      # site y: the fingers' closing direction
    y = y - (y @ z) * z
    y /= np.linalg.norm(y)
    want = np.eye(4)
    want[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    want[:3, 3] = com + np.array([0.0, 0.0, raise_m]) - want[:3, :3] @ PINCH
    return {"arm": int(arm), "T_obj_link6": inv(scene.object_T(data, k)) @ want @ inv(T6_site),
            "fingers": np.array([GRIP_MAX, GRIP_MAX]), "yaw": float(yaw), "tilt_deg": float(tilt_deg)}


def closing_yaws(env, scene, k: int) -> list[float]:
    """Yaws (about world z) to try for a top-down pinch of object k: fingers across each near-horizontal body axis
    whose extent fits the open gripper, narrowest first; then the two table axes."""
    f, n = scene.full, scene.names[k]
    ext = f._objs[n]["extent"]
    size = [ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]]
    Rb = env.data.xmat[scene.bodies[k]].reshape(3, 3)
    cands = []
    for i in range(3):
        axis = Rb[:, i]
        if abs(axis[2]) < 0.5 and size[i] <= GRIP_OPEN_M:
            cands.append((size[i], float(np.arctan2(axis[1], axis[0]))))
    yaws = []
    for _, y in sorted(cands):
        yaws.append(y)
    return yaws + [0.0, np.pi / 2]


def pinch_width(env, scene, k: int) -> float:
    """Narrowest near-horizontal extent of object k that fits the open gripper (what a pinch closes on), else the
    gripper's opening."""
    ext = scene.full._objs[scene.names[k]]["extent"]
    size = [ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]]
    Rb = env.data.xmat[scene.bodies[k]].reshape(3, 3)
    fit = [size[i] for i in range(3) if abs(Rb[2, i]) < 0.5 and size[i] <= GRIP_OPEN_M]
    return min(fit) if fit else GRIP_OPEN_M


def finger_gap(env, arm: int) -> float:
    ids = arm_ids(env.model, ARMS[arm])
    return float(sum(abs(env.data.qpos[q]) for q in ids.finger_qadr))


def _finger_contacts(model, data, scene) -> bool:
    f = scene.full
    for i in range(int(data.ncon)):
        c = data.contact[i]
        if int(c.geom1) in f._fingers or int(c.geom2) in f._fingers:
            if c.dist <= 1e-3:
                return True
    return False


def over_container_pose(env, scene, k: int, arm: int, rng, lift=(0.03, 0.08)) -> tuple[np.ndarray | None, str]:
    """link_6 pose that carries the held object k, as it is held now, to a point over its container."""
    model, data, f = env.model, env.data, scene.full
    n = scene.names[k]
    c = scene.goal_of[n]["containers"][0]
    ext = f._objs[c]["extent"]
    cb = f._body[c]
    Rc = data.xmat[cb].reshape(3, 3)
    mg = scene.cfg.over_margin_m + 0.015
    lx = rng.uniform(min(ext[0] + mg, 0.0), max(ext[1] - mg, 0.0)) * 0.6
    ly = rng.uniform(min(ext[2] + mg, 0.0), max(ext[3] - mg, 0.0)) * 0.6
    centre = data.xpos[cb] + Rc @ np.array([(ext[0] + ext[1]) / 2 + lx, (ext[2] + ext[3]) / 2 + ly, 0.0])
    lo, _ = f._zrange(data, f._geoms[n])
    _, top = f._zrange(data, f._geoms[c])
    com = data.xipos[scene.bodies[k]].copy()
    z = max(top, data.xpos[cb][2] + ext[5]) + (com[2] - lo) + float(rng.uniform(*lift))
    ids = arm_ids(model, ARMS[arm])
    T6 = T(data.xpos[ids.link6], data.xquat[ids.link6])
    target = np.array([centre[0], centre[1], z])
    best = (None, np.inf, np.inf)
    # the wrist may turn about the vertical on the way (the object turns with it): the hold's own yaw first
    for dyaw in sorted(np.linspace(-np.pi, np.pi, 12, endpoint=False), key=abs):
        Rz = np.array([[np.cos(dyaw), -np.sin(dyaw), 0.0], [np.sin(dyaw), np.cos(dyaw), 0.0], [0.0, 0.0, 1.0]])
        goal = np.eye(4)
        goal[:3, :3] = Rz @ T6[:3, :3]
        goal[:3, 3] = target + Rz @ (T6[:3, 3] - com)
        q, pe, re = ik_link6(model, data, ARMS[arm], goal, seeds=3)
        if q is not None and pe < best[1]:
            best = (goal, pe, re)
        if q is not None and pe <= 8e-3 and re <= 0.2:  # move_link's own tolerances
            return goal, ""
    goal, why = _sampled_pose_in_container(env, scene, k, arm, rng, lift)
    if goal is not None:
        return goal, ""
    return None, f"container out of the {ARMS[arm]} arm's reach from above (IK {best[1]:.3f} m); {why}"


def _arm_hits(model, data, arm: int, skip: set[int]) -> set[tuple[int, int]]:
    """Penetrating contacts between ``arm`` (links and fingers) and anything else but the geoms in ``skip``."""
    pre = ARMS[arm] + "_"
    out = set()
    for i in range(int(data.ncon)):
        c = data.contact[i]
        if c.dist >= -1e-4:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        a1 = model.body(int(model.geom_bodyid[g1])).name.startswith(pre)
        a2 = model.body(int(model.geom_bodyid[g2])).name.startswith(pre)
        if a1 != a2 and g1 not in skip and g2 not in skip:
            out.add((g1, g2))
    return out


def _sampled_pose_in_container(env, scene, k: int, arm: int, rng, lift=(0.03, 0.08), n: int = 40):
    """Tall or side-entry containers (shelf, cabinet, counter, rack): sample the held object's place inside the
    container's footprint at any height from its floor to above its top, and the wrist's turn about the vertical;
    keep the first sample the arm reaches with neither the object nor the arm touching anything."""
    model, data, f = env.model, env.data, scene.full
    name = scene.names[k]
    ids = arm_ids(model, ARMS[arm])
    T6 = T(data.xpos[ids.link6], data.xquat[ids.link6])
    To = scene.object_T(data, k)
    com = data.xipos[scene.bodies[k]].copy()
    lo, _ = f._zrange(data, f._geoms[name])
    a = int(scene.qpos_adr[k])
    own = set(f._geoms[name])
    state = np.zeros(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    base = _arm_hits(model, data, arm, own)
    conts = scene.goal_of[name]["containers"]
    mg = scene.cfg.over_margin_m + 0.015
    best = np.inf
    try:
        for i in range(n):
            c = conts[i % len(conts)]
            ext, cb = f._objs[c]["extent"], f._body[c]
            Rc = data.xmat[cb].reshape(3, 3)
            lx = rng.uniform(min(ext[0] + mg, 0.0), max(ext[1] - mg, 0.0))
            ly = rng.uniform(min(ext[2] + mg, 0.0), max(ext[3] - mg, 0.0))
            centre = data.xpos[cb] + Rc @ np.array([lx, ly, 0.0])
            _, top = f._zrange(data, f._geoms[c])
            zc = data.xpos[cb][2]
            z = rng.uniform(zc + ext[4] + (com[2] - lo) + 0.02, max(top, zc + ext[5]) + (com[2] - lo) + lift[1])
            dyaw = rng.uniform(-np.pi, np.pi)
            Rz = np.array([[np.cos(dyaw), -np.sin(dyaw), 0.0], [np.sin(dyaw), np.cos(dyaw), 0.0], [0.0, 0.0, 1.0]])
            goal = np.eye(4)
            goal[:3, :3] = Rz @ T6[:3, :3]
            goal[:3, 3] = np.array([centre[0], centre[1], z]) + Rz @ (T6[:3, 3] - com)
            q, pe, re = ik_link6(model, data, ARMS[arm], goal, seeds=2)
            if q is None:
                continue
            best = min(best, pe)
            if pe > 8e-3 or re > 0.2:
                continue
            data.qpos[ids.qadr] = q
            Tn = goal @ np.linalg.inv(T6) @ To  # the object rides with the gripper
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, Tn[:3, :3].flatten())
            data.qpos[a:a + 3], data.qpos[a + 3:a + 7] = Tn[:3, 3], quat
            mujoco.mj_forward(model, data)
            hit = touches_other(model, data, scene, k, arm) or bool(_arm_hits(model, data, arm, own) - base)
            mujoco.mj_setState(model, data, state, STATE_SPEC)
            mujoco.mj_forward(model, data)
            if not hit:
                return goal, ""
    finally:
        mujoco.mj_setState(model, data, state, STATE_SPEC)
        mujoco.mj_forward(model, data)
    return None, f"no reachable collision-free pose in the container in {n} samples (best IK {best:.3f} m)"


def hold_tilts(scene, k: int) -> tuple[float, ...]:
    """Approach tilts to try for a hold: from above for low open containers; from the side first when the container
    is tall (the arm reaches into or over it only with the gripper near horizontal)."""
    f = scene.full
    tall = max(f._objs[c]["extent"][5] - f._objs[c]["extent"][4] for c in scene.goal_of[scene.names[k]]["containers"]) > TALL_M
    return SIDE_TILTS_DEG if tall else TILTS_DEG


def build_start(env, seed: int, stage: str, *, rng: np.random.Generator | None = None, tries: int = 12,
                target: int | None = None, lift_m=(0.05, 0.15)) -> tuple[dict | None, list[dict]]:
    """A ``stage`` start for ``seed`` on an env made for ``r2g_<task>__<stage>``.
    -> (start or None, log). start: seed, stage, target, arm, state, rest_height (last row: pinch time), labels."""
    rng = rng or np.random.default_rng(seed)
    si = STAGES.index(stage)
    t0 = time.process_time()
    log: list[dict] = []
    yaws: list[float] = []
    for attempt in range(tries if si > R else 1):
        ok, why = reset_scene(env, seed, target)
        if not ok:
            return None, log + [{"attempt": attempt, "why": why}]
        model, data, ev = env.model, env.data, env._task_evaluator
        scene, cfg = ev.scene, ev.cfg
        k, arm = int(ev.target), int(ev.scene.arm[ev.target])
        rest = ev.tracker.rest_height.copy()
        info: dict = {"attempt": attempt, "target": k, "object": scene.names[k], "arm": ARMS[arm]}
        if ev.tracker.obs["in_goal"][k]:
            return None, log + [{**info, "why": "goal already met at reset"}]
        if si > R:
            if not yaws:
                yaws = closing_yaws(env, scene, k)
            # every closing direction at each tilt, upright first
            yaw = yaws[attempt % len(yaws)]
            tilts = TILTS_DEG if stage == "pick" else hold_tilts(scene, k)
            tilt = tilts[(attempt // len(yaws)) % len(tilts)]
            info.update(yaw=round(float(yaw), 3), tilt_deg=tilt)
            if stage == "pick":
                # low objects: the open fingers would be in the table with the pads at the centre of mass
                ok, gi = False, {"why": "no IK"}
                for raise_m in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
                    ok, gi = place_gripper(env, scene, k, pinch_grasp(env, scene, k, yaw, arm, raise_m, tilt), rng=rng)
                    if ok and _finger_contacts(model, data, scene):
                        ok, gi = False, {**gi, "why": "open fingers touch something"}
                    if ok:
                        info["raise_m"] = raise_m
                        break
                if ok:
                    act = hold_action(env, arm, gi["q"], 1.0)
                    for _ in range(10):
                        env.step(act, render_obs=False)
                    moved = float(np.linalg.norm(data.xipos[scene.bodies[k]] - rest[k, :3]))
                    if moved > 0.01:
                        ok, gi = False, {**gi, "why": f"target pushed {moved:.3f} m while settling"}
            else:
                a = int(scene.qpos_adr[k])
                pos, quat = data.qpos[a:a + 3].copy(), data.qpos[a + 3:a + 7].copy()
                pos[2] += float(rng.uniform(*lift_m))
                set_object_pose(env, scene, k, pos, quat)
                width = pinch_width(env, scene, k)  # before the hold turns the object
                ok, gi = hold_object(env, scene, k, pinch_grasp(env, scene, k, yaw, arm, 0.0, tilt), rng=rng)
                if ok and finger_gap(env, arm) > width + WEDGE_SLACK_M:
                    # both fingers touch it but did not close to its width: it is jammed across the open gripper
                    # (e.g. a screwdriver lengthwise) and would not fall when the gripper opens
                    ok, gi = False, {**gi, "why": f"wedged across the gripper, not pinched (fingers {finger_gap(env, arm):.3f} m apart)"}
                if ok:
                    goal, why = over_container_pose(env, scene, k, arm, rng)
                    if goal is None:  # no carry from this hold: try the next closing direction / tilt
                        ok, gi = False, {**gi, "why": why}
                    elif stage == "place":
                        ok, why = move_link(env, arm, goal, actions=96)
                        if ok:
                            act = idle_action(env)
                            for _ in range(16):
                                env.step(act, render_obs=False)
                        else:
                            gi = {**gi, "why": why}
            info.update({key: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                         for key, v in gi.items() if key in ("why", "drift", "speed", "held", "ik_pos_err")})
            if not ok:
                log.append(info)
                continue
        labels = labels_now(scene, data, rest, cfg)
        if int(labels[k]) != si:
            log.append({**info, "why": f"classified as {STAGES[int(labels[k])] if labels[k] < len(STAGES) else 'fallen'}"})
            continue
        obs = scene.observe(data, rest)
        pinch = float(obs["pinch_s"])
        if si >= C and pinch < cfg.pinch_s:
            log.append({**info, "why": f"only {pinch:.2f} s of pinch"})
            continue
        if obs["stock"].get("sim_fault"):
            log.append({**info, "why": "sim fault while building"})
            continue
        if time.process_time() - t0 > MAX_START_S:
            return None, log + [{**info, "why": f"build took {time.process_time() - t0:.1f} CPU s"}]
        rest_out = rest.copy()
        rest_out[-1, 0] = pinch
        log.append({**info, "ok": True, "pinch_s": round(pinch, 3), "cpu_s": round(time.process_time() - t0, 2)})
        return {"seed": int(seed), "stage": stage, "target": k, "arm": arm, "state": _state(env),
                "rest_height": rest_out, "labels": labels}, log
    return None, log


__all__ = ["SEED_BASE", "build_start", "labels_now", "reset_scene", "over_container_pose", "closing_yaws"]
