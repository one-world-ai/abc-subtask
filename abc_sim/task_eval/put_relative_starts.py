"""Build start states for the put_relative subtasks (see ``put_relative_stages``) in the simulator, for any seed.

A start is ``reset(seed)`` followed by a scripted setup that makes the stage's precondition true:

    reach   nothing (the reset scene)
    pick    one gripper open (top-down) around the mover, which still rests where it spawned; nothing touches it
    handoff cross-table episodes only: the giver G (arm on the mover's side) holds it top-down with the pinch centre
            on the mover's axis (``put_relative_handoff.giver_grasps``), lifted 5-15 cm above its spawn spot; kept
            only if a scripted handoff plan exists (G can carry it to the middle, V has a collision-free grasp)
    carry   same-side episodes: one gripper holding the mover top-down, lifted 5-15 cm above its spawn spot.
            Cross-table episodes: received by V in the air, G clear -- chained (a handoff start + the scripted
            handoff), else (grasp source 3) V holds it alone top-down in the middle
    place   one gripper holding the mover top-down, 3-10 cm above its rest height over a random point of the goal zone

Grasps are synthesized, not recorded: link_6 points straight down with the grasp site 1 cm below the mover's CoM,
at a random yaw; holding uses ``stagekit.hold_object`` (close on the weightless mover, then hold under gravity, with checks).
The arm is the one on the mover's side of the table (left for y > 0); for place, on the goal zone's side (cross-table
episodes: the receiver, over the zone's part on its own side).

Seeds whose reset raises (the randomizer cannot place the objects) or whose reset already satisfies the stock
directive are rejected (phase 0 of subtask_rewards/put_relative). The result restores exactly with
``env.reset_to_snapshot`` / ``env.reset_to_stage``.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from abc_sim.task_eval.put_relative_stages import LABELS, STAGES, C, H, R, StageConfig, classify
from abc_sim.task_eval.stagekit.geometry import inv
from abc_sim.task_eval.stagekit.holding import hold_action, hold_object, place_gripper, set_object_pose
from abc_sim.task_eval.stagekit.robot import GRIP_MAX, STATE_SPEC

SEED_BASE = 910_000_000  # start-set seeds: SEED_BASE + i
TABLE_BOUNDS = (0.39, 0.79, -0.46, 0.46)  # randomizer's object area (x0, x1, y0, y1)
_R_DOWN = np.diag([1.0, -1.0, -1.0])  # link_6 z (towards the fingers) -> world -z
_SITE_Z = 0.1347  # grasp site along link_6 z
# Grasp site 1 cm below the CoM: 12/12 holds vs 9/12 at the CoM (subtask_rewards/put_relative STATUS, repair log).
GRASP_BELOW_COM_M = 0.01
# After the hold check, one more chunk of holding: a start whose object sinks more than this in it is rejected
# (light objects creep through a pinch; median 0.4 mm/chunk, a few 5 mm/chunk and fall out).
MAX_CREEP_M = 0.001
CREEP_STEPS = 32


def topdown_grasp(data: mujoco.MjData, scene, k: int, yaw: float, arm: int,
                  below_com_m: float = GRASP_BELOW_COM_M) -> dict:
    """Grasp with link_6 pointing down, the grasp site ``GRASP_BELOW_COM_M`` below object k's CoM, rotated ``yaw``
    about world z; fingers open."""
    c = np.asarray(data.subtree_com[scene.bodies[k]], dtype=np.float64)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    T6 = np.eye(4)
    T6[:3, :3] = Rz @ _R_DOWN
    T6[:3, 3] = c + np.array([0.0, 0.0, _SITE_Z - below_com_m])
    return {"arm": int(arm), "T_obj_link6": inv(scene.object_T(data, k)) @ T6,
            "fingers": np.array([GRIP_MAX, GRIP_MAX]), "yaw": float(yaw)}


def _finger_contacts(model, data, scene) -> bool:
    for i in range(int(data.ncon)):
        c = data.contact[i]
        if scene.owner[int(c.geom1)][0] == 1 or scene.owner[int(c.geom2)][0] == 1:
            return True
    return False


def reset_scene(env, seed: int) -> tuple[bool, str]:
    """reset(seed) as a fresh episode; False + why for a seed the start sets skip."""
    env.forget_arm_state()
    try:
        env.reset(seed=seed, randomize=True)
    except RuntimeError as exc:
        return False, f"reset raised: {exc}"
    m = env.evaluate_task().metrics  # also begins the evaluator's plain episode (rest positions from the reset scene)
    if m.get("stock_success"):
        return False, "directive already satisfied at reset"
    return True, ""


def build_start(env, seed: int, stage: str, *, rng: np.random.Generator | None = None, tries: int = 8,
                lift_m=(0.05, 0.15), place_lift_m=(0.03, 0.10), zone_along=(0.10, 0.22), zone_perp=0.04,
                cfg: StageConfig | None = None) -> tuple[dict | None, list[dict]]:
    """Make a ``stage`` start for ``seed`` on an env made for a ``put_relative__<stage>`` task.
    -> (start or None, log of attempts). start: seed, stage, target, arm, MuJoCo state, rest positions, labels, info."""
    cfg = cfg or StageConfig()
    rng = rng or np.random.default_rng(seed)
    si = STAGES.index(stage)
    t_entry = time.process_time()
    if stage in ("handoff", "carry"):
        ok, why = reset_scene(env, seed)
        if not ok:
            return None, [{"attempt": 0, "why": why}]
        from abc_sim.task_eval.put_relative_handoff import sides

        sc = env._task_evaluator.scene
        cross = sides(sc.heights(env.data), sc.mover, sc.reference, sc.goal["axis"])[0]
        if stage == "handoff":
            if not cross:
                return None, [{"attempt": 0, "why": "zone on the mover's side (no handoff)"}]
            return build_handoff_start(env, seed, rng=rng, cfg=cfg)
        if cross:
            st, log = build_cross_carry_start(env, seed, rng=rng, cfg=cfg, t0=t_entry)
            if st is not None and time.process_time() - t_entry > MAX_START_S:
                return None, log + [{"why": f"build took {time.process_time() - t_entry:.1f} CPU s (> {MAX_START_S})"}]
            return st, log
    log: list[dict] = []
    for attempt in range(tries if si > R else 1):
        ok, why = reset_scene(env, seed)
        if not ok:
            return None, [{"attempt": attempt, "why": why}]
        model, data, ev = env.model, env.data, env._task_evaluator
        scene = ev.scene
        k = scene.mover
        rest = scene.heights(data)
        arm = 0 if rest[k, 1] > 0 else 1
        info: dict = {"attempt": attempt, "target": k, "mover": scene.names[k], "arm": ("left", "right")[arm]}
        if si > R:
            # Open fingers around a resting object go no lower than its CoM (lower ones hit the table).
            grasp = topdown_grasp(data, scene, k, float(rng.uniform(0.0, np.pi)), arm,
                                  below_com_m=0.0 if stage == "pick" else GRASP_BELOW_COM_M)
            info["yaw"] = round(grasp["yaw"], 3)
            if stage == "pick":
                ok, gi = place_gripper(env, scene, k, grasp, rng=rng)
                if ok and _finger_contacts(model, data, scene):
                    ok, gi = False, {**gi, "why": "open fingers touch something"}
                if ok:
                    act = hold_action(env, arm, gi["q"], 1.0)
                    for _ in range(10):
                        env.step(act, render_obs=False)
                    moved = float(np.linalg.norm(scene.heights(data)[k] - rest[k]))
                    if moved > 0.01:
                        ok, gi = False, {**gi, "why": f"mover pushed {moved:.3f} m while settling"}
            else:
                a = int(scene.qpos_adr[k])
                quat = data.qpos[a + 3:a + 7].copy()
                if stage == "carry":
                    zone_y = (rest[scene.reference, :2] + np.asarray(scene.goal["axis"]) * 0.14)[1]
                    if np.sign(zone_y) != np.sign(rest[k, 1]):
                        # One arm holding from above can't reach across the table (subtask_rewards/put_relative:
                        # 0/13 cross-table carries planned); such episodes need a handoff subtask.
                        return None, [{**info, "why": "zone across the table (needs a handoff)"}]
                    target = rest[k].copy()
                    target[2] += float(rng.uniform(*lift_m))
                else:  # place: over a random point of the goal zone
                    axis = np.asarray(scene.goal["axis"], dtype=np.float64)
                    perp = np.array([-axis[1], axis[0]])
                    ref_xy = rest[scene.reference, :2]
                    xy = ref_xy + axis * rng.uniform(*zone_along) + perp * rng.uniform(-zone_perp, zone_perp)
                    x0, x1, y0, y1 = TABLE_BOUNDS
                    if not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1):
                        log.append({**info, "why": "zone point off the object area"})
                        continue
                    target = np.array([xy[0], xy[1], rest[k, 2] + rng.uniform(*place_lift_m)])
                    info["zone_xy"] = [round(float(v), 3) for v in xy]
                    arm = 0 if xy[1] > 0 else 1  # the arm on the zone's side holds it
                    from abc_sim.task_eval.put_relative_handoff import sides

                    cross, _, V = sides(rest, k, scene.reference, scene.goal["axis"])
                    if cross:  # cross-table: the receiver holds it after the handoff (zones can straddle the middle)
                        if xy[1] * (1.0 if V == 0 else -1.0) <= 0.0:  # V reaching over G's side sags (seed +1)
                            log.append({**info, "why": "zone point on the giver's side"})
                            continue
                        arm = V
                    info["arm"] = ("left", "right")[arm]
                set_object_pose(env, scene, k, target, quat)
                grasp = topdown_grasp(data, scene, k, grasp["yaw"], arm)  # re-anchored at the moved mover
                ok, gi = hold_object(env, scene, k, grasp, rng=rng)
                if ok:
                    z0 = float(data.qpos[a + 2])
                    act = hold_action(env, arm, gi["q"], 0.0)
                    for _ in range(CREEP_STEPS):
                        env.step(act, render_obs=False)
                    creep = z0 - float(data.qpos[a + 2])
                    gi["creep_mm"] = round(1000 * creep, 2)
                    if creep > MAX_CREEP_M:
                        ok, gi = False, {**gi, "why": f"object creeps {1000 * creep:.1f} mm per chunk in the hand"}
            info.update({key: v for key, v in gi.items() if key != "q"})
            if not ok:
                log.append(info)
                continue
        obs = scene.observe(data, rest)
        labels = classify(obs, np.full(len(scene.names), R), cfg)
        labels = classify(obs, labels, cfg)
        info.update(lift=round(float(obs["lift"][k]), 3), dist=round(float(obs["dist"][k]), 3),
                    zone_gap=round(float(obs["gap"][k]), 3), label=LABELS[int(labels[k])])
        if int(labels[k]) != si:
            log.append({**info, "why": f"stage test says {info['label']}, not {stage}"})
            continue
        state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        log.append({**info, "ok": True})
        return {"seed": int(seed), "stage": stage, "target": int(k), "arm": int(arm) if si > R else -1,
                "state": state, "rest_height": rest, "labels": np.asarray(labels), "info": info}, log
    return None, log



# ------------------------------------------------------------------------------------------- cross-table episodes

HANDOFF_BUDGET_S = 9.0  # CPU seconds per handoff start (G3: <= 10 s)
# CPU s for the handoff start of a chained cross-table carry start, and for the whole start (G3 repairs 1-2: with
# 6 s and no total cap the slowest starts took 15 s; 4 s: 12.8 s)
# 3 s, cap 8.5 s: 12.6 s -- the resets before the timer started). Round 3: the timer starts at build_start's entry,
# cap 7.5 s, and a start whose build took more than MAX_START_S is discarded.
CHAIN_BUDGET_S = 3.0
CARRY_TOTAL_S = 7.5
MAX_START_S = 10.0


def _state(env) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(env.model, STATE_SPEC))
    mujoco.mj_getState(env.model, env.data, state, STATE_SPEC)
    return state


def _labels(scene, data, rest, cfg) -> np.ndarray:
    obs = scene.observe(data, rest)
    labels = np.full(len(scene.names), R)
    for _ in range(3):
        labels = classify(obs, labels, cfg)
    return labels


def hold_lifted(env, scene, k: int, arm: int, com_target, rng, *, deadline: float | None = None) -> tuple[bool, dict]:
    """Move mover k (weightless) so its origin is at ``com_target`` (orientation kept) and let ``arm`` hold it with the
    first of ``giver_grasps`` (pinch centre on the mover's axis) that is reachable, passes ``hold_object`` and creeps
    <= MAX_CREEP_M in one more chunk. -> (ok, info)."""
    from abc_sim.task_eval.inhand_starts import reachable, restore, snapshot, teleport_arm
    from abc_sim.task_eval.put_relative_handoff import MoverView, giver_grasps

    data = env.data
    a = int(scene.qpos_adr[k])
    set_object_pose(env, scene, k, np.asarray(com_target), data.qpos[a + 3:a + 7].copy())
    view = MoverView(env.model, scene.names[k], scene.joints[k])
    snap = snapshot(env)
    whys = []
    for name, T6 in giver_grasps(env, view, rng):
        if deadline is not None and time.process_time() > deadline:
            whys.append("time budget")
            break
        restore(env, snap)
        Tt, q = reachable(env, arm, T6)
        if Tt is None:
            whys.append(f"{name}: no IK")
            continue
        teleport_arm(env, arm, q)
        grasp = {"arm": int(arm), "T_obj_link6": inv(scene.object_T(data, k)) @ Tt,
                 "fingers": np.array([GRIP_MAX - 0.01, GRIP_MAX - 0.01])}
        ok, gi = hold_object(env, scene, k, grasp, rng=rng)
        if not ok:
            whys.append(f"{name}: {gi.get('why')}")
            continue
        z0 = float(data.qpos[a + 2])
        act = hold_action(env, arm, gi["q"], 0.0)
        for _ in range(CREEP_STEPS):
            env.step(act, render_obs=False)
        creep = z0 - float(data.qpos[a + 2])
        if creep > MAX_CREEP_M:
            whys.append(f"{name}: creeps {1000 * creep:.1f} mm per chunk")
            continue
        return True, {"grasp": name, "creep_mm": round(1000 * creep, 2), "drift": gi.get("drift")}
    return False, {"why": "; ".join(whys[-3:]) or "no grasp"}


def build_handoff_start(env, seed: int, *, rng: np.random.Generator | None = None, cfg: StageConfig | None = None,
                        tries: int = 3, lift_m=(0.05, 0.15), budget_s: float = HANDOFF_BUDGET_S,
                        keep_plan: bool = False):
    """A handoff start for a cross-table seed: G holds the mover lifted 5-15 cm above its spawn spot and a scripted
    handoff plan exists. -> (start or None, log). ``keep_plan`` adds the plan (for chaining) under ``plan``."""
    from abc_sim.task_eval.put_relative_handoff import MoverView, plan_handoff, sides, zone_avoider

    cfg = cfg or StageConfig()
    rng = rng or np.random.default_rng(seed)
    t0 = time.process_time()
    deadline = t0 + budget_s
    log = []
    for attempt in range(tries):
        ok, why = reset_scene(env, seed)
        if not ok:
            return None, [{"attempt": attempt, "why": why}]
        ev = env._task_evaluator
        scene, data = ev.scene, env.data
        k = scene.mover
        rest = scene.heights(data)
        cross, G, V = sides(rest, k, scene.reference, scene.goal["axis"])
        info = {"attempt": attempt, "target": k, "mover": scene.names[k], "arm": ("left", "right")[G],
                "receiver": ("left", "right")[V]}
        target = rest[k].copy()
        target[2] += float(rng.uniform(*lift_m))
        ok, gi = hold_lifted(env, scene, k, G, target, rng, deadline=deadline)
        info.update(gi)
        if not ok:
            log.append(info)
            if time.process_time() > deadline:
                break
            continue
        view = MoverView(env.model, scene.names[k], scene.joints[k])
        plans = plan_handoff(env, view, G, deadline=deadline, avoid=zone_avoider(scene, rest))
        if not plans:
            log.append({**info, "why": "no handoff plan" + (" (time budget)" if time.process_time() > deadline else "")})
            if time.process_time() > deadline:
                break
            continue
        p = plans[0]
        info.update(handoff_point=p["point"], handoff_yaw=round(float(np.rad2deg(p["yaw"]))),
                    handoff_tilt=round(float(np.rad2deg(p["tilt"]))), recv_grasps=[g[0] for g in p["grasps"]])
        labels = _labels(scene, data, rest, cfg)
        obs = scene.observe(data, rest)
        info.update(lift=round(float(obs["lift"][k]), 3), label=LABELS[int(labels[k])],
                    build_cpu_s=round(time.process_time() - t0, 2))
        if int(labels[k]) != H:
            log.append({**info, "why": f"stage test says {info['label']}, not handoff"})
            continue
        log.append({**info, "ok": True})
        start = {"seed": int(seed), "stage": "handoff", "target": int(k), "arm": int(G), "state": _state(env),
                 "rest_height": rest, "labels": np.asarray(labels), "info": info}
        if keep_plan:
            start["plan"] = p
        return start, log
    return None, log or [{"why": "no attempt"}]


def build_cross_carry_start(env, seed: int, *, rng: np.random.Generator | None = None,
                            cfg: StageConfig | None = None, budget_s: float = CHAIN_BUDGET_S,
                            total_s: float = CARRY_TOTAL_S, t0: float | None = None):
    """A carry start for a cross-table seed: V holds the mover alone in the air, G clear. Source 1 (chain): a
    handoff start + the scripted handoff, validated (received in the air, creep, stage test). Otherwise source 3: V
    holds it top-down (pinch centre on its axis) 13-17 cm above the table in the middle (x 0.50-0.60, |y| <= 3 cm)."""
    from abc_sim.task_eval.put_relative_handoff import MoverView, TABLE_Z, run_handoff

    cfg = cfg or StageConfig()
    rng = rng or np.random.default_rng(seed)
    t0 = time.process_time() if t0 is None else t0
    log = []
    st, hlog = build_handoff_start(env, seed, rng=rng, cfg=cfg, budget_s=budget_s, keep_plan=True, tries=2)
    log += hlog
    ev = env._task_evaluator
    scene, data = ev.scene, env.data
    why = "no handoff start" if st is None else ""
    if st is not None:
        k, G, rest = st["target"], st["arm"], st["rest_height"]
        view = MoverView(env.model, scene.names[k], scene.joints[k])
        ok, why = run_handoff(env, view, G, st["plan"])
        if ok:
            a = int(scene.qpos_adr[k])
            z0 = float(data.qpos[a + 2])
            from abc_sim.task_eval.inhand_starts import current_action, run

            run(env, current_action(env), CREEP_STEPS)
            creep = z0 - float(data.qpos[a + 2])
            labels = _labels(scene, data, rest, cfg)
            if creep > MAX_CREEP_M:
                ok, why = False, f"creeps {1000 * creep:.1f} mm per chunk in V's hand"
            elif int(labels[k]) != C:
                ok, why = False, f"chain ends in {LABELS[int(labels[k])]}"
        if ok:
            info = {**st["info"], "source": "chained handoff solver", "arm": ("left", "right")[1 - G],
                    "creep_mm": round(1000 * creep, 2), "build_cpu_s": round(time.process_time() - t0, 2)}
            log.append({**info, "ok": True})
            return {"seed": int(seed), "stage": "carry", "target": int(k), "arm": int(1 - G), "state": _state(env),
                    "rest_height": rest, "labels": np.asarray(labels), "info": info}, log
    log.append({"why": f"chain: {why}"})
    # Source 3: V holds it alone in the middle.
    from abc_sim.task_eval.put_relative_handoff import sides, zone_avoider

    for attempt in range(3):
        ok, why = reset_scene(env, seed)
        if not ok:
            return None, log + [{"why": why}]
        ev = env._task_evaluator
        scene, data = ev.scene, env.data
        k = scene.mover
        rest = scene.heights(data)
        _, G, V = sides(rest, k, scene.reference, scene.goal["axis"])
        target = np.array([rng.uniform(0.50, 0.60), rng.uniform(-0.03, 0.03),
                           rest[k, 2] + rng.uniform(0.13, 0.17)])
        if zone_avoider(scene, rest)(target[:2]):  # zone over the middle: hold it on G's side of the zone instead
            sG = 1.0 if G == 0 else -1.0
            target[:2] = [rng.uniform(0.50, 0.60), sG * rng.uniform(0.13, 0.17)]
            if zone_avoider(scene, rest)(target[:2]):
                log.append({"attempt": attempt, "why": "no fallback point outside the zone"})
                continue
        ok, gi = hold_lifted(env, scene, k, V, target, rng, deadline=t0 + total_s)
        info = {"attempt": attempt, "target": k, "mover": scene.names[k], "arm": ("left", "right")[V],
                "source": "sampled receiver grasp in the middle (grasp source 3)", **gi}
        if not ok:
            log.append(info)
            if time.process_time() > t0 + total_s:
                break
            continue
        labels = _labels(scene, data, rest, cfg)
        info.update(label=LABELS[int(labels[k])], build_cpu_s=round(time.process_time() - t0, 2))
        if int(labels[k]) != C:
            log.append({**info, "why": f"stage test says {info['label']}, not carry"})
            continue
        log.append({**info, "ok": True})
        return {"seed": int(seed), "stage": "carry", "target": int(k), "arm": int(V), "state": _state(env),
                "rest_height": rest, "labels": np.asarray(labels), "info": info}, log
    return None, log
