"""Build start states for the mug-flip subtasks (see ``mug_flip_stages``) directly in the simulator, for any seed.

A start is ``reset(seed)`` followed by a scripted setup that makes the stage's precondition true:

    reach   nothing (the normal scene)
    pick    one gripper open around a mug that still rests on the table
    rotate  one gripper holding a mug lifted 5-15 cm, still tilted (its spawn tilt, or partly turned)
    place   one gripper holding a mug upright, 5-15 cm above the table

A held mug is made the way a hand would hold it: the mug is put at the target pose, the arm is teleported (IK) so
the gripper sits at a grasp pose taken from a grasp library, the fingers close while the mug is weightless, then the
mug is held under gravity for ``hold_steps`` actions. The start is kept only if the mug stays in that gripper
(both fingers on it) without slipping, and the stage tests agree (``classify`` gives the requested stage).

Grasp library: gripper (link_6) poses in the mug's frame plus finger openings, taken from states where a policy was
really holding a mug (``grasp_from_state``); simulation/experiments/mug_stages/extract_grasps.py builds it from
the recorded OpenWAM episodes. Grasps from the same mug model are preferred.

The result is a dict that ``env.reset_to_snapshot(seed, state)`` restores exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from abc_sim.task_eval.mug_flip_stages import P, PL, R, ROT, STAGES, MugScene, StageConfig, classify

ARMS = ("left", "right")
GRIP_MAX = 0.0475  # finger joint travel; env action 1 = open, 0 = closed
STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


# -- small rigid-transform helpers -----------------------------------------------------------------------------------

def _T(pos, quat) -> np.ndarray:
    T = np.eye(4)
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat, np.float64))
    T[:3, :3], T[:3, 3] = m.reshape(3, 3), pos
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _quat(Rm: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(Rm).reshape(9))
    return q


def _axis_angle_quat(axis, angle) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, np.float64), float(angle))
    return q


def _qmul(a, b) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mulQuat(q, np.asarray(a, np.float64), np.asarray(b, np.float64))
    return q


# -- model lookups ---------------------------------------------------------------------------------------------------

@dataclass
class ArmIds:
    qadr: np.ndarray  # 6 arm joints
    dadr: np.ndarray
    rng: np.ndarray
    finger_qadr: tuple[int, int]
    finger_dadr: tuple[int, int]
    link6: int


def arm_ids(model: mujoco.MjModel, arm: str) -> ArmIds:
    j = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_joint{k}") for k in range(1, 7)]
    f = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{s}_finger") for s in ("left", "right")]
    return ArmIds(
        qadr=np.array([model.jnt_qposadr[i] for i in j]), dadr=np.array([model.jnt_dofadr[i] for i in j]),
        rng=np.array([model.jnt_range[i] for i in j]),
        finger_qadr=tuple(int(model.jnt_qposadr[i]) for i in f), finger_dadr=tuple(int(model.jnt_dofadr[i]) for i in f),
        link6=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_link_6"),
    )


def mug_variant(model: mujoco.MjModel, scene: MugScene, k: int) -> str:
    """Mesh variant of mug k (its child body is named mug_<i>_<variant>_object)."""
    body = scene.mug_bodies[k]
    for b in range(model.nbody):
        if int(model.body_parentid[b]) == body:
            name = model.body(b).name
            parts = name.split("_")
            if len(parts) >= 4 and name.endswith("_object"):
                return "_".join(parts[2:-1])
    return "?"


def mug_T(data: mujoco.MjData, scene: MugScene, k: int) -> np.ndarray:
    a = int(scene.qpos_adr[k])
    return _T(data.qpos[a:a + 3], data.qpos[a + 3:a + 7])


# -- grasps ----------------------------------------------------------------------------------------------------------

def grasp_from_state(model: mujoco.MjModel, data: mujoco.MjData, scene: MugScene, k: int, arm: int) -> dict:
    """The grasp ``arm`` has on mug k right now: link_6 pose in the mug frame, the mug's world orientation and
    the finger openings."""
    ids = arm_ids(model, ARMS[arm])
    T6 = _T(data.xpos[ids.link6], data.xquat[ids.link6])
    a = int(scene.qpos_adr[k])
    return {"arm": arm, "T_mug_link6": _inv(mug_T(data, scene, k)) @ T6,
            "mug_quat": np.array(data.qpos[a + 3:a + 7]),  # world orientation the mug was held at
            "fingers": np.array([data.qpos[ids.finger_qadr[0]], data.qpos[ids.finger_qadr[1]]]),
            "variant": mug_variant(model, scene, k)}


def ik_link6(model: mujoco.MjModel, data: mujoco.MjData, arm: str, T_goal: np.ndarray, *, seeds: int = 6,
             iters: int = 200, rng: np.random.Generator | None = None) -> tuple[np.ndarray | None, float, float]:
    """Damped-least-squares IK of link_6's full pose. -> (6 joint values, position error m, rotation error rad)."""
    ids = arm_ids(model, arm)
    rng = rng or np.random.default_rng(0)
    d2 = mujoco.MjData(model)
    best = (None, np.inf, np.inf)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for s in range(seeds):
        d2.qpos[:] = data.qpos
        if s > 0:
            d2.qpos[ids.qadr] = rng.uniform(ids.rng[:, 0], ids.rng[:, 1])
        for _ in range(iters):
            mujoco.mj_kinematics(model, d2)
            mujoco.mj_comPos(model, d2)
            p, Rm = d2.xpos[ids.link6], d2.xmat[ids.link6].reshape(3, 3)
            ep = T_goal[:3, 3] - p
            Re = T_goal[:3, :3] @ Rm.T
            er = np.empty(3)
            mujoco.mju_quat2Vel(er, _quat(Re), 1.0)
            mujoco.mj_jacBody(model, d2, jp, jr, ids.link6)
            J = np.vstack([jp[:, ids.dadr], jr[:, ids.dadr]])
            e = np.concatenate([ep, 0.5 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            d2.qpos[ids.qadr] = np.clip(d2.qpos[ids.qadr] + np.clip(dq, -0.2, 0.2), ids.rng[:, 0], ids.rng[:, 1])
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
        pe, re = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
        if pe + 0.05 * re < best[1] + 0.05 * best[2]:
            best = (d2.qpos[ids.qadr].copy(), pe, re)
        if pe < 2e-3 and re < 0.03:
            break
    return best


# -- holding a mug ---------------------------------------------------------------------------------------------------

def _hold_action(env, arm: int, q: np.ndarray, grip: float) -> np.ndarray:
    """14-D action: ``arm`` at joints q with gripper ``grip`` (1 open, 0 closed), the other arm where it is now."""
    act = np.zeros(14, np.float32)
    for i, ci in enumerate(env._ctrl_indices):  # current targets
        act[i] = env.data.ctrl[ci] / GRIP_MAX if i in env._gripper_set else env.data.ctrl[ci]
    other = 1 - arm
    act[other * 7:other * 7 + 6] = env.get_init_q()[other * 7:other * 7 + 6]
    act[other * 7 + 6] = 1.0
    act[arm * 7:arm * 7 + 6], act[arm * 7 + 6] = q, grip
    return act


def _touches_other(model, data, scene: MugScene, k: int, arm: int) -> bool:
    """Mug k touches something other than ``arm``'s fingers (table, tray, other mug, other arm)."""
    own = {arm * 2, arm * 2 + 1}
    for i in range(int(data.ncon)):
        c = data.contact[i]
        o1, o2 = scene.owner[int(c.geom1)], scene.owner[int(c.geom2)]
        for a, b in ((o1, o2), (o2, o1)):
            if a[0] == 0 and a[1] == k and not (b[0] == 1 and int(b[1]) in own):
                return True
    return False


def place_gripper(env, scene: MugScene, k: int, grasp: dict, *, open_extra: float = 0.0,
                  rng: np.random.Generator | None = None) -> tuple[bool, dict]:
    """Teleport grasp['arm'] so its link_6 sits at the grasp pose on mug k (as it is now); fingers at the grasp's
    opening plus ``open_extra`` (m). Returns (ok, info) with the joint solution."""
    model, data = env.model, env.data
    arm = int(grasp["arm"])
    T_goal = mug_T(data, scene, k) @ grasp["T_mug_link6"]
    q, pe, re = ik_link6(model, data, ARMS[arm], T_goal, rng=rng)
    if q is None or pe > 3e-3 or re > 0.05:
        return False, {"why": f"no IK (pos err {pe:.4f} m, rot err {re:.3f} rad)"}
    ids = arm_ids(model, ARMS[arm])
    data.qpos[ids.qadr], data.qvel[ids.dadr] = q, 0.0
    f = np.clip(np.abs(grasp["fingers"]) + open_extra, 0.0, GRIP_MAX)
    data.qpos[ids.finger_qadr[0]], data.qpos[ids.finger_qadr[1]] = f[0], -f[1]
    data.qvel[list(ids.finger_dadr)] = 0.0
    mujoco.mj_forward(model, data)
    return True, {"q": q, "ik_pos_err": pe, "ik_rot_err": re}


def hold_mug(env, scene: MugScene, k: int, grasp: dict, *, hold_steps: int = 30, close_steps: int = 10,
             rng: np.random.Generator | None = None) -> tuple[bool, dict]:
    """Make grasp['arm'] hold mug k (already at its target pose) with ``grasp``. -> (ok, info)."""
    model, data = env.model, env.data
    arm = int(grasp["arm"])
    a = int(scene.qpos_adr[k])
    mug_q = data.qpos[a:a + 7].copy()
    ok, info = place_gripper(env, scene, k, grasp, open_extra=0.01, rng=rng)
    if not ok:
        return False, info
    data.qpos[a:a + 7] = mug_q
    dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{scene.mug_names[k]}_jnt")])
    data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    if _touches_other(model, data, scene, k, arm):
        return False, {**info, "why": "gripper/mug pose collides with something"}
    act = _hold_action(env, arm, info["q"], 0.0)
    sub = [b for b in range(model.nbody) if _is_desc(model, b, scene.mug_bodies[k])]
    gc = model.body_gravcomp[sub].copy()
    model.body_gravcomp[sub] = 1.0
    try:
        for _ in range(close_steps):  # close on the weightless mug
            env.step(act, render_obs=False)
    finally:
        model.body_gravcomp[sub] = gc
    for _ in range(hold_steps):
        env.step(act, render_obs=False)
    obs = scene.observe(data, np.zeros(len(scene.mug_names)))
    drift = float(np.linalg.norm(data.qpos[a:a + 3] - mug_q[:3]))
    speed = float(np.linalg.norm(data.qvel[dof:dof + 3]))
    held = int(obs["grasp_arm"][k]) in (arm, 2)
    info.update(drift=round(drift, 4), speed=round(speed, 4), held=held)
    if not held:
        return False, {**info, "why": "mug not held by both fingers"}
    if drift > 0.03 or speed > 0.05:
        return False, {**info, "why": f"mug slipped (drift {drift:.3f} m, speed {speed:.3f} m/s)"}
    if _touches_other(model, data, scene, k, arm):
        return False, {**info, "why": "held mug touches something else"}
    return True, info


def _is_desc(model, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = int(model.body_parentid[b])
    return False


# -- stage starts ----------------------------------------------------------------------------------------------------

def _set_mug(env, scene: MugScene, k: int, pos, quat) -> None:
    model, data = env.model, env.data
    a = int(scene.qpos_adr[k])
    data.qpos[a:a + 3], data.qpos[a + 3:a + 7] = pos, np.asarray(quat) / np.linalg.norm(quat)
    dof = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{scene.mug_names[k]}_jnt")])
    data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, data)


def _tilt_toward_upright(quat: np.ndarray, frac: float) -> np.ndarray:
    """Rotate the mug a fraction of the way from ``quat`` to upright, about the horizontal axis that does it."""
    Rm = np.empty(9)
    mujoco.mju_quat2Mat(Rm, quat)
    z = Rm.reshape(3, 3)[:, 2]
    axis = np.cross(z, [0.0, 0.0, 1.0])
    n = np.linalg.norm(axis)
    if n < 1e-6:  # exactly upside down: any horizontal axis
        axis, n = np.array([1.0, 0.0, 0.0]), 1.0
    angle = float(np.arccos(np.clip(z[2], -1, 1)))
    return _qmul(_axis_angle_quat(axis / n, frac * angle), quat)


def _upright_quat(quat: np.ndarray) -> np.ndarray:
    return _tilt_toward_upright(quat, 1.0)


def build_start(env, seed: int, stage: str, library: list[dict], *, rng: np.random.Generator | None = None,
                mug: int | None = None, arm: int | None = None, lift_m=(0.05, 0.15), tilt_frac=(0.0, 0.3),
                yaw_jitter: float = 0.5,
                tries: int = 12, cfg: StageConfig | None = None) -> tuple[dict | None, list[dict]]:
    """Make a ``stage`` start for ``seed``. -> (start or None, log of attempts).

    start: seed, stage, mug, arm, grasp index, MuJoCo state (``mjSTATE_INTEGRATION``), rest heights, details."""
    cfg = cfg or StageConfig()
    rng = rng or np.random.default_rng(seed)
    si = STAGES.index(stage)
    log: list[dict] = []
    for attempt in range(tries if si > R else 1):
        env.forget_arm_state()
        env.reset(seed=seed, randomize=True)
        env.evaluate_task()  # the stock evaluator's rest heights come from the reset scene
        model, data = env.model, env.data
        scene = MugScene(model)
        rest = scene.heights(data)
        m = len(scene.mug_names)
        k = int(rng.integers(m)) if mug is None else int(mug)
        info: dict = {"attempt": attempt, "mug": k, "variant": mug_variant(model, scene, k)}
        if si == R:
            labels = np.full(m, R)
            g_idx, a_used = -1, -1
        else:
            # Grasps recorded in this stage (held tilted for rotate and pick, held upright for place), by the arm on
            # the mug's side of the table unless ``arm`` is given.
            a_side = arm if arm is not None else (0 if data.qpos[int(scene.qpos_adr[k]) + 1] > 0 else 1)
            want = STAGES[si]
            cands = [i for i, g in enumerate(library) if g["arm"] == a_side and g.get("stage", "rotate") == want]
            same = [i for i in cands if library[i]["variant"] == info["variant"]]
            pool = same or cands
            if not pool:
                return None, [{"why": "empty grasp library"}]
            g_idx = int(rng.choice(pool))
            grasp = library[g_idx]
            a_used = int(grasp["arm"])
            a = int(scene.qpos_adr[k])
            pos, quat = data.qpos[a:a + 3].copy(), data.qpos[a + 3:a + 7].copy()
            if si == P:
                # The mug rests as in the recorded grasp (its resting orientation, turned by a random yaw), then the
                # open gripper is put at that grasp, 1.2 cm wider, and must not move the mug.
                q_rest = _qmul(_axis_angle_quat([0, 0, 1], float(rng.uniform(-yaw_jitter, yaw_jitter))), grasp["mug_quat"])
                _set_mug(env, scene, k, pos + [0.0, 0.0, 0.01], q_rest)
                act = _hold_action(env, a_used, env.get_init_q()[a_used * 7:a_used * 7 + 6], 1.0)
                for _ in range(30):  # let the mug settle on the table with the arms at home
                    env.step(act, render_obs=False)
                pos, quat = data.qpos[a:a + 3].copy(), data.qpos[a + 3:a + 7].copy()
                ok, gi = place_gripper(env, scene, k, grasp, open_extra=0.012, rng=rng)
                if ok:
                    data.qpos[a:a + 7] = np.concatenate([pos, quat])
                    mujoco.mj_forward(model, data)
                    if scene.observe(data, rest)["touched"][k]:  # a finger already on it would push the mug
                        ok, gi = False, {**gi, "why": "open fingers already touch the mug"}
                    else:
                        act = _hold_action(env, a_used, gi["q"], 1.0)
                        for _ in range(10):
                            env.step(act, render_obs=False)
                        moved = float(np.linalg.norm(data.qpos[a:a + 3] - pos))
                        if moved > 0.01:
                            ok, gi = False, {**gi, "why": f"mug pushed {moved:.3f} m while settling"}
            else:
                # The recorded hold's world orientation (how the policy really held the mug, so the gripper comes
                # from a reachable direction), turned by a small random yaw and moved to this mug's spot.
                lift = float(rng.uniform(*lift_m))
                q_new = _qmul(_axis_angle_quat([0, 0, 1], float(rng.uniform(-yaw_jitter, yaw_jitter))), grasp["mug_quat"])
                if si == ROT and tilt_frac[1] > 0:
                    q_new = _tilt_toward_upright(q_new, float(rng.uniform(*tilt_frac)))
                _set_mug(env, scene, k, pos + [0.0, 0.0, lift + 0.02], q_new)
                info.update(lift_target=round(lift, 3))
                ok, gi = hold_mug(env, scene, k, grasp, rng=rng)
            info.update({key: v for key, v in gi.items() if key != "q"}, grasp=g_idx, arm=ARMS[a_used])
            if not ok:
                log.append(info)
                continue
            obs = scene.observe(data, rest)
            labels = classify(obs, np.full(m, R), cfg)
            labels = classify(obs, labels, cfg)
            info.update(upz=round(float(obs["upz"][k]), 3), lift=round(float(obs["lift"][k]), 3),
                        dist=round(float(obs["dist"][k]), 3), label=STAGES[int(labels[k])] if labels[k] < len(STAGES) else "fallen")
            if int(labels[k]) != si:
                log.append({**info, "why": f"stage test says {info['label']}, not {stage}"})
                continue
        state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        log.append({**info, "ok": True})
        return {"seed": int(seed), "stage": stage, "mug": k, "arm": a_used, "grasp": g_idx, "state": state,
                "rest_height": rest, "labels": np.asarray(labels), "info": info}, log
    return None, log


DEFAULT_GRASP_LIBRARY = __import__("pathlib").Path(__file__).parent / "data" / "mug_flip_grasps.npz"


def load_grasp_library(path=DEFAULT_GRASP_LIBRARY) -> list[dict]:
    """The grasps recorded from OpenWAM episodes (see the module docstring), as a list of dicts."""
    z = np.load(path)
    return [{"arm": int(a), "T_mug_link6": T, "fingers": f, "mug_quat": q, "stage": str(s), "variant": str(v),
             "source": str(src)}
            for a, T, f, q, s, v, src in zip(z["arm"], z["T_mug_link6"], z["fingers"], z["mug_quat"], z["stage"],
                                             z["variant"], z["source"])]
