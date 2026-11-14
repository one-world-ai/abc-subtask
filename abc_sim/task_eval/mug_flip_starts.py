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


import mujoco
import numpy as np

from abc_sim.task_eval.mug_flip_stages import P, PL, R, ROT, STAGES, MugScene, StageConfig, classify
from abc_sim.task_eval.stagekit import holding as _holding
from abc_sim.task_eval.stagekit.geometry import T as _T  # noqa: F401  (old private names, used by tooling)
from abc_sim.task_eval.stagekit.geometry import axis_angle_quat as _axis_angle_quat
from abc_sim.task_eval.stagekit.geometry import inv as _inv  # noqa: F401
from abc_sim.task_eval.stagekit.geometry import qmul as _qmul
from abc_sim.task_eval.stagekit.geometry import quat_of as _quat  # noqa: F401
from abc_sim.task_eval.stagekit.holding import hold_action as _hold_action
from abc_sim.task_eval.stagekit.holding import object_variant, place_gripper  # noqa: F401
from abc_sim.task_eval.stagekit.holding import set_object_pose as _set_mug
from abc_sim.task_eval.stagekit.holding import touches_other as _touches_other  # noqa: F401
from abc_sim.task_eval.stagekit.robot import ARMS, GRIP_MAX, STATE_SPEC, ArmIds, arm_ids, ik_link6  # noqa: F401
from abc_sim.task_eval.stagekit.scene import is_desc as _is_desc  # noqa: F401


# -- mug lookups -----------------------------------------------------------------------------------------------------

def mug_variant(model: mujoco.MjModel, scene: MugScene, k: int) -> str:
    """Mesh variant of mug k (its child body is named mug_<i>_<variant>_object)."""
    return object_variant(model, scene.mug_bodies[k])


def mug_T(data: mujoco.MjData, scene: MugScene, k: int) -> np.ndarray:
    return scene.object_T(data, k)


# -- grasps ----------------------------------------------------------------------------------------------------------

def grasp_from_state(model: mujoco.MjModel, data: mujoco.MjData, scene: MugScene, k: int, arm: int) -> dict:
    """The grasp ``arm`` has on mug k right now: link_6 pose in the mug frame, the mug's world orientation and
    the finger openings (mug-library keys ``T_mug_link6`` / ``mug_quat``)."""
    g = _holding.grasp_from_state(model, data, scene, k, arm)
    return {"arm": g["arm"], "T_mug_link6": g["T_obj_link6"], "mug_quat": g["obj_quat"], "fingers": g["fingers"],
            "variant": g["variant"]}


# -- holding a mug ---------------------------------------------------------------------------------------------------

def hold_mug(env, scene: MugScene, k: int, grasp: dict, *, hold_steps: int = 30, close_steps: int = 10,
             rng: np.random.Generator | None = None) -> tuple[bool, dict]:
    """Make grasp['arm'] hold mug k (already at its target pose) with ``grasp``. -> (ok, info)."""
    return _holding.hold_object(env, scene, k, grasp, hold_steps=hold_steps, close_steps=close_steps, rng=rng)


# -- stage starts ----------------------------------------------------------------------------------------------------

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
