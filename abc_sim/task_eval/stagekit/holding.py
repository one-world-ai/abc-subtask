"""Putting a gripper on an object and making it hold the object, for building mid-task start states.

A grasp is a dict: ``arm`` (0/1), link_6 pose in the object's frame, the object's world orientation when it was held,
the finger openings and the object's mesh ``variant``. Generic keys are ``T_obj_link6`` / ``obj_quat``; the mug-flip
grasp library uses ``T_mug_link6`` / ``mug_quat``, which are accepted too.
"""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.task_eval.stagekit.geometry import T, inv
from abc_sim.task_eval.stagekit.robot import ARMS, GRIP_MAX, arm_ids, ik_link6
from abc_sim.task_eval.stagekit.scene import ObjectScene, is_desc


def _key(grasp: dict, generic: str, mug: str):
    return grasp[generic] if generic in grasp else grasp[mug]


def object_variant(model: mujoco.MjModel, body: int) -> str:
    """Mesh variant of an object body whose child body is named <kind>_<i>_<variant>_object."""
    for b in range(model.nbody):
        if int(model.body_parentid[b]) == body:
            name = model.body(b).name
            parts = name.split("_")
            if len(parts) >= 4 and name.endswith("_object"):
                return "_".join(parts[2:-1])
    return "?"


def grasp_from_state(model: mujoco.MjModel, data: mujoco.MjData, scene: ObjectScene, k: int, arm: int) -> dict:
    """The grasp ``arm`` has on object k right now (generic keys)."""
    ids = arm_ids(model, ARMS[arm])
    T6 = T(data.xpos[ids.link6], data.xquat[ids.link6])
    a = int(scene.qpos_adr[k])
    return {"arm": arm, "T_obj_link6": inv(scene.object_T(data, k)) @ T6,
            "obj_quat": np.array(data.qpos[a + 3:a + 7]),  # world orientation the object was held at
            "fingers": np.array([data.qpos[ids.finger_qadr[0]], data.qpos[ids.finger_qadr[1]]]),
            "variant": object_variant(model, scene.bodies[k])}


def hold_action(env, arm: int, q: np.ndarray, grip: float) -> np.ndarray:
    """14-D action: ``arm`` at joints q with gripper ``grip`` (1 open, 0 closed), the other arm at home, open."""
    act = np.zeros(14, np.float32)
    for i, ci in enumerate(env._ctrl_indices):  # current targets
        act[i] = env.data.ctrl[ci] / GRIP_MAX if i in env._gripper_set else env.data.ctrl[ci]
    other = 1 - arm
    act[other * 7:other * 7 + 6] = env.get_init_q()[other * 7:other * 7 + 6]
    act[other * 7 + 6] = 1.0
    act[arm * 7:arm * 7 + 6], act[arm * 7 + 6] = q, grip
    return act


def touches_other(model, data, scene: ObjectScene, k: int, arm: int) -> bool:
    """Object k touches something other than ``arm``'s fingers (table, container, other object, other arm)."""
    own = {arm * 2, arm * 2 + 1}
    for i in range(int(data.ncon)):
        c = data.contact[i]
        o1, o2 = scene.owner[int(c.geom1)], scene.owner[int(c.geom2)]
        for a, b in ((o1, o2), (o2, o1)):
            if a[0] == 0 and a[1] == k and not (b[0] == 1 and int(b[1]) in own):
                return True
    return False


def set_object_pose(env, scene: ObjectScene, k: int, pos, quat) -> None:
    model, data = env.model, env.data
    a, dof = int(scene.qpos_adr[k]), int(scene.dof_adr[k])
    data.qpos[a:a + 3], data.qpos[a + 3:a + 7] = pos, np.asarray(quat) / np.linalg.norm(quat)
    data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, data)


def place_gripper(env, scene: ObjectScene, k: int, grasp: dict, *, open_extra: float = 0.0,
                  rng: np.random.Generator | None = None) -> tuple[bool, dict]:
    """Teleport grasp['arm'] so its link_6 sits at the grasp pose on object k (as it is now); fingers at the grasp's
    opening plus ``open_extra`` (m). Returns (ok, info) with the joint solution."""
    model, data = env.model, env.data
    arm = int(grasp["arm"])
    T_goal = scene.object_T(data, k) @ _key(grasp, "T_obj_link6", "T_mug_link6")
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


def hold_object(env, scene: ObjectScene, k: int, grasp: dict, *, hold_steps: int = 30, close_steps: int = 10,
                max_drift_m: float = 0.03, max_speed: float = 0.05,
                rng: np.random.Generator | None = None) -> tuple[bool, dict]:
    """Make grasp['arm'] hold object k (already at its target pose) with ``grasp``: teleport the gripper (1 cm wider),
    close it for ``close_steps`` actions while the object is weightless, then hold ``hold_steps`` actions under
    gravity. Kept only if both fingers are on it, it drifted < ``max_drift_m``, moves slower than ``max_speed`` and
    touches nothing else. -> (ok, info)."""
    model, data = env.model, env.data
    arm = int(grasp["arm"])
    a, dof = int(scene.qpos_adr[k]), int(scene.dof_adr[k])
    obj_q = data.qpos[a:a + 7].copy()
    ok, info = place_gripper(env, scene, k, grasp, open_extra=0.01, rng=rng)
    if not ok:
        return False, info
    data.qpos[a:a + 7] = obj_q
    data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    if touches_other(model, data, scene, k, arm):
        return False, {**info, "why": "gripper/object pose collides with something"}
    act = hold_action(env, arm, info["q"], 0.0)
    sub = [b for b in range(model.nbody) if is_desc(model, b, scene.bodies[k])]
    gc = model.body_gravcomp[sub].copy()
    model.body_gravcomp[sub] = 1.0
    try:
        for _ in range(close_steps):  # close on the weightless object
            env.step(act, render_obs=False)
    finally:
        model.body_gravcomp[sub] = gc
    for _ in range(hold_steps):
        env.step(act, render_obs=False)
    obs = scene.contact_obs(data)
    drift = float(np.linalg.norm(data.qpos[a:a + 3] - obj_q[:3]))
    speed = float(np.linalg.norm(data.qvel[dof:dof + 3]))
    held = int(obs["grasp_arm"][k]) in (arm, 2)
    info.update(drift=round(drift, 4), speed=round(speed, 4), held=held)
    name = scene.names[k].split("_")[0]
    if not held:
        return False, {**info, "why": f"{name} not held by both fingers"}
    if drift > max_drift_m or speed > max_speed:
        return False, {**info, "why": f"{name} slipped (drift {drift:.3f} m, speed {speed:.3f} m/s)"}
    if touches_other(model, data, scene, k, arm):
        return False, {**info, "why": f"held {name} touches something else"}
    return True, info
