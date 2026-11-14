"""The bimanual YAM arms: joint lookups and link_6 inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from abc_sim.task_eval.stagekit.geometry import quat_of

ARMS = ("left", "right")
GRIP_MAX = 0.0475  # finger joint travel; env action 1 = open, 0 = closed
STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


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


def actuator_limits(model: mujoco.MjModel, arm: str) -> np.ndarray:
    """(6, 2) joint ranges the arm's position actuators can actually command: the joint range intersected with each
    actuator's ctrlrange (targets outside it are clipped, so an IK solution beyond it is not reachable)."""
    ids = arm_ids(model, arm)
    lim = ids.rng.astype(np.float64).copy()
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_joint{k}") for k in range(1, 7)]
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and int(model.actuator_trnid[a, 0]) in joints:
            i = joints.index(int(model.actuator_trnid[a, 0]))
            if model.actuator_ctrllimited[a]:
                lim[i, 0] = max(lim[i, 0], model.actuator_ctrlrange[a, 0])
                lim[i, 1] = min(lim[i, 1], model.actuator_ctrlrange[a, 1])
    return lim


def ik_link6(model: mujoco.MjModel, data: mujoco.MjData, arm: str, T_goal: np.ndarray, *, seeds: int = 6,
             iters: int = 200, rng: np.random.Generator | None = None,
             limits: np.ndarray | None = None) -> tuple[np.ndarray | None, float, float]:
    """Damped-least-squares IK of link_6's full pose. -> (6 joint values, position error m, rotation error rad).
    ``limits`` (6, 2) bounds the solution (default: the joint ranges; ``actuator_limits`` gives what the position
    actuators can command, which can be narrower)."""
    ids = arm_ids(model, arm)
    if limits is not None:
        ids.rng = np.asarray(limits, dtype=np.float64)
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
            mujoco.mju_quat2Vel(er, quat_of(Re), 1.0)
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
