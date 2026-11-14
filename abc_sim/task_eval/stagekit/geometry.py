"""Small rigid-transform helpers (4x4 homogeneous transforms, wxyz quaternions)."""

from __future__ import annotations

import mujoco
import numpy as np


def T(pos, quat) -> np.ndarray:
    out = np.eye(4)
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat, np.float64))
    out[:3, :3], out[:3, 3] = m.reshape(3, 3), pos
    return out


def inv(Tm: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Tm[:3, :3].T
    out[:3, 3] = -Tm[:3, :3].T @ Tm[:3, 3]
    return out


def quat_of(Rm: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(Rm).reshape(9))
    return q


def axis_angle_quat(axis, angle) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, np.float64), float(angle))
    return q


def qmul(a, b) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mulQuat(q, np.asarray(a, np.float64), np.asarray(b, np.float64))
    return q
