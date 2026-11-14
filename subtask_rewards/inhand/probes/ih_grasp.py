"""Grasp synthesis for the inhand probes: object principal axis from its collision mesh, top-down and side grasps
at a point along the axis, two-arm actions. (Probe code; a stagekit proposal if it survives phase 3.)

Gripper facts (yam_inhand_transfer_base.xml): link_6 z points to the fingers; grasp site 0.1347 m along link_6 z;
the fingers close along link_6 x (measured: finger pads at link_6 x = +-0.067 open, +-0.02 closed)."""
import mujoco
import numpy as np
from abc_sim.task_eval.stagekit import ARMS, GRIP_MAX, arm_ids, inv, ik_link6

SITE_Z = 0.1347
R_DOWN = np.diag([1.0, -1.0, -1.0])  # link_6 z -> world -z


def world_verts(env, scene, k=0, n_samples=6000):
    """Dense surface points of the object's collision meshes (vertices + area-weighted samples on the faces; long
    primitives such as cylinders have vertices only at their ends)."""
    m, d = env.model, env.data
    vs, tris = [], []
    for g in range(m.ngeom):
        if scene.owner[g][0] != 0 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] == 0:
            continue
        mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
        w = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        vs.append(w)
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        tris.append(w[m.mesh_face[fa:fa + fn]])
    v = np.concatenate(vs)
    t = np.concatenate(tris)
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    i = rng.choice(len(t), size=n_samples, p=area / area.sum())
    r1, r2 = rng.random(n_samples), rng.random(n_samples)
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * t[i, 0] + (s * (1 - r2))[:, None] * t[i, 1] + (s * r2)[:, None] * t[i, 2]
    return np.concatenate([v, pts])


def principal(env, scene):
    """(centre, unit long axis (3D), half-lengths along it (t_min, t_max), vertices)."""
    v = world_verts(env, scene)
    c = 0.5 * (v.min(0) + v.max(0))
    w, U = np.linalg.eigh(np.cov((v - c).T))
    u = U[:, -1]
    t = (v - c) @ u
    return c, u, (float(t.min()), float(t.max())), v


def section(v, c, u, t, band=0.012):
    """Centre and vertical extent of the object's cross-section at axis coordinate t."""
    tt = (v - c) @ u
    s = v[np.abs(tt - t) < band]
    if len(s) < 3:
        s = v[np.argsort(np.abs(tt - t))[:20]]
    mid = 0.5 * (s.min(0) + s.max(0))
    return mid, s.min(0), s.max(0)


def Rz(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])


def topdown_T6(point, axis_yaw, flip, below=0.01):
    """link_6 straight down, fingers closing across the horizontal axis direction, grasp site ``below`` under point."""
    T6 = np.eye(4)
    T6[:3, :3] = Rz(axis_yaw + np.pi / 2 + (np.pi if flip else 0.0)) @ R_DOWN  # closing axis (link_6 x) _|_ axis
    T6[:3, 3] = np.asarray(point) + np.array([0.0, 0.0, SITE_Z - below])
    return T6


def tilt_T6(T6, beta, axis_col=0):
    """Rotate a grasp pose by ``beta`` about link_6's own axis column ``axis_col`` (0 = the closing axis), keeping
    the grasp site fixed."""
    from abc_sim.task_eval.stagekit import axis_angle_quat, T as _T
    site = T6[:3, 3] + T6[:3, 2] * SITE_Z
    Rr = _T(np.zeros(3), axis_angle_quat(T6[:3, axis_col], beta))[:3, :3]
    out = np.eye(4); out[:3, :3] = Rr @ T6[:3, :3]
    out[:3, 3] = site - out[:3, 2] * SITE_Z
    return out


TILTS = np.deg2rad([0, 25, -25, 45, -45, 65, -65])


def reachable(env, arm, T6, tilts=TILTS, axis_cols=(0, 1), tol=(2.5e-3, 0.04)):
    """First tilt of T6 (about its closing axis, then about link_6 y) the arm's IK reaches -> (T6, tilt) or
    (None, best error). Tolerance a bit under hold_object's (3 mm, 0.05 rad) so its own IK run also succeeds."""
    best = np.inf
    for col, b in [(c, b) for c in axis_cols for b in tilts]:
        if col == 1 and b == 0:
            continue
        Tt = tilt_T6(T6, b, col)
        q, pe, re = ik_link6(env.model, env.data, ARMS[arm], Tt)
        if q is not None and pe < tol[0] and re < tol[1]:
            return Tt, float(b)
        best = min(best, pe + 0.05 * re)
    return None, best


def side_T6(point, approach, roll):
    """link_6 z horizontal along ``approach`` (unit, world), rotated ``roll`` about it; grasp site at point."""
    z = np.asarray(approach, float); z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R0 = np.stack([x, y, z], axis=1)
    cr, sr = np.cos(roll), np.sin(roll)
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]])
    T6 = np.eye(4); T6[:3, :3] = R0 @ Rr
    T6[:3, 3] = np.asarray(point) - z * SITE_Z
    return T6


def two_arm_action(env, q, grip):
    """q: dict arm -> 6 joints (None keeps that arm's current target); grip: dict arm -> 0..1."""
    act = np.zeros(14, np.float32)
    for i, ci in enumerate(env._ctrl_indices):
        act[i] = env.data.ctrl[ci] / GRIP_MAX if i in env._gripper_set else env.data.ctrl[ci]
    for a in (0, 1):
        if q.get(a) is not None:
            act[a * 7:a * 7 + 6] = q[a]
        if a in grip:
            act[a * 7 + 6] = grip[a]
    return act


def arm_contacts(env, arm):
    """Names of bodies touching arm ``arm``'s link_6 subtree (gripper) other than itself."""
    m, d = env.model, env.data
    ids = arm_ids(m, ARMS[arm])
    def in_g(b):
        while b > 0:
            if b == ids.link6:
                return True
            b = int(m.body_parentid[b])
        return False
    out = set()
    for c in d.contact[:d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if in_g(b1) and not in_g(b2):
            out.add(m.body(b2).name or "world")
        elif in_g(b2) and not in_g(b1):
            out.add(m.body(b1).name or "world")
    return out


def teleport_arm(env, arm, T6, opening=GRIP_MAX, rng=None, tol=(3e-3, 0.05)):
    m, d = env.model, env.data
    q, pe, re = ik_link6(m, d, ARMS[arm], T6, rng=rng)
    if q is None or pe > tol[0] or re > tol[1]:
        return None, f"no IK ({pe:.3f} m, {re:.2f} rad)"
    ids = arm_ids(m, ARMS[arm])
    d.qpos[ids.qadr], d.qvel[ids.dadr] = q, 0.0
    d.qpos[ids.finger_qadr[0]], d.qpos[ids.finger_qadr[1]] = opening, -opening
    d.qvel[list(ids.finger_dadr)] = 0.0
    mujoco.mj_forward(m, d)
    return q, ""
