"""Shared helpers for the inhand_transfer phase-0 probes (not kit code)."""
import mujoco
import numpy as np
import abc_sim
from abc_sim.task_eval.stagekit import ObjectScene

TASK = "inhand_transfer_item_to_other_side"


def make():
    print("abc_sim from", abc_sim.__file__, flush=True)
    return abc_sim.make_env(task=TASK, render_cameras=False)


def reset(env, seed):
    env.forget_arm_state()
    env.reset(seed=seed, randomize=True)
    md = env._last_randomization.metadata
    j = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_joint")
    scene = ObjectScene(env.model, [("task_object", j)])
    return scene, md


def score(env):
    r = env.evaluate_task()
    return float(np.asarray(r.reward).ravel()[0]), bool(np.asarray(r.success).ravel()[0]), r.metrics


def idle(env, n):
    a = env.get_state().astype(np.float32)
    for _ in range(n):
        env.step(a, render_obs=False)


def obj_pos(env, scene):
    a = int(scene.qpos_adr[0])
    return env.data.qpos[a:a + 3].copy()


def obj_extent(env, scene):
    """World AABB of the object's collision geoms (vertex-exact for meshes)."""
    m, d = env.model, env.data
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if scene.owner[g][0] != 0 or m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n] @ R.T + p
        else:
            v = p[None] + np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_rbound[g]
        lo = np.minimum(lo, v.min(0)); hi = np.maximum(hi, v.max(0))
    return lo, hi
