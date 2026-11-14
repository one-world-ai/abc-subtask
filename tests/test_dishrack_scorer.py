"""Dish-rack scorer: a plate counts only when it stands in the rack, not when it leans on the rack's outside.

Run (CPU, no GL):
    MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_dishrack_scorer.py

The stock rule (plate origin inside the rack's bounding-sphere box + 5 cm, touching the rack, no gripper,
|normal_z| <= 0.55) accepted plates leaning on the outside of the rack with their lower edge on the table: exp12
worlds 12 and 54 (2026-11-13). The fix requires the plate's centre and lowest point to lie over the rack's tight
collision footprint. Poses below are plate freejoint qpos (xyz + wxyz) taken from the bit-exact replays of those
worlds (subtask_rewards/dishrack/probes/dr_replay_final.py, renders checked).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco
import numpy as np

import abc_sim

TASK = "load_plates_into_dish_rack"

# exp12 world 12 (seed 1404717598, dish_rack_3): plate leaning on the rack's long side, edge on the table.
LEAN_OUTSIDE_W12 = (1404717598, [0.703374, 0.201272, 0.822247, 0.215313, -0.00177, -0.496073, 0.841159])
# exp12 world 54 (seed 1928549573, dish_rack_4): same, other rack model.
LEAN_OUTSIDE_W54 = (1928549573, [0.623772, -0.240788, 0.813121, 0.288015, -0.176626, 0.509855, 0.791138])
# exp12 world 49 (seed 720949075, dish_rack_6): plate upright in a slot, released (gripper opened, arm retreated).
IN_SLOT_W49 = (720949075, [0.673767, 0.214238, 0.888004, -0.307401, -0.557077, 0.225006, 0.737931])


def _env(seed: int, options=None):
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    env.forget_arm_state()
    env.reset(seed=seed, randomize=True, options=options)
    return env


def _plate_adr(env) -> int:
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint")
    return int(env.model.jnt_qposadr[jid])


def _set_plate(env, qpos, settle_steps: int = 0):
    a = _plate_adr(env)
    env.data.qpos[a : a + 7] = qpos
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(settle_steps):
        mujoco.mj_step(env.model, env.data)
    return env.evaluate_task()


def test_resets_score_zero():
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    for seed in range(10):
        env.forget_arm_state()
        env.reset(seed=seed, randomize=True)
        r = env.evaluate_task()
        assert float(np.asarray(r.reward).ravel()[0]) == 0.0, seed
        assert not bool(np.asarray(r.success).ravel()[0]), seed


def test_lean_on_outside_is_rejected():
    for seed, qpos in (LEAN_OUTSIDE_W12, LEAN_OUTSIDE_W54):
        env = _env(seed)
        r = _set_plate(env, qpos)
        m = r.metrics
        # what the stock rule looked at: inside its loose box and upright ...
        assert m["plate_spatially_in_rack_mask"][0][0], seed
        assert m["plate_normal_abs_z"][0][0] <= 0.55, seed
        # ... but the lower edge is on the table outside the rack
        assert not m["plate_over_rack_footprint_mask"][0][0], seed
        lo, hi = np.asarray(m["rack_footprint_xy"][0][0]), np.asarray(m["rack_footprint_xy"][0][1])
        bottom = np.asarray(m["plate_bottom_rack_xy"][0][0])
        assert np.any(bottom < lo - 0.03) or np.any(bottom > hi + 0.03), (seed, bottom, lo, hi)
        assert not bool(r.success[0]), seed


def test_released_plate_in_slot_counts():
    seed, qpos = IN_SLOT_W49
    env = _env(seed)
    assert bool(_set_plate(env, qpos).success[0])
    env = _env(seed)
    assert bool(_set_plate(env, qpos, settle_steps=1000).success[0])  # and stays


def _insert_upright(env, axis: int, t: float):
    """Stand plate 0 upright (flat normal along rack axis ``axis``) at ``t`` on the footprint centre line, lower it
    until just above a > 1 mm penetration of the rack, release and settle 1.5 s (as probes/dr_slots.py)."""
    from abc_sim.task_eval.dishrack import _geom_points_world

    m, d = env.model, env.data
    ev = env._task_evaluator
    ev.evaluate_model_data(m, d)
    rack = ev._rack_body_id
    rp, Rr = d.xpos[rack].copy(), d.xmat[rack].reshape(3, 3).copy()
    lo, hi = ev._rack_footprint(d)
    top = max(float(((_geom_points_world(m, d, g) - rp) @ Rr)[:, 2].max()) for g in ev._rack_collision_geom_ids)
    pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "plate")
    pgs = ev._plate_collision_geom_ids[ev.plate_indices[0]]
    pts = np.concatenate([_geom_points_world(m, d, g) for g in pgs])
    loc = (pts - d.xpos[pb]) @ d.xmat[pb].reshape(3, 3)
    pc = (loc.min(0) + loc.max(0)) / 2
    radius = float(np.linalg.norm(loc[:, :2] - pc[:2], axis=1).max())
    ez, ex = Rr[:, axis], Rr[:, 2]
    Rp = np.stack([ex, np.cross(ez, ex), ez], 1)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, Rp.reshape(9))
    a, pset, rset = _plate_adr(env), set(pgs), set(ev._rack_collision_geom_ids)
    c = np.zeros(3)
    c[axis], c[1 - axis] = t, (lo[1 - axis] + hi[1 - axis]) / 2
    best = None
    for zlow in np.arange(top + 0.01, 0.002, -0.005):
        c[2] = zlow + radius
        d.qpos[a : a + 3], d.qpos[a + 3 : a + 7] = rp + Rr @ c - Rp @ pc, q
        mujoco.mj_forward(m, d)
        if any(((int(x.geom1) in pset and int(x.geom2) in rset) or (int(x.geom2) in pset and int(x.geom1) in rset))
               and x.dist < -0.001 for x in d.contact[: d.ncon]):
            break
        best = zlow
    assert best is not None and best < top - 0.03
    c[2] = best + radius
    d.qpos[a : a + 3] = rp + Rr @ c - Rp @ pc
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(750):
        mujoco.mj_step(m, d)
    return env.evaluate_task()


def test_plate_lowered_into_slot_counts_on_two_racks():
    # slot centres from probes/dr_slots.py (seed 0, one plate, rack x axis, centre line)
    for rack, t in (("dish_rack_4", -0.0408), ("dish_rack_6", -0.0723)):
        env = _env(0, {"randomization": {"plate_count": 1, "dish_rack_variant": rack, "randomize_variants": True}})
        r = _insert_upright(env, 0, t)
        assert bool(r.success[0]), (rack, r.metrics["plate_normal_abs_z"][0], r.metrics["plate_bottom_rack_xy"][0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
