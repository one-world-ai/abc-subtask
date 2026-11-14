"""LEGO scorer regression cases found by the 2026-11-13 audit."""
from __future__ import annotations

import os
import json
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "disable")

import abc_sim
import mujoco
import numpy as np

COLORS = ("red", "yellow", "blue")


def rotation(quat):
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, quat)
    return out.reshape(3, 3)


def env_for(seed=0):
    env = abc_sim.make_env(task="sort_lego_blocks", render_cameras=False)
    env.forget_arm_state()
    env.reset(seed=seed)
    return env


def put_block(env, color, *, dest=None, tilt=0.0, outside=False):
    ev, d = env._task_evaluator, env.data
    dest = dest or color
    ba = ev._bin_qpos_addrs[dest]
    a = int(ev._block_qpos_addrs[color][0])
    center, size = ev._goal_regions[dest]
    local = center.copy()
    local[2] = center[2] - size[2] + 0.012
    if outside:
        local[0] = center[0] + size[0] + 0.025
    if tilt:
        d.qpos[ba+3:ba+7] = [np.cos(tilt/2), np.sin(tilt/2), 0, 0]
        d.qpos[ba+2] = 0.85
    d.qpos[a:a+3] = d.qpos[ba:ba+3] + rotation(d.qpos[ba+3:ba+7]) @ local
    d.qpos[a+3:a+7] = [1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)


def test_resets_score_zero():
    env = env_for()
    try:
        for seed in range(20):
            env.forget_arm_state()
            env.reset(seed=seed)
            r = env.evaluate_task()
            assert float(r.reward[0]) == 0 and not bool(r.success[0]), seed
    finally:
        env.close()


def test_matching_bins_accept_real_settled_blocks():
    for seed, color in enumerate(COLORS):
        env = env_for(seed)
        try:
            put_block(env, color)
            for _ in range(int(1.5/env.model.opt.timestep)):
                mujoco.mj_step(env.model, env.data)
            assert int(env.evaluate_task().metrics[f"num_{color}_sorted"][0]) == 1, (seed,color)
        finally:
            env.close()


def test_wrong_color_and_outside_rejected():
    for color in COLORS:
        env = env_for()
        try:
            put_block(env, color, dest=COLORS[(COLORS.index(color)+1)%3])
            r = env.evaluate_task()
            assert r.metrics["num_correct"][0] == 0 and r.metrics["num_wrong_bin"][0] == 1
            env.forget_arm_state(); env.reset(seed=0)
            put_block(env, color, outside=True)
            assert env.evaluate_task().metrics["num_correct"][0] == 0
        finally:
            env.close()


def test_sideways_and_inverted_bins_rejected():
    for theta in (np.pi/2, np.pi, -np.pi/2):
        for color in COLORS:
            env = env_for()
            try:
                put_block(env,color,tilt=theta)
                assert env.evaluate_task().metrics["num_correct"][0] == 0, (color,theta)
            finally:
                env.close()


def test_batch_yaw_and_quaternion_sign():
    env = env_for()
    try:
        for color in COLORS:
            put_block(env,color)
        q = np.stack([env.data.qpos.copy()]*3)
        ev = env._task_evaluator
        # q and -q describe exactly the same orientation.
        for ba in ev._bin_qpos_addrs.values(): q[1,ba+3:ba+7] *= -1
        ba = ev._bin_qpos_addrs['red']
        center = ev._goal_regions['red'][0]
        a = int(ev._block_qpos_addrs['red'][0])
        q[2,ba+3:ba+7] = [0,1,0,0]
        q[2,a:a+3] = q[2,ba:ba+3] + rotation(q[2,ba+3:ba+7]) @ center
        r = ev.evaluate_qpos_batch(q)
        assert r.metrics['num_correct'].tolist() == [3,3,2]
    finally:
        env.close()


def test_confirmed_near_wall_resting_blocks_count():
    saved = json.loads((Path(__file__).parents[1]/'subtask_rewards/lego/probes/settle.json').read_text())['rows']
    for seed,trial in ((0,59),(1,35)):
        row = next(r for r in saved if r['seed']==seed and r['trial']==trial)
        env = env_for(seed)
        try:
            ev,d = env._task_evaluator,env.data
            color = row['color']; a=int(ev._block_qpos_addrs[color][0]); ba=ev._bin_qpos_addrs[color]
            d.qpos[a:a+7]=row['object_qpos']; d.qpos[ba:ba+7]=row['bin_qpos']
            d.qvel[:]=0; mujoco.mj_forward(env.model,d)
            assert env.evaluate_task().metrics['num_correct'][0] == 1
            for _ in range(int(1.0/env.model.opt.timestep)):
                mujoco.mj_step(env.model,d)
            assert env.evaluate_task().metrics['num_correct'][0] == 1
        finally:
            env.close()


def test_wall_solids_inside_outer_bounds_do_not_count():
    env = env_for()
    try:
        q0 = env.data.qpos.copy()
        ev,d = env._task_evaluator,env.data
        for color in COLORS:
            d.qpos[:]=q0
            a=int(ev._block_qpos_addrs[color][0]); ba=ev._bin_qpos_addrs[color]
            _,hi=ev._bin_xy_bounds[color]
            local=np.array([hi[0]*0.98,0,ev._goal_regions[color][0][2]])
            d.qpos[a:a+3]=d.qpos[ba:ba+3]+rotation(d.qpos[ba+3:ba+7])@local
            mujoco.mj_forward(env.model,d)
            assert env.evaluate_task().metrics['num_correct'][0] == 0, color
    finally:
        env.close()


if __name__ == "__main__":
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); print('ok',name,flush=True)
