"""Clutter box scorer regressions: physical vertical geometry, not the oversized goal site.

MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_grab_clutter_scorer.py
"""
import os
os.environ.setdefault('MUJOCO_GL', 'disable')
import mujoco
import numpy as np
import abc_sim

TASK = 'grab_specific_object_from_clutter'

def make(seed=0):
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    env.forget_arm_state()
    env.reset(seed=seed)
    return env

def put_target(env, local):
    ev = env._task_evaluator
    box = env.model.body('grab_clutter_target_box').id
    env.data.qpos[ev._target_qpos_addr:ev._target_qpos_addr+3] = (
        env.data.xpos[box] + env.data.xmat[box].reshape(3,3) @ np.array(local))
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model,env.data)
    return env.evaluate_task()

def success(result):
    return bool(np.ravel(result.success)[0])

def test_resets_score_zero():
    env = make()
    try:
        for seed in range(20):
            env.forget_arm_state(); env.reset(seed=seed)
            result = env.evaluate_task()
            assert not success(result),seed
            assert float(np.ravel(result.reward)[0]) == 0.,seed
    finally: env.close()

def test_above_rim_and_below_floor_are_not_inside():
    for seed in (0,1,2):
        env = make(seed)
        try:
            for z in (.20,.28,-.04):
                result=put_target(env,[0,0,z])
                assert bool(np.ravel(result.metrics['target_in_goal_region'])[0]),(seed,z)
                assert not success(result),(seed,z,result.metrics)
        finally: env.close()

def test_real_placements_survive_settling():
    for seed in (0,1):  # distinct yogurt and tomato collision models
        env=make(seed)
        try:
            put_target(env,[0,0,.08])
            env.data.ctrl[env._ctrl_indices]=env.data.qpos[env._qpos_indices]
            for _ in range(1500): mujoco.mj_step(env.model,env.data)
            result=env.evaluate_task()
            assert success(result),(seed,result.metrics)
            assert float(np.linalg.norm(env.data.qvel[env.model.jnt_dofadr[env.model.joint('target_object_joint').id]:][:3])) < .02
            for _ in range(500): mujoco.mj_step(env.model,env.data)
            assert success(env.evaluate_task()),seed
        finally: env.close()

def test_wrong_object_does_not_satisfy_target():
    env=make()
    try:
        ev=env._task_evaluator
        a=int(env.model.jnt_qposadr[env.model.joint('clutter_obj_01_joint').id])
        box=env.model.body('grab_clutter_target_box').id
        env.data.qpos[a:a+3]=env.data.xpos[box]+env.data.xmat[box].reshape(3,3)@np.array([0,0,.05])
        mujoco.mj_forward(env.model,env.data)
        assert not success(env.evaluate_task())
    finally:env.close()

def test_batched_geometry_uses_each_target_pose():
    env=make()
    try:
        q=[]
        for z in (.20,-.04,.08):
            put_target(env,[0,0,z]);q.append(env.data.qpos.copy())
        r=env._task_evaluator.evaluate_qpos_batch(np.array(q))
        assert np.array_equal(r.success,np.array([False,False,True])),r.metrics
        assert np.all(np.isfinite(env._task_evaluator._target_collision_points))
    finally:env.close()

if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn();print('ok',name,flush=True)
