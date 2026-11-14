"""Partition semantics and live bin-frame geometry for the stock nuts/bolts scorer.

Run: MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python tests/test_nuts_bolts_scorer.py
Release/rest/bin-stability are deliberately separate subtask design decisions; this scorer receives qpos only.
"""
from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL','disable')
import abc_sim,mujoco,numpy as np
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch


def make_env():
    e=abc_sim.make_env(task='sort_nuts_and_bolts',render_cameras=False)
    e.reset(seed=0)
    return e


def group(e,addrs,binadr,center):
    q=e.data.qpos[binadr:binadr+7];r=_quat_to_rotmat_batch(q[3:][None])[0]
    for i,a in enumerate(addrs):
        local=center+np.array([((i%5)-2)*.018,((i//5)-1.5)*.018,0])
        e.data.qpos[a:a+3]=q[:3]+r@local
    mujoco.mj_forward(e.model,e.data)


def arrange(e,inverted=False):
    v=e._task_evaluator
    bins=[(v._nuts_bin_qpos_addr,v._nuts_goal_local_center),(v._bolts_bin_qpos_addr,v._bolts_goal_local_center)]
    if inverted:bins.reverse()
    group(e,v._nut_qpos_addrs,*bins[0]);group(e,v._bolt_qpos_addrs,*bins[1])


def test_either_consistent_partition_succeeds():
    e=make_env()
    for inverted in (False,True):
        arrange(e,inverted);r=e.evaluate_task()
        assert bool(r.success[0]) and r.reward[0]==1
        assert r.metrics['sorted_orientation']==['inverted' if inverted else 'spec']
    e.close()


def test_mixed_bins_and_one_bin_are_not_complete():
    e=make_env();v=e._task_evaluator;arrange(e)
    group(e,v._nut_qpos_addrs[:1],v._bolts_bin_qpos_addr,v._bolts_goal_local_center)
    assert not bool(e.evaluate_task().success[0])
    group(e,v._nut_qpos_addrs,v._nuts_bin_qpos_addr,v._nuts_goal_local_center)
    group(e,v._bolt_qpos_addrs,v._nuts_bin_qpos_addr,v._nuts_goal_local_center)
    assert not bool(e.evaluate_task().success[0])
    e.close()


def test_bin_pose_and_size_define_live_regions():
    e=make_env();v=e._task_evaluator
    for a in (v._nuts_bin_qpos_addr,v._bolts_bin_qpos_addr):
        e.data.qpos[a:a+3]+=np.array([-.04,.03,.02])
        e.data.qpos[a+3:a+7]=[np.cos(.6),0,0,np.sin(.6)]
    arrange(e)
    assert bool(e.evaluate_task().success[0])
    a=v._nut_qpos_addrs[0];b=v._nuts_bin_qpos_addr
    q=e.data.qpos[b:b+7];R=_quat_to_rotmat_batch(q[3:][None])[0]
    e.data.qpos[a:a+3]=q[:3]+R@(v._nuts_goal_local_center+np.array([v._nuts_goal_size[0]+.01,0,0]))
    mujoco.mj_forward(e.model,e.data)
    assert not bool(e.evaluate_task().success[0])
    e.close()


def test_active_mask_excludes_parked_parts():
    e=make_env();v=e._task_evaluator
    v.set_active_object_joints([v.nut_names[0]+'_joint',v.bolt_names[0]+'_joint'])
    group(e,v._nut_qpos_addrs[:1],v._nuts_bin_qpos_addr,v._nuts_goal_local_center)
    group(e,v._bolt_qpos_addrs[:1],v._bolts_bin_qpos_addr,v._bolts_goal_local_center)
    r=e.evaluate_task();assert bool(r.success[0]) and int(r.metrics['num_correct'][0])==2
    e.close()

if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_'):
            fn();print('ok',name,flush=True)
