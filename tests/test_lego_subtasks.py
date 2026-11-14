"""LEGO CPU reward contract, support/release adversaries, and exact start restores."""
from __future__ import annotations

import os
os.environ.setdefault('MUJOCO_GL','disable')
import traceback
import mujoco
import numpy as np
import abc_sim
from abc_sim.task_eval.lego_stages import R,P,H,C,D,DONE,STAGES,COLORS,StageConfig,classify
from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval import lego_handoff as LH
from abc_sim.task_eval.lego_starts import (SEED_BASE,build_start,idle_action,reject_numerics,NumericalInvalid)
from abc_sim.task_eval.stagekit import ARMS,arm_ids,axis_angle_quat,qmul,set_object_pose
from abc_sim.task_specs import get_task_spec


def env_for(stage='reach',seed=SEED_BASE,park=False):
    env=abc_sim.make_env(task='sort_lego_blocks__'+stage,render_cameras=False)
    env.forget_arm_state();env.reset(seed=seed,options={'randomization':{'park_inactive':park}})
    env.evaluate_task();return env


def updates(env,n=6):
    for _ in range(n):env.evaluate_task()
    return env._task_evaluator.chunk_reward()


def in_bin_pose(env,k,color=None,height=.04):
    scene=env._task_evaluator.scene;c=scene.colors[k] if color is None else color
    a=scene.bin_adrs[c];Rbin=env.data.xmat[scene.bin_bodies[c]].reshape(3,3)
    return env.data.qpos[a:a+3]+Rbin@[0,0,height]


def move_object(env,k,pos):
    scene=env._task_evaluator.scene;a=scene.qpos_adr[k]
    set_object_pose(env,scene,k,pos,env.data.qpos[a+3:a+7].copy())


def test_lookup_preserves_stock():
    assert get_task_spec('sort_lego_blocks').evaluator_name=='lego_blocks_sorting'
    for stage in STAGES[:-1]:
        spec=get_task_spec('sort_lego_blocks__'+stage)
        assert get_task_spec(spec.prompt).name==spec.name
        assert spec.evaluator_kwargs()=={'stage':stage}


def test_qpos_only_rejected():
    env=env_for()
    try:env._task_evaluator.evaluate_qpos_batch(env.data.qpos[None])
    except RuntimeError as e:assert 'real contacts and velocities' in str(e)
    else:raise AssertionError('qpos scratch must not manufacture rest')
    env.close()


def test_openings_measured_across_randomized_models():
    env=env_for()
    for seed in range(SEED_BASE,SEED_BASE+20):
        env.forget_arm_state();env.reset(seed=seed);ev=env._task_evaluator
        assert (ev.scene.opening_hi>ev.scene.opening_lo).all()
        assert (ev.scene.rim_local>.05).all()
        assert ev.evaluate_model_data(env.model,env.data).metrics['target_name'] in ev.scene.names
    env.close()


def test_parked_inactive_parts_not_targets_or_failures():
    env=env_for(park=True);ev=env._task_evaluator;scene=ev.scene
    assert (~scene.active).any()
    active=scene.active.copy();k=ev.target
    assert active[k]
    ev.begin(env.model,env.data,k)
    assert np.array_equal(active,scene.active)
    assert sum(int(x.sum()) for x in ev.stock._active.values())==int(active.sum())
    j=int(np.flatnonzero(~active)[0]);move_object(env,j,[.8,0,.2])
    assert updates(env)[1] is False
    env.close()


def test_all_bin_motion_and_tips_fail():
    for c in range(3):
        for tip in (False,True):
            env=env_for();a=env._task_evaluator.scene.bin_adrs[c]
            if tip:env.data.qpos[a+3:a+7]=qmul(axis_angle_quat(np.array([1.,0,0]),np.pi/2),env.data.qpos[a+3:a+7].copy())
            else:env.data.qpos[a]+=.06
            mujoco.mj_forward(env.model,env.data)
            assert updates(env)==(-1.,True)
            assert env._task_evaluator._failed=='bin moved or tipped';env.close()


def test_push_and_wrong_bin_fail():
    for stage in ('reach','pick'):
        env=env_for(stage);ev=env._task_evaluator
        ev.begin(env.model,env.data,ev.target,labels=np.full(len(ev.scene.names),STAGES.index(stage)))
        env.data.qpos[ev.scene.qpos_adr[ev.target]]+=.04;mujoco.mj_forward(env.model,env.data)
        assert updates(env)==(-1.,True) and ev._failed=='pushed';env.close()
    env=env_for();ev=env._task_evaluator;k=ev.target
    move_object(env,k,in_bin_pose(env,k,color=(ev.scene.colors[k]+1)%3))
    assert updates(env)==(-1.,True) and ev._failed=='wrong bin';env.close()


def test_finger_contact_far_from_grasp_site_is_not_reach():
    env=env_for();ev=env._task_evaluator;obs={key:np.copy(value) for key,value in ev.tracker.obs.items()};k=ev.target;n=len(ev.scene.names)
    obs['speed'][k]=obs['angular_speed'][k]=0.;obs['resting_contact'][k]=True
    obs['touched'][k]=True;obs['grasped'][k]=False;obs['dist'][k]=.08
    assert classify(obs,np.full(n,R),ev.cfg)[k]==R
    assert classify(obs,np.full(n,P),ev.cfg)[k]==P
    obs['touched'][k]=False;obs['dist'][k]=.045
    assert classify(obs,np.full(n,R),ev.cfg)[k]==P
    obs['grasped'][k]=obs['touched'][k]=True;obs['dist'][k]=.07;obs['xy'][k]=ev._start_xy+[.04,0]
    assert ev.step_failure(k,obs,R)=='pushed';env.close()


def test_target_and_background_off_table_fail():
    for target in (True,False):
        env=env_for();ev=env._task_evaluator;k=ev.target if target else next(i for i in range(len(ev.scene.names)) if i!=ev.target)
        env.data.qpos[ev.scene.qpos_adr[k]+2]-=.2;mujoco.mj_forward(env.model,env.data)
        assert updates(env)==(-1.,True)
        assert ev._failed==('fell off the table' if target else 'another block fell off the table');env.close()


def test_previously_sorted_contents_protected():
    env=env_for();ev=env._task_evaluator;k=ev.target;j=next(i for i in range(len(ev.scene.names)) if i!=k)
    move_object(env,j,in_bin_pose(env,j));ev.begin(env.model,env.data,k)
    assert ev._sorted[j]
    env.data.qpos[ev.scene.qpos_adr[j]]+=.3;mujoco.mj_forward(env.model,env.data)
    assert updates(env)==(-1.,True) and ev._failed=='knocked a sorted block out';env.close()


def test_float_and_airborne_contact_cycles_not_support():
    env=env_for();ev=env._task_evaluator;scene=ev.scene;rest=scene.heights(env.data)
    k=ev.target;j=next(i for i in range(len(scene.names)) if i!=k and scene.colors[i]==scene.colors[k])
    for i,height in ((k,.045),(j,.057)):
        a=scene.qpos_adr[i];set_object_pose(env,scene,i,in_bin_pose(env,i,height=height),np.array([1.,0,0,0]))
    obs=scene.observe(env.data,rest)
    assert obs['in_bin'][[k,j]].all()
    assert obs['resting_contact'][[k,j]].all()
    assert not obs['supported'][[k,j]].any()
    assert not (classify(obs,np.full(len(scene.names),D),ev.cfg)[[k,j]]==DONE).any()
    env.close()


def test_every_release_after_regrasp_is_checked():
    env=env_for('drop');ev=env._task_evaluator;obs=ev.tracker.obs;k=ev.target
    ev._released=True;ev._was_touched=True;obs['touched'][k]=False;obs['over'][k]=False
    assert ev.step_failure(k,obs,D)=='released outside the opening';env.close()


def test_safe_lowering_allowed_and_one_finger_carry_not_goal():
    env=env_for('drop');ev=env._task_evaluator;obs=ev.tracker.obs;k=ev.target;n=len(ev.scene.names)
    obs['in_bin'][k]=True;obs['over'][k]=False;obs['above_rim'][k]=-.03
    obs['touched'][k]=True;obs['grasped'][k]=True
    assert classify(obs,np.full(n,D),ev.cfg)[k]==D
    ev._was_touched=True;obs['touched'][k]=False
    assert ev.step_failure(k,obs,D)==''
    obs['in_bin'][k]=False;obs['over'][k]=True;obs['touched'][k]=True;obs['grasped'][k]=False
    assert classify(obs,np.full(n,C),ev.cfg)[k]!=D;env.close()


def cross_env(stage='handoff'):
    """Seed SEED_BASE (cross-side: its target is on the left, its bin on the right)."""
    env=env_for(stage);ev=env._task_evaluator;k=ev.target
    assert ev.tracker.obs['cross'][k] and ev.tracker.obs['giver'][k]==0 and ev.tracker.obs['receiver'][k]==1
    return env,ev,k


def synthetic(ev,k,**kw):
    obs={key:np.copy(value) for key,value in ev.tracker.obs.items()}
    obs['speed'][k]=obs['angular_speed'][k]=0.;obs['resting_contact'][k]=False
    for key,value in kw.items():obs[key][k]=value
    return obs


def test_handoff_labels_cross_and_same_side():
    env,ev,k=cross_env();n=len(ev.scene.names);cfg=ev.cfg
    mid=np.array([.5,0.,.9]);lift=.9-ev.tracker.rest_height[k,2]
    held=dict(grasped=True,touched=True,lift=.08,f_give=2,f_recv=0,touch_give=True,touch_recv=False,
              gap_give=0.,env_touch=False,fingers=2,dist=.01)
    # giver's lifted hold anywhere = handoff (pick's goal for a cross-side block)
    assert classify(synthetic(ev,k,**held),np.full(n,P),cfg)[k]==H
    # receiver holds it alone, giver clear, in the air = carry (the shared handoff goal)
    got=dict(held,com=mid,m_gap=0.,lift=lift,f_give=0,f_recv=2,touch_give=False,touch_recv=True,gap_give=.08)
    assert classify(synthetic(ev,k,**got),np.full(n,H),cfg)[k]==C
    # both hold / giver within 6 cm = still handoff (in the region)
    assert classify(synthetic(ev,k,**dict(got,f_give=2,touch_give=True,gap_give=0.)),np.full(n,H),cfg)[k]==H
    assert classify(synthetic(ev,k,**dict(got,gap_give=.05)),np.full(n,H),cfg)[k]==H
    # received but touching the table is not the goal
    assert classify(synthetic(ev,k,**dict(got,env_touch=True)),np.full(n,H),cfg)[k]==H
    # the giver carrying it over the opening is not drop (it is the receiver's block)
    over=dict(held,over=True,opening_gap=0.,above_rim=.03)
    assert classify(synthetic(ev,k,**over),np.full(n,H),cfg)[k]==H
    assert classify(synthetic(ev,k,**dict(over,f_give=0,touch_give=False,f_recv=2,touch_recv=True,gap_give=.1)),
                    np.full(n,C),cfg)[k]==D
    # same-side: a lifted grasp by either arm is carry, as before
    assert classify(synthetic(ev,k,**dict(held,cross=False)),np.full(n,P),cfg)[k]==C
    # progress: shared handoff progress minus the region gap
    o=synthetic(ev,k,**got)
    assert abs(ev.progress(H,o)[k]-hs.progress(o,cfg.handoff)[k])<1e-12 and abs(ev.progress(H,o)[k]-1.25)<1e-9
    env.close()


def test_handoff_failures():
    env,ev,k=cross_env();obs=ev.tracker.obs
    assert ev.stage==H
    free=synthetic(ev,k,touched=False,touch_give=False,touch_recv=False,f_give=0,lift=.1)
    assert ev.step_failure(k,free,H)=='dropped during the handoff'
    assert ev.step_failure(k,synthetic(ev,k,env_touch=True),H)=='touched the table during the handoff'
    across=synthetic(ev,k,touch_give=True,f_give=2,env_touch=False,y_recv=.25)
    assert ev.step_failure(k,across,H)=='giver carried it across'
    assert ev.step_failure(k,synthetic(ev,k,touch_give=True,f_give=2,env_touch=False,y_recv=.15),H)==''
    # the physical drop from the saved-style start: the giver lets go in the air
    start,_=build_start(env,SEED_BASE,'handoff');assert start is not None
    env.reset_to_stage(start['seed'],start);ev=env._task_evaluator
    act=idle_action(env);act[6]=1.
    for _ in range(4):
        for _ in range(32):env.step(act,render_obs=False)
        reward,done=ev.chunk_reward()
        if done:break
    assert (reward,done)==(-1.,True) and ev._failed=='dropped during the handoff',ev._failed
    env.close()


def test_handoff_solver_reaches_carry():
    env=env_for('handoff');start,_=build_start(env,SEED_BASE,'handoff');assert start is not None
    env.reset_to_stage(start['seed'],start);ev=env._task_evaluator
    ok,why=LH.handoff_solver(env,ev.scene,ev.target,int(start['arm']))
    assert ok,why
    assert ev._success and not ev._failed and ev._last['label']=='carry',ev._last
    env.close()


def test_big_move_equals_eight_small_moves():
    env=env_for();ev=env._task_evaluator;initial=ev._phi_chunk;totals=[]
    for count in (1,8):
        ev._phi_chunk=initial;total=0
        for value in np.linspace(initial,initial+.06,count+1)[1:]:
            ev._last['progress']=value;reward,done=ev.chunk_reward();assert not done;total+=reward
        totals.append(total)
    assert abs(totals[0]-totals[1])<1e-6;env.close()


def test_numerical_warning_is_rejected():
    env=env_for();step=env.step
    def bad(*args,**kwargs):
        result=step(*args,**kwargs)
        env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number+=1
        return result
    env.step=bad
    try:
        with reject_numerics(env):env.step(idle_action(env),render_obs=False)
    except NumericalInvalid:pass
    else:raise AssertionError('numerical warning was ignored')
    env.close()


def test_built_starts_idle_and_restore_exactly():
    for stage in ('reach','pick','handoff','carry','drop'):
        env=env_for(stage);checked=0
        # carry/drop: seed 0 is cross-side (receiver after a handoff), 4 and 7 same-side (7 was rejected by the
        # round-2 drop builder); handoff exists only for cross-side seeds.
        for i in ((0,4,7,5,8) if stage in ('carry','drop') else range(30)):
            start,_=build_start(env,SEED_BASE+i,stage)
            if start is None:continue
            env.reset_to_stage(start['seed'],start);assert env.evaluate_task().metrics['label']==stage
            action=idle_action(env);total=0
            with reject_numerics(env):
                for _ in range(10):
                    for _ in range(32):env.step(action,render_obs=False)
                    reward,done=env._task_evaluator.chunk_reward();assert not done,(stage,i,env._task_evaluator._last)
                    total+=reward
            assert abs(total)<.05,(stage,i,total)
            runs=[]
            for _ in range(2):
                env.reset_to_stage(start['seed'],start)
                for _ in range(32):env.step(action,render_obs=False)
                runs.append(env.data.qpos.copy())
            assert np.array_equal(*runs);checked+=1
            if checked==3:break
        env.close();assert checked==3,(stage,checked)


def main():
    failed=0;tests=[fn for name,fn in globals().items() if name.startswith('test_') and callable(fn)]
    for fn in tests:
        try:fn();print('ok',fn.__name__,flush=True)
        except Exception:failed+=1;traceback.print_exc();print('FAIL',fn.__name__,flush=True)
    print(f'{len(tests)-failed}/{len(tests)} passed',flush=True);raise SystemExit(bool(failed))


if __name__=='__main__':main()
