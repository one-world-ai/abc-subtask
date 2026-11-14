"""Real-contact nuts/bolts reward contract and adversarial terminal checks (CPU)."""
from __future__ import annotations
import os,traceback
os.environ.setdefault('MUJOCO_GL','disable')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import abc_sim,mujoco,numpy as np
from abc_sim.task_specs import get_task_spec
from abc_sim.task_eval.nuts_bolts_stages import R,P,H,C,D,DONE,FALLEN,STAGES,classify
from abc_sim.task_eval.nuts_bolts_starts import SEED_BASE,build_start,idle_action
from abc_sim.task_eval.stagekit import ARMS,arm_ids,axis_angle_quat,qmul,set_object_pose

def env_for(stage='reach',seed=SEED_BASE):
    e=abc_sim.make_env(task='sort_nuts_and_bolts__'+stage,render_cameras=False);e.reset(seed=seed);e.evaluate_task();return e

def updates(e,n=6):
    for _ in range(n):e.evaluate_task()
    return e._task_evaluator.chunk_reward()

def bin_position(e,k,local,wrong=False):
    s=e._task_evaluator.scene;b=int(s.assigned[k]);b=1-b if wrong else b
    return e.data.qpos[s.bin_adrs[b]:s.bin_adrs[b]+3]+e.data.xmat[s.bin_bodies[b]].reshape(3,3)@local

def set_part(e,k,pos,quat=None):
    s=e._task_evaluator.scene;a=s.qpos_adr[k]
    set_object_pose(e,s,k,pos,e.data.qpos[a+3:a+7].copy() if quat is None else quat)

def test_lookup_and_cpu_contract():
    assert get_task_spec('sort_nuts_and_bolts').evaluator_name=='nuts_bolts_sorting'
    for st in STAGES[:-1]:
        spec=get_task_spec('sort_nuts_and_bolts__'+st)
        assert get_task_spec(spec.prompt).name==spec.name and spec.evaluator_kwargs()=={'stage':st}
    e=env_for()
    try:e._task_evaluator.evaluate_qpos_batch(e.data.qpos[None])
    except RuntimeError as exc:assert 'real contacts and velocities' in str(exc)
    else:raise AssertionError('qpos-only evaluator manufactured a settled state')
    e.close()

def test_fixed_mapping_wrong_bin_and_frozen_target():
    e=env_for('carry');ev=e._task_evaluator;s=ev.scene;k=ev.target
    other=next(i for i in range(len(s.names)) if i!=k)
    ev.begin(e.model,e.data,k,labels=np.full(len(s.names),C))
    b=int(s.assigned[k]);local=s.goal_centers[1-b]
    set_part(e,k,bin_position(e,k,local,True))
    assert updates(e)==(-1.,True) and ev._failed=='wrong bin' and ev.target==k
    e.close()

def test_push_bin_motion_tip_and_off_table_fail():
    for case in ('push','bin_move','bin_tip','off_table'):
        e=env_for();ev=e._task_evaluator;s=ev.scene;k=ev.target
        if case=='push':e.data.qpos[s.qpos_adr[k]]+=.04
        elif case=='bin_move':e.data.qpos[s.bin_adrs[0]]+=.04
        elif case=='bin_tip':
            a=s.bin_adrs[1];e.data.qpos[a+3:a+7]=qmul(axis_angle_quat(np.array([1.,0,0]),np.pi/2),e.data.qpos[a+3:a+7].copy())
        else:e.data.qpos[s.qpos_adr[k]+2]-=.2
        mujoco.mj_forward(e.model,e.data)
        assert updates(e)==(-1.,True),(case,ev._last)
        assert ev._failed=={'push':'pushed','bin_move':'bin moved or tipped','bin_tip':'bin moved or tipped','off_table':'part fell off the table'}[case]
        e.close()

def test_floating_contact_cycle_does_not_support():
    e=env_for('drop');ev=e._task_evaluator;s=ev.scene;ks=[i for i in range(len(s.names)) if s.assigned[i]==0][:2]
    rest=s.heights(e.data);b=0
    for k,x in zip(ks,(0.,.014)):
        set_part(e,k,bin_position(e,k,s.goal_centers[b]+[x,0,.005]),np.array([1.,0,0,0]))
    o=s.observe(e.data,rest)
    assert o['in_bin'][ks].all() and o['resting_contact'][ks].all()
    assert not o['supported'][ks].any()
    assert not (classify(o,np.full(len(s.names),D),ev.cfg)[ks]==DONE).any()
    e.close()

def rooted_stack(e):
    ev=e._task_evaluator;s=ev.scene;ks=[i for i in range(len(s.names)) if s.assigned[i]==0][:2]
    hs=[float(e.model.geom_size[s.collision_geoms[k][0],2]) for k in ks]
    z=s.floor[0]+hs[0]-.0002
    for i,k in enumerate(ks):
        set_part(e,k,bin_position(e,k,np.array([0.,0.,z if i==0 else z+hs[0]+hs[1]-.0002])),np.array([1.,0,0,0]))
    return ks

def test_fixture_rooted_stacks_and_knocked_contents():
    e=env_for();ev=e._task_evaluator;s=ev.scene;rest=s.heights(e.data);ks=rooted_stack(e)
    o=s.observe(e.data,rest)
    assert o['in_bin'][ks].all() and o['supported'][ks].all(),(o['in_bin'][ks],o['supported'][ks])
    target=next(k for k in range(len(s.names)) if k not in ks)
    ev.begin(e.model,e.data,target,rest_height=rest)
    assert ev._sorted[ks].all()
    e.data.qpos[s.qpos_adr[ks[0]]]+=.3;mujoco.mj_forward(e.model,e.data)
    assert updates(e)==(-1.,True) and ev._failed=='knocked a sorted part out'
    e.close()

def test_real_velocity_and_all_release_checks():
    e=env_for('drop');ev=e._task_evaluator;o=ev.tracker.obs;k=ev.target
    ev._released=True;ev._was_touched=True;o['touched'][k]=False;o['over'][k]=False
    assert ev.step_failure(k,o,D)=='released outside the opening'
    ev._was_touched=True;o['in_bin'][k]=True;o['above_rim'][k]=-.02;o['clearance'][k]=0.
    assert ev.step_failure(k,o,D)==''
    o['in_bin'][k]=False;o['resting_contact'][k]=False;o['speed'][k]=0;o['angular_speed'][k]=0;ev._was_touched=False
    assert ev.step_failure(k,o,D)==''
    assert classify(o,np.full(len(ev.scene.names),D),ev.cfg)[k]==D
    e.close()

def test_lowered_inside_opening_stays_drop():
    # Rim is above the scorer box top: a held part lowered between them is inside the bin, not back in carry.
    e=env_for('drop');ev=e._task_evaluator;s=ev.scene;k=ev.target;b=int(s.assigned[k]);rest=s.heights(e.data)
    top=s.goal_centers[b,2]+s.goal_sizes[b,2];assert s.rim[b]>top
    set_part(e,k,bin_position(e,k,np.array([0.,0.,top+.004])),np.array([1.,0,0,0]))
    o=s.observe(e.data,rest)
    assert o['inside'][k] and not o['over'][k] and not o['in_bin'][k],(o['above_rim'][k],o['xy_gap'][k])
    o['touched'][k]=True;o['grasped'][k]=True;o['lift'][k]=.06;n=len(s.names)
    assert classify(o,np.full(n,D),ev.cfg)[k]==D
    assert classify(o,np.full(n,C),ev.cfg)[k]==C  # entry still needs the part over the opening
    ev._was_touched=True;o['touched'][k]=False
    # Drop amendment: 4 mm above the scorer box top is far above the floor, so releasing there is too high.
    assert o['clearance'][k]>ev.cfg.release_max_clearance_m and ev.step_failure(k,o,D)=='released too high'
    set_part(e,k,bin_position(e,k,np.array([s.inner[b,0]+.03,0.,top+.004])),np.array([1.,0,0,0]))
    o=s.observe(e.data,rest);assert not o['inside'][k]
    ev._was_touched=True;o['touched'][k]=False
    assert ev.step_failure(k,o,D)=='released outside the opening'
    e.close()

def test_drop_release_only_low_in_bin():
    # Drop amendment (2026-11-14): a release with the part's lowest collision extent > 5 mm above the floor is -1.
    e=env_for('drop');ev=e._task_evaluator;s=ev.scene;k=ev.target;b=int(s.assigned[k]);rest=s.heights(e.data)
    assert abs(ev.cfg.release_max_clearance_m-.005)<1e-12
    def place(z):
        set_part(e,k,bin_position(e,k,np.array([0.,0.,z])),np.array([1.,0,0,0]));return s.observe(e.data,rest)
    z0=s.floor[b]+.03;c0=float(place(z0)['clearance'][k])
    def released_with(clearance,stage=D):
        o=place(z0-c0+clearance);assert abs(o['clearance'][k]-clearance)<1e-6 and o['inside'][k]
        ev._released=False;ev._was_touched=True;o['touched'][k]=False;saved=ev.stage;ev.stage=stage
        try:return ev.step_failure(k,o,D)
        finally:ev.stage=saved
    for c in (0.,.002,.0049):assert released_with(c)=='' and ev._released,c
    for c in (.0051,.02,.05):assert released_with(c)=='released too high',c
    assert released_with(.05,stage=C)==''  # carry is unchanged: only the opening rule judges its releases
    e.close()

def test_real_linear_and_angular_velocity_reject_settling():
    for angular in (False,True):
        e=env_for('drop');ev=e._task_evaluator;scene=ev.scene;ks=rooted_stack(e)
        k=ks[0];rest=scene.heights(e.data)
        assert scene.observe(e.data,rest)['supported'][k]
        e.data.qvel[scene.dof_adr[k]+(3 if angular else 0)]=1.
        o=scene.observe(e.data,rest)
        assert not o['supported'][k]
        assert classify(o,np.full(len(scene.names),D),ev.cfg)[k]!=DONE
        e.close()

def test_actual_held_goal_is_not_terminal():
    e=env_for('drop');start,log=build_start(e,SEED_BASE,'drop');assert start is not None,log
    e.reset_to_stage(SEED_BASE,start);r=e.evaluate_task();ev=e._task_evaluator
    assert r.metrics['touched'] and not r.metrics['supported'] and not r.success[0]
    assert ev.chunk_reward()[1] is False
    e.close()

def test_numerical_faults_latch_finite_failure():
    for case in ('time','warning','qpos','qvel'):
        e=env_for();ev=e._task_evaluator
        if case=='time':ev._time=e.data.time+1
        elif case=='warning':e.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number+=1
        elif case=='qpos':e.data.qpos[0]=np.nan
        else:e.data.qvel[0]=np.nan
        r=e.evaluate_task()
        assert np.isfinite(r.reward).all() and not r.success[0] and ev.chunk_reward()==(-1.,True)
        assert ev._failed=='numerical instability'
        r=e.evaluate_task();assert np.isfinite(r.reward).all() and ev.chunk_reward()==(-1.,True)
        e.close()

def test_inactive_parked_parts_and_stock_masks_survive_begin():
    e=env_for();ev=e._task_evaluator;s=ev.scene
    active=[0,next(i for i,n in enumerate(s.names) if n.startswith('bolt_'))]
    ev.set_active_object_joints([s.names[k]+'_joint' for k in active])
    ev.begin(e.model,e.data,active[0])
    assert ev.stock._active_nuts.sum()==1 and ev.stock._active_bolts.sum()==1
    inactive=next(i for i in range(len(s.names)) if i not in active)
    e.data.qpos[s.qpos_adr[inactive]+2]-=1.;mujoco.mj_forward(e.model,e.data)
    assert updates(e)[1] is False,ev._last
    assert ev.target in active
    e.close()

def test_one_move_equals_eight_small_moves():
    e=env_for();ev=e._task_evaluator;initial=ev._phi_chunk;returns=[]
    for count in (1,8):
        ev._phi_chunk=initial;total=0
        for phi in np.linspace(initial,initial+.06,count+1)[1:]:
            ev._last['progress']=phi;r,done=ev.chunk_reward();assert not done;total+=r
        returns.append(total)
    assert abs(returns[0]-returns[1])<1e-6
    e.close()

def test_starts_idle_and_exact_restores():
    for stage in STAGES[:-1]:
        e=env_for(stage);checked=0
        for i in range(20):
            s,_=build_start(e,SEED_BASE+i,stage)
            if s is None:continue
            e.reset_to_stage(s['seed'],s);ev=e._task_evaluator
            assert e.evaluate_task().metrics['label']==stage
            action=idle_action(e);total=0
            for _ in range(10):
                for _ in range(32):e.step(action,render_obs=False)
                r,done=ev.chunk_reward();assert not done,(stage,i,ev._last);total+=r
            assert abs(total)<.05,(stage,i,total)
            results=[]
            for _ in range(2):
                e.reset_to_stage(s['seed'],s)
                for _ in range(32):e.step(action,render_obs=False)
                results.append(e.data.qpos.copy())
            assert np.array_equal(*results)
            checked+=1
            if checked==3:break
        e.close();assert checked==3,(stage,checked)

# ---------------------------------------------------------------------------------------------- handoff amendment
CROSS=SEED_BASE+1  # first cross-side seed (bolt at +y, orange bin at -y): screen_cross.py

def old_classify(obs,prev,cfg):
    """Frozen copy of classify before the handoff amendment (labels reach 0, pick 1, carry 2, drop 3, done 4, fallen 5)."""
    Ro,Po,Co,Do,DONEo=range(5)
    near=(obs['dist']<=np.where(prev==Po,cfg.reach_exit_m,cfg.reach_m))|obs['touched']
    out=np.full(prev.shape,Ro,np.int64);out[near]=Po
    out[obs['grasped']&(obs['lift']>=cfg.lift_m)]=Co
    safe=obs['in_bin']&(obs['above_rim']<=0)
    over=obs['over']|safe|((prev==Do)&(((obs['xy_gap']<=cfg.opening_exit_m)&(obs['above_rim']>=-cfg.opening_exit_m))|obs['inside']))
    out[over&obs['touched']]=Do
    at_rest=(obs['speed']<=cfg.at_rest_speed)&(obs['angular_speed']<=cfg.at_rest_angular_speed)
    free=~obs['touched']&(~at_rest|~obs['resting_contact'])
    out[free]=prev[free]
    out[obs['supported']&~obs['touched']&at_rest]=DONEo
    out[(obs['lift']<-cfg.fallen_m)&~obs['any_bin']]=5
    return out

OLD_TO_NEW=np.array([R,P,C,D,DONE,FALLEN])

def test_same_side_labels_unchanged():
    # Same-side parts are classified exactly as before the amendment, from every stage's start and after moves.
    rng=np.random.default_rng(0);checked=0
    for stage in ('reach','pick','carry','drop'):
        e=env_for(stage);ev=e._task_evaluator;s=ev.scene
        st,_=build_start(e,SEED_BASE,stage);e.reset_to_stage(st['seed'],st);rest=ev.tracker.rest_height
        for step in range(40):
            e.step(idle_action(e)+rng.normal(0,.02,14).astype(np.float32)*(step%2),render_obs=False)
            o=s.observe(e.data,rest);same=~o['cross']
            for prev_old in range(5):
                prev=np.full(len(s.names),prev_old)
                new=classify(o,OLD_TO_NEW[prev],ev.cfg);old=OLD_TO_NEW[old_classify(o,prev,ev.cfg)]
                assert np.array_equal(new[same],old[same]),(stage,step,prev_old)
                checked+=int(same.sum())
        e.close()
    assert checked>1000

def test_cross_side_roles_and_pick_goal():
    e=env_for('handoff',CROSS);st,log=build_start(e,CROSS,'handoff');assert st is not None,log
    e.reset_to_stage(CROSS,st);ev=e._task_evaluator;k=ev.target;o=ev.tracker.obs;n=len(ev.scene.names)
    assert o['cross'][k] and o['receiver'][k]==1 and st['arm']==0 and ev.scene.names[k].startswith('bolt_')
    assert ev.stage_prompt()=='hand the bolt to the right gripper'
    # Held and lifted by the giver = the pick goal of a cross-side part = handoff (not carry).
    assert o['f_give'][k]==2 and classify(o,np.full(n,P),ev.cfg)[k]==H
    # The receiver picking it alone (giver far, nothing else touched) meets the handoff goal: carry.
    q={key:np.array(v,copy=True) for key,v in o.items()}
    q['f_give'][k]=0;q['touch_give'][k]=False;q['f_recv'][k]=2;q['touch_recv'][k]=True;q['gap_give'][k]=.2
    assert classify(q,np.full(n,H),ev.cfg)[k]==C
    q['gap_give'][k]=.05  # giver's pinch centre < 6 cm from the part: not clear yet
    assert classify(q,np.full(n,H),ev.cfg)[k]==H
    q['gap_give'][k]=.2;q['env_touch'][k]=True  # touching something else: not received in the air
    assert classify(q,np.full(n,H),ev.cfg)[k]==H
    assert classify(q,np.full(n,C),ev.cfg)[k]==C  # once in carry, contact with the bin/rim does not regress
    q['touch_give'][k]=True  # the giver touches it again in carry: back to handoff
    assert classify(q,np.full(n,C),ev.cfg)[k]==H
    e.close()

def test_handoff_failures():
    e=env_for('handoff',CROSS);st,_=build_start(e,CROSS,'handoff');e.reset_to_stage(CROSS,st);ev=e._task_evaluator;k=ev.target
    a=idle_action(e);a[st['arm']*7+6]=1.  # the giver opens with nobody else holding it
    for _ in range(64):e.step(a,render_obs=False)
    assert ev.chunk_reward()==(-1.,True) and ev._failed=='dropped during the handoff',ev._last
    e.reset_to_stage(CROSS,st);o={key:np.array(v,copy=True) for key,v in ev.tracker.obs.items()};o['env_touch'][k]=True
    assert ev.step_failure(k,o,H)=='touched the table during the handoff'
    e.close()

def test_both_holding_is_not_the_handoff_goal():
    e=env_for('handoff',CROSS);st,_=build_start(e,CROSS,'handoff');e.reset_to_stage(CROSS,st);ev=e._task_evaluator;k=ev.target
    o={key:np.array(v,copy=True) for key,v in ev.tracker.obs.items()};n=len(ev.scene.names)
    o['f_recv'][k]=2;o['touch_recv'][k]=True
    assert classify(o,np.full(n,H),ev.cfg)[k]==H  # both hold it
    o['f_give'][k]=0  # giver's fingers off, but its hand still touches it
    assert classify(o,np.full(n,H),ev.cfg)[k]==H
    e.close()

def test_cross_carry_giver_touch_and_drop_only_by_receiver():
    e=env_for('carry',CROSS);st,log=build_start(e,CROSS,'carry');assert st is not None,log
    e.reset_to_stage(CROSS,st);ev=e._task_evaluator;k=ev.target;o=ev.tracker.obs;n=len(ev.scene.names)
    assert st['arm']==1 and o['f_recv'][k]==2 and o['f_give'][k]==0 and e.evaluate_task().metrics['label']=='carry'
    q={key:np.array(v,copy=True) for key,v in o.items()};q['touch_give'][k]=True
    assert ev.step_failure(k,q,C)=='giver touched the part again'
    q['over'][k]=True;q['touched'][k]=True
    assert classify(q,np.full(n,C),ev.cfg)[k]!=D  # the giver on it over the opening is not drop
    q['touch_give'][k]=False
    assert classify(q,np.full(n,C),ev.cfg)[k]==D
    e.close()

if __name__=='__main__':
    failed=0;tests=[(n,f) for n,f in list(globals().items()) if n.startswith('test_')]
    for name,fn in tests:
        try:fn();print('ok',name,flush=True)
        except Exception:failed+=1;traceback.print_exc();print('FAIL',name,flush=True)
    print(f'{len(tests)-failed}/{len(tests)} passed',flush=True);raise SystemExit(bool(failed))
