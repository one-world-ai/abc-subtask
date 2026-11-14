"""CPU clutter reward contracts, physical support/release checks, and exact restores."""
import os
os.environ.setdefault('MUJOCO_GL','disable');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import traceback
import mujoco
import numpy as np
import abc_sim
from abc_sim.task_eval.clutter_stages import R,P,H,C,PL,DONE,FALLEN,STAGES,StageConfig,classify,progress
from abc_sim.task_eval.clutter_starts import SEED_BASE,build_start,idle_action,_teleport_held
from abc_sim.task_eval.stagekit import grasp_from_state,arm_ids,ARMS,axis_angle_quat,qmul,set_object_pose
from abc_sim.task_specs import get_task_spec

TASK='grab_specific_object_from_clutter'
def env_for(stage='reach',seed=SEED_BASE):
 e=abc_sim.make_env(task=f'{TASK}__{stage}',render_cameras=False);e.forget_arm_state();e.reset(seed=seed);e.evaluate_task();return e

def updates(e):
 for _ in range(6):e.evaluate_task()
 return e._task_evaluator.chunk_reward()

def test_lookup_preserves_stock_and_target_identity():
 assert get_task_spec(TASK).evaluator_name=='grab_clutter_target_in_box'
 for s in STAGES[:-1]:
  spec=get_task_spec(f'{TASK}__{s}');assert get_task_spec(spec.prompt).name==spec.name
 e=env_for();ev=e._task_evaluator;assert ev.target==0 and ev.scene.names[ev.target]=='target_object'
 try:ev.begin(e.model,e.data,1)
 except ValueError:pass
 else:raise AssertionError('distractor accepted as target')
 e.close()

def test_qpos_only_is_rejected():
 e=env_for()
 try:e._task_evaluator.evaluate_qpos_batch(e.data.qpos[None])
 except RuntimeError as ex:assert 'real CPU contacts and velocities' in str(ex)
 else:raise AssertionError('manufactured zero velocities')
 e.close()

def test_moved_or_tipped_box_fails():
 for tip in (False,True):
  e=env_for();ev=e._task_evaluator;a=ev.scene.box_adr
  if tip:e.data.qpos[a+3:a+7]=qmul(axis_angle_quat(np.array([1.,0,0]),np.pi/2),e.data.qpos[a+3:a+7].copy())
  else:e.data.qpos[a]+=.04
  mujoco.mj_forward(e.model,e.data);assert updates(e)==(-1.,True);assert ev._failed=='box moved or tipped';e.close()

def test_push_and_distractor_loss_fail():
 for other in (False,True):
  e=env_for();ev=e._task_evaluator;a=ev.scene.qpos_adr[1 if other else 0]
  e.data.qpos[a]+=.6 if other else .04;mujoco.mj_forward(e.model,e.data)
  assert updates(e)==(-1.,True);assert ev._failed==('distractor off table' if other else 'pushed target');e.close()

def test_wrong_object_lift_and_harmless_nudge():
 e=env_for();ev=e._task_evaluator;o=ev.tracker.obs
 assert ev.step_failure(0,o,R)==''
 o['moved_xy'][1]=.02;assert ev.step_failure(0,o,R)==''
 o['lift'][1]=.04;o['touched'][1]=True;assert ev.step_failure(0,o,R)=='wrong object lifted';e.close()

def test_all_release_events_respect_opening_and_height():
 e=env_for('place');ev=e._task_evaluator;o=ev.tracker.obs
 o['touched'][0]=False;o['in_opening'][0]=False;o['in_box'][0]=False;o['clearance'][0]=.01
 ev._released=True;ev._was_touched=True;assert ev.step_failure(0,o,PL)=='released outside opening'
 ev._was_touched=True;o['in_opening'][0]=True;o['clearance'][0]=.10
 assert ev.step_failure(0,o,PL)=='released too high'
 ev._was_touched=True;o['clearance'][0]=.01;assert ev.step_failure(0,o,PL)==''
 e.close()

def test_angular_motion_is_not_a_landing():
 e=env_for('place');ev=e._task_evaluator;o=ev.tracker.obs
 ev._released=True;ev._was_touched=False;o['touched'][0]=False;o['resting_contact'][0]=True
 o['speed'][0]=0;o['angular_speed'][0]=1;o['in_box'][0]=False
 assert ev.step_failure(0,o,PL)=='';e.close()

def test_floating_target_cannot_finish_place():
 e=env_for('place');ev=e._task_evaluator;s=ev.scene;rest=s.heights(e.data);b=e.data.qpos[s.box_adr:s.box_adr+3].copy()
 local=np.array([0,0,.08]);r=e.data.xmat[s.box_body].reshape(3,3)
 set_object_pose(e,s,0,b+r@local,e.data.qpos[s.qpos_adr[0]+3:s.qpos_adr[0]+7].copy())
 o=s.observe(e.data,rest);assert o['in_box'][0] and not o['floor_support'][0]
 assert classify(o,np.full(len(s.names),PL),ev.cfg)[0]!=DONE;e.close()

def test_two_real_floor_supported_placements_finish():
 for seed in (SEED_BASE,SEED_BASE+1):
  e=env_for('place',seed);ev=e._task_evaluator;s=ev.scene;rest=s.heights(e.data)
  b=e.data.qpos[s.box_adr:s.box_adr+3].copy();r=e.data.xmat[s.box_body].reshape(3,3)
  set_object_pose(e,s,0,b+r@np.array([0,0,.08]),e.data.qpos[s.qpos_adr[0]+3:s.qpos_adr[0]+7].copy())
  e.data.ctrl[e._ctrl_indices]=e.data.qpos[e._qpos_indices]
  for _ in range(1500):mujoco.mj_step(e.model,e.data)
  labels=np.full(len(s.names),R);labels[0]=PL;ev.begin(e.model,e.data,0,rest_height=rest,labels=labels)
  assert updates(e)==(1.,True),(seed,ev._last)
  assert ev._last['floor_support'] and not ev._last['touched'];e.close()

def test_progress_is_independent_of_partition_on_real_poses():
 e=env_for();ev=e._task_evaluator;q=e.data.qpos.copy();rest=ev.tracker.rest_height.copy();ids=arm_ids(e.model,'left');totals=[]
 for n in (1,8):
  e.data.qpos[:]=q;mujoco.mj_forward(e.model,e.data);ev.begin(e.model,e.data,0,rest_height=rest,labels=np.full(len(ev.scene.names),R));e.evaluate_task();total=0
  for delta in np.linspace(0,.025,n+1)[1:]:
   e.data.qpos[ids.qadr[0]]=q[ids.qadr[0]]+delta;mujoco.mj_forward(e.model,e.data);e.evaluate_task();r,done=ev.chunk_reward();assert not done;total+=r
  totals.append(total)
 assert abs(totals[0]-totals[1])<1e-6;e.close()

HANDOFF_SEED=SEED_BASE+1  # donut, picked by the left arm, box at -y: a handoff scene
def test_built_starts_idle_and_restore_exactly():
 for stage,seed in [(s,SEED_BASE) for s in STAGES[:-1] if s!='handoff']+[('handoff',HANDOFF_SEED),('carry',HANDOFF_SEED)]:
  e=env_for(stage,seed);start,log=build_start(e,seed,stage)
  assert start is not None,(stage,log[-3:]);e.reset_to_stage(start['seed'],start)
  assert e.evaluate_task().metrics['label']==stage
  a=idle_action(e);total=0
  for _ in range(10):
   for _ in range(32):e.step(a,render_obs=False)
   r,done=e._task_evaluator.chunk_reward();assert not done,(stage,e._task_evaluator._last);total+=r
  assert abs(total)<.05,(stage,total)
  states=[]
  for _ in range(2):
   e.reset_to_stage(start['seed'],start)
   for _ in range(32):e.step(a,render_obs=False)
   states.append(e.data.qpos.copy())
  assert np.array_equal(*states),stage;e.close()

def test_numerical_warnings_are_invalid():
 e=env_for();e.data.warning[0].number=1
 try:e.evaluate_task()
 except RuntimeError as ex:assert 'invalid MuJoCo' in str(ex)
 else:raise AssertionError('numerical warning accepted')
 e.close()

def test_actual_high_release_fails():
 e=env_for('place');start,log=build_start(e,SEED_BASE,'place');assert start is not None,log
 e.reset_to_stage(start['seed'],start);a=idle_action(e);a[start['arm']*7+6]=1
 for _ in range(32):e.step(a,render_obs=False)
 assert e._task_evaluator.chunk_reward()==(-1.,True),e._task_evaluator._last
 assert e._task_evaluator._failed=='released too high';e.close()

def test_shoving_contact_is_not_a_reach():
 # A finger on a moving (shoved) target must not enter pick from reach; the same contact at rest does.
 e=env_for();ev=e._task_evaluator;o={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in ev.tracker.obs.items()};n=len(ev.scene.names)
 o['touched'][0]=True;o['dist'][0]=.2;o['speed'][0]=.10;prev=np.full(n,R)
 assert classify(o,prev,ev.cfg)[0]==R
 o['speed'][0]=.014;assert classify(o,prev,ev.cfg)[0]==R  # slow shove measured in scripted_r2
 o['speed'][0]=0.;o['angular_speed'][0]=0.;o['moved_xy'][0]=0.;assert classify(o,prev,ev.cfg)[0]==P
 o['moved_xy'][0]=.035;assert classify(o,prev,ev.cfg)[0]==R
 o['speed'][0]=.10;prev[0]=P;assert classify(o,prev,ev.cfg)[0]==P  # an ongoing grasp keeps its pick label
 e.close()

def test_place_start_has_verified_lowering():
 e=env_for('place');start,log=build_start(e,SEED_BASE,'place');assert start is not None,log
 assert log[-1].get('lowering_verified') is True,log[-1];e.close()

def _obs(e):
 return {k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in e._task_evaluator.tracker.obs.items()}

def test_handoff_scene_roles_and_direct_scene_has_no_handoff_start():
 e=env_for('handoff',HANDOFF_SEED);o=_obs(e);assert (o['giver'],o['receiver'])==(0,1),(o['giver'],o['receiver'])
 assert build_start(e,SEED_BASE,'handoff')[0] is None  # direct-reach scene (box arm picks)
 e.close()

def test_handoff_classify_chain():
 e=env_for('handoff',HANDOFF_SEED);o=_obs(e);n=len(e._task_evaluator.scene.names);cfg=StageConfig();prev=np.full(n,P)
 def lab(prev_label,**kv):
  x={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in o.items()}
  for k,v in kv.items():x[k][0]=v
  p=np.full(n,R);p[0]=prev_label;return int(classify(x,p,cfg)[0])
 lifted=dict(lift=.08,touched=True,grasped=True,fingers=2,dist=.01)
 # giver holds it lifted: handoff, not carry (pick goal in a handoff scene); still touching clutter: stays pick
 assert lab(P,f_give=2,touch_give=True,**lifted)==H
 assert lab(P,f_give=2,touch_give=True,env_touch=True,**lifted)==P
 inm=dict(com=np.array([.55,0.,.95]),m_gap=0.)
 # both hold in the region: handoff; receiver alone but giver within 6 cm: handoff; giver clear: carry
 assert lab(H,f_give=2,f_recv=2,touch_give=True,touch_recv=True,**inm,**lifted)==H
 assert lab(H,f_give=0,f_recv=2,touch_give=False,touch_recv=True,gap_give=.03,**inm,**lifted)==H
 assert lab(H,f_give=0,f_recv=2,touch_give=False,touch_recv=True,gap_give=.07,**inm,**lifted)==C
 # received while touching something else is not the goal
 assert lab(H,f_give=0,f_recv=2,touch_give=False,touch_recv=True,gap_give=.07,env_touch=True,**inm,**lifted)!=C
 # direct-reach pick by the box arm goes straight to carry (unchanged chain)
 assert lab(P,f_give=0,f_recv=2,touch_give=False,touch_recv=True,gap_give=.30,**lifted)==C
 # in carry, the giver grabbing it again sends it back to handoff (a regression = -1)
 assert lab(C,f_give=2,f_recv=2,touch_give=True,touch_recv=True,**inm,**lifted)==H
 # the giver can never be the arm over the box
 assert lab(C,f_give=2,f_recv=0,touch_give=True,touch_recv=False,over=True,**lifted)!=PL
 e.close()

def test_handoff_failures_fire():
 # giver opens: dropped; giver lowers it onto the clutter/table: touched the table
 e=env_for('handoff',HANDOFF_SEED);start,log=build_start(e,HANDOFF_SEED,'handoff');assert start,log
 for case in ('drop','set_down'):
  e.reset_to_stage(start['seed'],start);ev=e._task_evaluator;a=idle_action(e)
  if case=='drop':a[start['arm']*7+6]=1.
  else:
   s=ev.scene;rest=ev.tracker.rest_height;com=e.data.subtree_com[s.bodies[0]].copy()
   set_object_pose(e,s,0,e.data.qpos[s.qpos_adr[0]:s.qpos_adr[0]+3]-np.array([0,0,com[2]-rest[0,2]-.002]),e.data.qpos[s.qpos_adr[0]+3:s.qpos_adr[0]+7].copy())
  r=None
  for _ in range(4):
   for _ in range(32):e.step(a,render_obs=False)
   r=ev.chunk_reward()
   if r[1]:break
  assert r==(-1.,True),(case,ev._last);assert ev._failed==('dropped during the handoff' if case=='drop' else 'touched the table during the handoff'),(case,ev._failed)
 e.close()

def test_handoff_progress_partition_and_terms():
 e=env_for('handoff',HANDOFF_SEED);start,log=build_start(e,HANDOFF_SEED,'handoff');assert start,log
 e.reset_to_stage(start['seed'],start);ev=e._task_evaluator;o=_obs(e);cfg=ev.cfg
 base=float(progress(H,o,cfg)[0]);assert abs(base+min(o['m_gap'][0],cfg.transport_clip))<1e-9  # giver only: transport term
 x=dict(o);x['m_gap']=np.r_[0.,o['m_gap'][1:]];x['f_recv']=np.r_[2,o['f_recv'][1:]];x['f_give']=np.r_[0,o['f_give'][1:]]
 x['touch_give']=np.r_[False,o['touch_give'][1:]];x['gap_give']=np.r_[.06,o['gap_give'][1:]]
 assert abs(float(progress(H,x,cfg)[0])-1.25)<1e-9
 q=e.data.qpos.copy();ids=arm_ids(e.model,ARMS[start['arm']]);totals=[]
 rel=grasp_from_state(e.model,e.data,ev.scene,0,start['arm'])['T_obj_link6']
 for n in (1,8):
  e.data.qpos[:]=q;mujoco.mj_forward(e.model,e.data);ev.begin(e.model,e.data,0,rest_height=start['rest_height'],labels=start['labels']);e.evaluate_task();total=0
  for delta in np.linspace(0,.02,n+1)[1:]:
   qa=q[ids.qadr].copy();qa[0]+=delta;_teleport_held(e,start['arm'],qa,rel);e.evaluate_task();r,done=ev.chunk_reward();assert not done,ev._last;total+=r
  totals.append(total)
 assert abs(totals[0]-totals[1])<1e-6 and abs(totals[0])>1e-4,totals;e.close()

def main():
 tests=[fn for name,fn in globals().items() if name.startswith('test_') and callable(fn)];failed=0
 for fn in tests:
  try:fn();print('ok',fn.__name__,flush=True)
  except Exception:failed+=1;traceback.print_exc();print('FAIL',fn.__name__,flush=True)
 print(f'{len(tests)-failed}/{len(tests)} passed',flush=True);raise SystemExit(bool(failed))
if __name__=='__main__':main()
