"""Clutter scripted solvers/hacks: physical actions only, exactly 32 actions per scored chunk."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault('MUJOCO_GL','disable');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import mujoco,numpy as np
import abc_sim
from abc_sim.task_eval import clutter_starts as CS, inhand_starts as IS
from abc_sim.task_eval.clutter_starts import idle_action,move_link,carry_plan,native_grasp,plan_link,place_plan,_rotation
from abc_sim.task_eval.stagekit import axis_angle_quat
from abc_sim.task_eval.put_relative_starts import topdown_grasp
from abc_sim.task_eval.stagekit import ARMS,T,arm_ids
class Finished(Exception):pass

def behavior(e,start,name):
 ev,d,m=e._task_evaluator,e.data,e.model;s=ev.scene;arm=int(start['arm'])
 if arm<0:arm=int(ev.tracker.obs['near_arm'][0])
 ids=arm_ids(m,ARMS[arm])
 def hold(chunks,grip=None):
  a=idle_action(e)
  if grip is not None:a[arm*7+6]=grip
  for _ in range(chunks*32):e.step(a,render_obs=False)
 def shift(delta,grip=0.,held=True,allow_target_collision=False):
  goal=T(d.xpos[ids.link6],d.xquat[ids.link6]);goal[:3,3]+=delta
  return move_link(e,arm,goal,held=held,grip=grip,actions=96,allow_target_collision=allow_target_collision)
 def lower():
  # Same collision-checked lowering the place start builder verified (release 3 cm above the floor).
  ok,why=place_plan(e,arm,execute=True);return ok,('' if ok else why)
 def command(actor,goal,grip=0.,actions=96,settle=0):
  # Deliberate contact policies use real joint actions. Collision avoidance is
  # disabled only here: their purpose is to push an object/fixture physically.
  # settle > 0 keeps commanding the final joint target (a fast push must not stop where the arm lagged).
  path,why=plan_link(e,actor,goal,collision_check=False)
  if path is None:return False,why
  a=idle_action(e)
  for q in path:
   a[actor*7:actor*7+6]=q;a[actor*7+6]=grip
   for _ in range(max(1,actions//len(path))):e.step(a,render_obs=False)
  for _ in range(settle):e.step(a,render_obs=False)
  return True,''
 def local_hold(actor,chunks,grip):
  a=idle_action(e);a[actor*7+6]=grip
  for _ in range(chunks*32):e.step(a,render_obs=False)
 def object_approach(k,actor):
  angle=float(np.arctan2(d.xmat[s.bodies[k]].reshape(3,3)[1,0],d.xmat[s.bodies[k]].reshape(3,3)[0,0]))
  for yaw in (angle,angle+np.pi):
   grasp=native_grasp(d,s,k,yaw,actor,pad_z=.145);goal=s.object_T(d,k)@grasp['T_obj_link6'];above=goal.copy();above[2,3]+=.10
   path,why=plan_link(e,actor,above,collision_check=False)
   if path is None:continue
   ok,why=command(actor,above,1.)
   if ok:
    ok,why=command(actor,goal,1.)
    if ok:return True,goal,''
  return False,None,'no feasible adversarial approach'
 if start['stage']=='handoff' or name=='regrasp_giver':
  giver=int(ev.tracker.obs['giver']);B=1-giver
  def transported():
   plan,why=CS.plan_handoff(e,giver)
   if plan is None:return None,'plan: '+why
   ok,why=move_link(e,giver,plan[0],held=True,actions=128)
   return (plan[1] if ok else None),why
  if name=='solver':return CS.handoff_solver(e,giver)
  if name=='drop_between':
   # The giver brings it to the region and lets go before the receiver has it.
   hint,why=transported()
   if hint is None:return False,why
   local_hold(giver,4,1.);return True,''
  if name=='both_hold':
   # The receiver grasps and closes; the giver never opens.
   hint,why=transported()
   if hint is None:return False,why
   got=IS.check_receiver(e,CS.TargetView(s),B,s.object_T(d,0)@hint)
   if got is None:return False,'planned receiver grasp no longer reachable'
   qpre=IS.approach_pose(e,B,*got)
   if qpre is None:return False,'pre-grasp unreachable'
   IS.move_joints(e,B,qpre,actions=96,grip=1.);ok,why=IS.move_link(e,B,got[0],actions=48,grip=1.,n_way=4)
   if not ok:return False,why
   IS.run(e,IS.action_with(e,None,{B:0.}),32);hold(6);return True,''
  if name=='set_down_regrasp':
   # The giver puts it back down on the clutter/table, lets go, and the receiver reaches for it there.
   low=float(np.min(s.world_points(d)[:,2]))-s.table_z
   ok,why=IS.move_link(e,giver,_down(e,giver,low),actions=96,grip=0.,tol=(.015,.4))
   if not ok:return False,why
   local_hold(giver,1,1.);CS.retreat(e,giver)
   view=CS.TargetView(s)
   for T6 in CS.receiver_candidates(e,view,giver)[:36]:
    got=IS.check_receiver(e,view,B,T6)
    if got is None:continue
    IS.move_joints(e,B,got[1],actions=96,grip=1.);local_hold(B,2,0.);return True,''
   local_hold(B,2,1.);return True,'receiver found no grasp on the set-down target (still a set-down)'
  if name=='giver_alone':
   # The giver carries it towards the box by itself: into the handoff region, then on towards the box as far as its
   # IK reaches, and holds it there (G5 repair round 1: a straight path from the clutter was unplannable on 19/20).
   ok,why=CS.transport(e,giver)
   if not ok:return False,'transport: '+why
   box=d.qpos[s.box_adr:s.box_adr+3];sgn=np.sign(box[1])
   for dy in (.20,.15,.10,.07,.05):
    goal=IS.link6_T(e,giver);goal[1,3]+=sgn*dy
    if plan_link(e,giver,goal,held=True)[0] is None:continue
    ok,why=move_link(e,giver,goal,held=True,actions=128)
    if ok:hold(4);return True,f'{dy:.2f} m further towards the box'
   hold(4);return True,'could not go beyond the handoff region'
  if name=='regrasp_giver':
   # Carry after a handoff: the giver reaches back for the held target with open fingers and closes.
   view=CS.TargetView(s);tries=0
   for T6 in CS.receiver_candidates(e,view,B)[:36]:
    got=IS.check_receiver(e,view,giver,T6)
    if got is None:continue
    qpre=IS.approach_pose(e,giver,*got)
    if qpre is None:continue
    IS.move_joints(e,giver,qpre,actions=96,grip=1.);ok,why=IS.move_link(e,giver,got[0],actions=48,grip=1.,n_way=4)
    if not ok:continue
    IS.run(e,IS.action_with(e,None,{giver:0.}),32)
    if ev.tracker.obs['touch_give'][0] or tries>=2:hold(2);return True,''
    # Closed on air beside the receiver's gripper (carry G5 repair round 1): open, back off, next grasp.
    tries+=1;IS.run(e,IS.action_with(e,None,{giver:1.}),12);IS.move_joints(e,giver,qpre,actions=32,grip=1.)
   return False,'giver has no free grasp on the held target'
 if name=='wrong_lift':
  actor=1-arm if start['arm']>=0 else arm
  distances=np.linalg.norm(d.subtree_com[s.bodies[1:]]-d.site_xpos[s.sites[actor]],axis=1)
  for k in np.argsort(distances)+1:
   ok,goal,why=object_approach(int(k),actor)
   if not ok:continue
   local_hold(actor,1,0.);goal[2,3]+=.10;ok,why=command(actor,goal,0.)
   if ok:local_hold(actor,2,0.);return True,''
  return False,'wrong-object lift could not be exercised'
 if name=='distractor_loss':
  # Drag a distractor (fingers closed around it) off the nearest table edge, holding the final target.
  actor=1-arm if start['arm']>=0 else arm
  x0,x1,y0,y1=s.table_bounds;com=d.subtree_com[s.bodies[1:]]
  edges=np.stack([com[:,0]-x0,y1-com[:,1],com[:,1]-y0],1)  # near (robot) edge and the two side edges
  dirs=np.array([[-1.,0,0],[0,1.,0],[0,-1.,0]])
  for k in np.argsort(edges.min(1))+1:
   j=int(np.argmin(edges[k-1]))
   ok,goal,why=object_approach(int(k),actor)
   if not ok:continue
   goal[:3,3]+=dirs[j]*(edges[k-1,j]+.10);local_hold(actor,1,0.);ok,why=command(actor,goal,0.,actions=48,settle=96)
   if ok:return True,''
  return False,'distractor-loss push could not be exercised'
 if name in ('box_move','box_tip') and start['stage']=='place':
  # Holding arm is the box-side arm: lower the held target into the box, then drag (or lift-drag) the box with it.
  ok,why=lower()
  if not ok:return False,why
  rb=d.xmat[s.box_body].reshape(3,3);bp=d.qpos[s.box_adr:s.box_adr+3]
  base=d.xpos[m.jnt_bodyid[m.joint(ARMS[arm]+'_joint1').id]];toward=base-bp;toward[2]=0;toward/=np.linalg.norm(toward)
  goal=T(d.xpos[ids.link6],d.xquat[ids.link6]);goal[:3,3]+=toward*.15
  if name=='box_tip':goal[2,3]+=.08
  ok,why=command(arm,goal,0.,actions=24,settle=128)
  return ok,why
 if name in ('box_move','box_tip'):
  # Push the box through its centre with the arm on the box's side. Candidates: the four box sides, two jaw
  # yaws and a level / leaning hand; the first whose approach, contact and push poses all have IK is used.
  rb=d.xmat[s.box_body].reshape(3,3);bp=d.qpos[s.box_adr:s.box_adr+3];side=0 if bp[1]>0 else 1
  h=.10 if name=='box_tip' else .05;tried=0;executed=0
  # Box-side arm first; the other arm too unless it holds the target (reach: no arm holds it).
  actors=[side]+([1-side] if start['stage']=='reach' or start['arm']!=1-side or start['stage']=='pick' else [])
  if start['stage']=='pick':actors.sort(key=lambda a:a==start['arm'])  # the pick arm leaving = regression first
  for actor in actors:
   held=start['arm']==actor and start['stage']!='pick'
   actor_grip=0. if held else 1.
   base=d.xpos[m.jnt_bodyid[m.joint(ARMS[actor]+'_joint1').id]]
   sides=[rb@n for n in (np.array([1.,0,0]),np.array([-1.,0,0]),np.array([0,1.,0]),np.array([0,-1.,0]))]
   sides.sort(key=lambda n:float(np.linalg.norm((bp+n*.13)[:2]-base[:2])))
   for normal in sides:
    for yaw in (0.,np.pi/2):
     for lean in (0.,-30.,30.,-45.):
      for off in (.13,.11):
       r=np.array([[np.cos(yaw),np.sin(yaw),0.],[np.sin(yaw),-np.cos(yaw),0.],[0.,0.,-1.]])
       if lean:
        axis=np.array([-normal[1],normal[0],0.]);r=_rotation(axis_angle_quat(axis,np.deg2rad(lean)))@r
       contact=bp+normal*off+np.array([0,0,h]);pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=contact-r@np.array([0.,-.044,.1297])
       above=pose.copy();above[2,3]+=.12;push=pose.copy();push[:3,3]-=normal*.20;tried+=1
       if not all(plan_link(e,actor,g,collision_check=False)[0] is not None for g in (above,pose,push)):continue
       for goal in (above,pose):
        ok,why=command(actor,goal,actor_grip)
        if not ok:return False,why
       ok,why=command(actor,push,actor_grip,actions=24,settle=128)
       o=s.observe(d,ev.tracker.rest_height)
       if not ok or o['box_moved']>.01 or o['box_tilt']>.05 or executed>=2:return ok,why
       # The push missed the box (handoff G5 repair round 1): back up and try the next push pose.
       executed+=1;command(actor,above,actor_grip)
  return False,f'no reachable box push pose in {tried} candidates'
 if name=='reach_push':
  grasp=native_grasp(d,s,0,0.,arm,pad_z=.1297);goal=s.object_T(d,0)@grasp['T_obj_link6'];direction=goal[:3,0].copy()
  far=goal.copy();far[:3,3]-=direction*.10;above=far.copy();above[2,3]+=.10
  for pose in (above,far):
   ok,why=command(arm,pose,0.)
   if not ok:return False,why
  goal[:3,3]+=direction*.10;ok,why=command(arm,goal,0.,actions=12,settle=96)
  return ok,why
 if name=='idle':hold(10);return True,''
 if name=='away':
  a=idle_action(e);scratch=mujoco.MjData(m);choices=[]
  target=d.subtree_com[s.bodies[0]]
  for sign in (-1,1):
   scratch.qpos[:]=d.qpos;scratch.qpos[ids.qadr[0]]+=sign*.15;mujoco.mj_forward(m,scratch)
   choices.append((np.linalg.norm(scratch.site_xpos[s.sites[arm]]-target),sign))
  a[arm*7]+=max(choices)[1]*.15;a[arm*7+6]=1
  for _ in range(64):e.step(a,render_obs=False)
  return True,''
 if name=='close_only':hold(5,0);return True,''
 if name in ('release_high','release_early'):hold(10,1);return True,''
 if name=='lift_open':
  a=idle_action(e);scratch=mujoco.MjData(m);choices=[]
  for sign in (-1,1):
   scratch.qpos[:]=d.qpos;scratch.qpos[ids.qadr[1]]+=sign*.20;mujoco.mj_forward(m,scratch)
   choices.append((scratch.site_xpos[s.sites[arm],2],sign))
  a[arm*7+1]+=max(choices)[1]*.20;a[arm*7+6]=1.
  for _ in range(96):e.step(a,render_obs=False)
  return True,''
 if name=='push':
  # Open jaws sweep laterally from the valid pregrasp; the target is never teleported.
  hold(1,1);goal=T(d.xpos[ids.link6],d.xquat[ids.link6]);goal[:3,3]+=goal[:3,0]*.10
  ok,why=command(arm,goal,1.,actions=24,settle=96)
  return ok,why
 if name=='wrong_way':
  box=d.qpos[s.box_adr:s.box_adr+3];origin=d.subtree_com[s.bodies[0]]
  direction=origin[:2]-box[:2];direction/=max(np.linalg.norm(direction),1e-9)
  return shift(np.r_[direction*.015,0.])
 if name=='shake':
  for sign in (1,-1,1,-1):
   ok,why=shift(np.array([sign*.008,0,0]))
   if not ok:return ok,why
  return True,''
 if name=='lower_hold':
  ok,why=lower()
  if ok:hold(5,0)
  return ok,why
 if name=='leave_opening':
  notes=[]
  for delta in ((.10,0,0),(-.10,0,0),(0,.10,0),(0,-.10,0)):
   ok,why=shift(np.array(delta))
   if ok:return True,''
   notes.append(why)
  return False,'; '.join(notes)
 if name=='wrong_object':
  # Choose a genuinely reachable non-target, then approach it with open fingers.
  notes=[]
  for k in np.argsort(s.contact_obs(d)['dist'][1:])+1:
   grasp=topdown_grasp(d,s,int(k),0.,arm,below_com_m=-.03)
   goal=s.object_T(d,int(k))@grasp['T_obj_link6']
   ok,why=move_link(e,arm,goal,held=False,grip=1,actions=128)
   if ok:hold(2,1);return True,''
   notes.append(why)
  return False,'; '.join(notes)
 if name=='solver':
  if start['stage']=='reach':
   grasp=topdown_grasp(d,s,0,0.,arm,below_com_m=-.025)
   goal=s.object_T(d,0)@grasp['T_obj_link6'];above=goal.copy();above[2,3]+=.10
   for pose in (above,goal):
    ok,why=move_link(e,arm,pose,held=False,grip=1,actions=96)
    if not ok:return False,why
   hold(2,1);return True,''
  if start['stage']=='pick':
   pad=d.xpos[ids.link6]+d.xmat[ids.link6].reshape(3,3)@np.array([0.,-.044,.1297])
   correction=d.subtree_com[s.bodies[0]]-pad
   if np.linalg.norm(correction)>.03 and np.linalg.norm(correction)<.07:
    correction*=min(1.,.025/np.linalg.norm(correction));correction[2]=max(correction[2],-.01)
    ok,why=shift(correction,grip=1.,held=False,allow_target_collision=True)
    if not ok:return False,why
   hold(1,0);ok,why=shift(np.array([0.,0.,.08]))
   if ok:hold(2,0)
   return ok,why
  if start['stage']=='carry':
   ok,why=carry_plan(e,arm,execute=True)
   if ok:hold(2,0)
   return ok,why
  if start['stage']=='place':
   ok,why=lower()
   if ok:hold(10,1)
   return ok,why
 raise ValueError((start['stage'],name))

def _down(e,arm,dz):
 goal=IS.link6_T(e,arm);goal[2,3]-=dz;return goal

def run(job):
 stage,start,name,expect,out=job;e=abc_sim.make_env(task='grab_specific_object_from_clutter__'+stage,render_cameras=False)
 e.reset_to_stage(start['seed'],start);ev=e._task_evaluator;actions=[];rewards=[];metrics=[];step=e.step
 def traced(a,**kwargs):
  result=step(a,**kwargs);actions.append(np.asarray(a).copy())
  if len(actions)%32==0:
   r,done=ev.chunk_reward();rewards.append(r);metrics.append(dict(ev._last))
   if done:raise Finished()
  return result
 e.step=traced;planned=True;why=''
 try:planned,why=behavior(e,start,name)
 except Finished:pass
 except Exception as ex:planned=False;why=repr(ex)
 outcome='+1' if ev._success else '-1' if ev._failed else '0'
 required_reason=dict(box_move='box moved or tipped',box_tip='box moved or tipped',wrong_lift='wrong object lifted',distractor_loss='distractor off table',reach_push='pushed target',push='pushed target').get(name)
 exercised=required_reason is None or ev._failed==required_reason
 ret=float(sum(rewards))
 matched=(outcome!='+1' and (name!='both_hold' or ret<=.5)) if expect=='not+1' else outcome==expect
 record=dict(seed=int(start['seed']),behavior=name,expect=expect,planned=bool(planned),outcome=outcome,exercised=exercised,as_expected=bool(planned) and matched,return_=float(sum(rewards)),actions=len(actions),chunks=len(rewards),failed=ev._failed,why=why,target_moved_xy_m=float(ev.tracker.obs['moved_xy'][0]),first_event=ev._failed or ('stage goal' if ev._success else 'none'))
 path=Path(out)/stage;path.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(path/f'{name}_{start["seed"]}.npz',actions=np.asarray(actions),rewards=rewards)
 (path/f'{name}_{start["seed"]}.json').write_text(json.dumps(dict(record=record,metrics=metrics),indent=2));e.close();return record

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('reach','pick','handoff','carry','place'));p.add_argument('--scene',choices=('all','handoff','direct'),default='all');p.add_argument('--starts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=20);p.add_argument('--workers',type=int,default=6);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 print('CHECKOUT',abc_sim.__file__,flush=True)
 behaviors=dict(solver='+1',idle='0',box_move='-1',box_tip='-1',distractor_loss='-1')
 behaviors.update(dict(away='0',wrong_object='0',reach_push='-1') if a.stage=='reach' else dict(drop_between='-1',both_hold='not+1',set_down_regrasp='-1',giver_alone='not+1') if a.stage=='handoff' else dict(close_only='0',push='-1',lift_open='-1',wrong_lift='-1') if a.stage=='pick' else dict(release_early='-1',wrong_way='0',shake='0',**(dict(regrasp_giver='-1') if a.scene=='handoff' else {})) if a.stage=='carry' else dict(release_high='-1',lower_hold='0',leave_opening='-1'))
 z=np.load(a.starts/f'{a.stage}.npz',allow_pickle=True)
 flags=z['handoff_scene'] if 'handoff_scene' in z.files else np.full(len(z['seed']),a.stage=='handoff')
 keep=[i for i in range(len(z['seed'])) if a.scene=='all' or bool(flags[i])==(a.scene=='handoff')][:a.count]
 rows=[dict(seed=int(z['seed'][i]),target=int(z['target'][i]),arm=int(z['arm'][i]),stage=a.stage,state=z['state'][i],rest_height=z['rest_height'][i],labels=z['labels'][i]) for i in keep]
 if len(rows)<12:raise RuntimeError(f'need 12+ starts; have {len(rows)}')
 records=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for record in pool.map(run,[(a.stage,r,b,x,str(a.out)) for b,x in behaviors.items() for r in rows]):records.append(record);print(json.dumps(record),flush=True)
 summary={}
 for name in behaviors:
  rs=[r for r in records if r['behavior']==name]
  summary[name]=dict(tried=len(rs),planned=sum(r['planned'] for r in rs),successes=sum(r['outcome']=='+1' for r in rs),as_expected=sum(r['as_expected'] for r in rs),exercised=sum(r.get('exercised',True) for r in rs),mean_return=float(np.mean([r['return_'] for r in rs])))
 sol=summary['solver'];summary['G4']=sol['successes']/sol['tried']>=.4 and sol['successes']/max(sol['planned'],1)>=.7
 hacks=[v for k,v in summary.items() if k not in ('solver','G4')];summary['margin']=sol['mean_return']-max(v['mean_return'] for v in hacks)
 summary['G5']=summary['margin']>=.5 and all(h['as_expected']/h['tried']>=.8 and h['successes']/h['tried']<=.1 for h in hacks)
 (a.out/f'summary_{a.stage}{"" if a.scene=="all" else "_"+a.scene}.json').write_text(json.dumps(dict(summary=summary,rows=records),indent=2));print('SUMMARY',json.dumps(summary),flush=True)
if __name__=='__main__':main()
