"""CPU scorer audit; run from this checkout with its PYTHONPATH. Writes measured evidence, no stage code."""
from pathlib import Path
import json, os
os.environ.setdefault('MUJOCO_GL','disable')
import abc_sim, mujoco, numpy as np
from abc_sim.task_eval.bottles import _quat_to_rotmat_batch
OUT=Path(__file__).parent
TASK='sort_nuts_and_bolts'

def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x

def env(seed=0):
    e=abc_sim.make_env(task=TASK,render_cameras=False)
    e.forget_arm_state(); e.reset(seed=seed)
    return e

def score(e):
    r=e.evaluate_task()
    return {'reward':float(r.reward[0]),'success':bool(r.success[0]),'metrics':plain(r.metrics)}

def to_world(e,binadr,local):
    q=e.data.qpos[binadr:binadr+7]
    R=_quat_to_rotmat_batch(q[3:][None])[0]
    return q[:3]+R@local

def put_group(e,addrs,binadr,center):
    for i,a in enumerate(addrs):
        # Small grid: each body origin is strictly inside the declared region. Hand poses are evaluated before settling.
        local=center+np.array([((i%5)-2)*.018,((i//5)-1.5)*.018,0])
        e.data.qpos[a:a+3]=to_world(e,binadr,local)
    mujoco.mj_forward(e.model,e.data)

def handmade():
    rows=[]
    for case in ('spec','inverted','mixed','all_in_one_bin','floating_spec','tipped_90_spec','tipped_180_spec'):
        e=env();v=e._task_evaluator
        if case.startswith('tipped'):
            angle=np.pi/2 if '90' in case else np.pi
            for a in (v._nuts_bin_qpos_addr,v._bolts_bin_qpos_addr):
                e.data.qpos[a+3:a+7]=[np.cos(angle/2),np.sin(angle/2),0,0]
        na,ba=v._nuts_bin_qpos_addr,v._bolts_bin_qpos_addr
        nc,bc=v._nuts_goal_local_center,v._bolts_goal_local_center
        if case=='inverted': na,ba,nc,bc=ba,na,bc,nc
        if case=='all_in_one_bin': ba,bc=na,nc
        if case=='floating_spec': nc=nc+np.array([0,0,.025]);bc=bc+np.array([0,0,.025])
        put_group(e,v._nut_qpos_addrs,na,nc)
        put_group(e,v._bolt_qpos_addrs,ba,bc)
        if case=='mixed':
            a=v._nut_qpos_addrs[0];e.data.qpos[a:a+3]=to_world(e,ba,bc);mujoco.mj_forward(e.model,e.data)
        rows.append({'case':case,**score(e)})
        np.savez_compressed(OUT/f'{case}.npz',qpos=e.data.qpos,seed=0)
        e.close()
    return rows

def settles():
    rng=np.random.default_rng(960000000);rows=[]
    for seed in range(3):
        e=env(seed);m,d=e.model,e.data;v=e._task_evaluator
        base=d.qpos.copy()
        for kind in ('nut','bolt'):
            addrs=v._nut_qpos_addrs if kind=='nut' else v._bolt_qpos_addrs
            a=int(addrs[0]); j=int(np.where(m.jnt_qposadr==a)[0][0]); bid=int(m.jnt_bodyid[j]);dof=int(m.jnt_dofadr[j])
            binadr=v._nuts_bin_qpos_addr if kind=='nut' else v._bolts_bin_qpos_addr
            center=v._nuts_goal_local_center if kind=='nut' else v._bolts_goal_local_center
            size=v._nuts_goal_size if kind=='nut' else v._bolts_goal_size
            binbid=int(m.jnt_bodyid[np.where(m.jnt_qposadr==binadr)[0][0]])
            for trial in range(30):
                # Fresh integrator/warm-start state; hold both arms at their reset pose rather than driving ctrl=0.
                mujoco.mj_resetData(m,d)
                d.qpos[:]=base;d.qvel[:]=0
                for u in range(m.nu):
                    d.ctrl[u]=d.qpos[m.jnt_qposadr[m.actuator_trnid[u,0]]]
                loc=np.array([rng.uniform(-.12,.12),rng.uniform(-.13,.13),.16])
                d.qpos[a:a+3]=to_world(e,binadr,loc)
                q=rng.normal(size=4);d.qpos[a+3:a+7]=q/np.linalg.norm(q)
                mujoco.mj_forward(m,d)
                for _ in range(650):mujoco.mj_step(m,d)
                qbin=d.qpos[binadr:binadr+7];R=_quat_to_rotmat_batch(qbin[3:][None])[0]
                local=R.T@(d.qpos[a:a+3]-qbin[:3])
                inside=bool(np.all(np.abs(local-center)<=size))
                floor_names=[]
                for c in d.contact:
                    b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                    if bid in (b1,b2):
                        g=int(c.geom2 if b1==bid else c.geom1)
                        floor_names.append(m.geom(g).name)
                touching_bin=any(name and name.startswith(kind+'s_bin_') for name in floor_names)
                support='bin' if touching_bin else 'table_or_other'
                got=v._inside_bin(d.qpos[a:a+3][None,None],d.qpos[None],binadr,center,size)[0,0]
                warnings={str(i):int(w.number) for i,w in enumerate(d.warning) if w.number}
                valid=not warnings and abs(d.time-650*m.opt.timestep)<1e-8
                row={'seed':seed,'kind':kind,'trial':trial,'inside_goal':inside,'accepted':bool(got),'speed':float(np.linalg.norm(d.qvel[dof:dof+3])), 'local':local.tolist(),'support':support,'contacts':floor_names,'valid':valid,'warnings':warnings,'time':float(d.time), 'part_qpos':d.qpos[a:a+7].tolist(),'bin_qpos':qbin.tolist()}
                rows.append(row)
                if (trial<2) or (bool(got) and support!='bin'):
                    np.savez_compressed(OUT/f'settle_{seed}_{kind}_{trial}.npz',qpos=d.qpos,seed=seed)
        print('settled seed',seed,flush=True);e.close()
    return rows

if __name__=='__main__':
    print('abc_sim',abc_sim.__file__,flush=True)
    reset=[];e=env()
    for seed in range(40):
        try:
            e.forget_arm_state();e.reset(seed=seed);r=score(e);v=e._task_evaluator
            reset.append({'seed':seed,**r,'nuts':len(v.nut_names),'bolts':len(v.bolt_names)})
        except Exception as exc: reset.append({'seed':seed,'error':repr(exc)})
    e.close();print('reset',len(reset),flush=True)
    result={'checkout':abc_sim.__file__,'reset':reset,'handmade':handmade()}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    result['settles']=settles()
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    print('written',OUT/'audit.json',flush=True)
