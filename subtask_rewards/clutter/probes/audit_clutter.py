"""Reproducible CPU scorer audit. Run from the ABC checkout with PYTHONPATH=$PWD.

Writes audit.json plus snapshots.npz; no model/policy inference. Classifies settles
from target contacts and the target origin in the live box frame, independently
of the scorer's goal site. This does not certify arbitrary mesh containment.
"""
import argparse
import json
from pathlib import Path

import abc_sim
import mujoco
import numpy as np

TASK = 'grab_specific_object_from_clutter'

def score(env):
    r = env.evaluate_task()
    return {'reward': float(np.ravel(r.reward)[0]), 'success': bool(np.ravel(r.success)[0])}

def target_local(env):
    ev = env._task_evaluator
    box = env.model.body('grab_clutter_target_box').id
    return env.data.xmat[box].reshape(3, 3).T @ (env.data.qpos[ev._target_qpos_addr:ev._target_qpos_addr + 3] - env.data.xpos[box])

def target_contacts(env):
    target = env.model.body('target_object').id
    seen = set()
    for c in env.data.contact:
        a, b = int(env.model.geom_bodyid[c.geom1]), int(env.model.geom_bodyid[c.geom2])
        if a == target or b == target:
            other = int(c.geom2 if a == target else c.geom1)
            seen.add(env.model.geom(other).name)
    return sorted(seen)

def classify(env):
    local = target_local(env)
    contacts = target_contacts(env)
    on_box = any(x.startswith('grab_clutter_target_box_') for x in contacts)
    on_table = any(x in ('table_plane', 'floor_collision') for x in contacts)
    v = float(np.linalg.norm(env.data.qvel[env.model.jnt_dofadr[env.model.joint('target_object_joint').id]:][:6]))
    if on_box and abs(local[0]) < .07 and abs(local[1]) < .065 and 0 < local[2] < .104:
        kind = 'inside_supported'
    elif on_box and local[2] >= .104:
        kind = 'rim_supported'
    elif on_box:
        kind = 'outside_box_supported'
    elif on_table:
        kind = 'table_supported'
    else:
        kind = 'no_fixture_contact'
    return kind, local.tolist(), contacts, v

def reset(env, seed):
    env.forget_arm_state()
    _, info = env.reset(seed=seed)
    return info['randomization'].metadata

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resets', type=int, default=40)
    p.add_argument('--settles-per-seed', type=int, default=50)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--out', type=Path, default=Path('subtask_rewards/clutter/probes'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = abc_sim.make_env(task=TASK, render_cameras=False)
    result = {'checkout': abc_sim.__file__, 'resets': [], 'poses': [], 'settles': []}
    states = []
    for seed in range(args.resets):
        try:
            meta = reset(env, seed)
            ev = env._task_evaluator
            t = env.data.qpos[ev._target_qpos_addr:ev._target_qpos_addr+3]
            b = env.data.qpos[ev._box_qpos_addr:ev._box_qpos_addr+3]
            row = {'seed': seed, **score(env), 'category': meta['target_category'], 'variant': meta['target_variant'],
                   'objects': len(meta['objects']), 'target_y': float(t[1]), 'box_y': float(b[1]),
                   'same_side': bool(t[1]*b[1]>0)}
        except Exception as ex:
            row = {'seed': seed, 'error': repr(ex)}
        result['resets'].append(row)
        print('reset', row, flush=True)
    # Set target origin in live box frame. Includes points well above/below physical interior.
    for seed in (0, 1, 2):
        meta = reset(env, seed)
        m, d, ev = env.model, env.data, env._task_evaluator
        original_q = d.qpos.copy()
        original_v = d.qvel.copy()
        box_id = m.body('grab_clutter_target_box').id
        for name, local in [('center_float', [0,0,.20]), ('above_rim_float', [0,0,.28]),
                            ('below_floor', [0,0,-.04]), ('outside_above', [0,0,.40]),
                            ('table_beside', [.14,0,.04]), ('origin_on_rim', [0,.07,.12])]:
            d.qpos[:] = original_q; d.qvel[:] = original_v
            rot = d.xmat[box_id].reshape(3,3)
            d.qpos[ev._target_qpos_addr:ev._target_qpos_addr+3] = d.xpos[box_id] + rot @ np.array(local)
            d.qvel[:] = 0; mujoco.mj_forward(m,d)
            row = {'seed': seed, 'pose': name, 'category':meta['target_category'], 'local_origin':target_local(env).tolist(),
                   'contacts':target_contacts(env), **score(env)}
            result['poses'].append(row); print('pose',row,flush=True)
            states.append((f'{seed}_{name}', d.qpos.copy()))
        # Distractor alone inside must not fulfill target success.
        d.qpos[:] = original_q; d.qvel[:] = original_v; mujoco.mj_forward(m,d)
        j = m.joint('clutter_obj_01_joint').id; a=int(m.jnt_qposadr[j])
        d.qpos[a:a+3] = d.xpos[box_id] + d.xmat[box_id].reshape(3,3) @ np.array([0,0,.05])
        mujoco.mj_forward(m,d)
        row={'seed':seed,'pose':'wrong_object_in_box',**score(env)}
        result['poses'].append(row);print('pose',row,flush=True)
        # Tip the box, then put target in its moving goal. No support is required by the stock rule.
        d.qpos[:] = original_q; d.qvel[:] = 0
        d.qpos[ev._box_qpos_addr+3:ev._box_qpos_addr+7]=[np.sqrt(.5),np.sqrt(.5),0,0]
        mujoco.mj_forward(m,d)
        d.qpos[ev._target_qpos_addr:ev._target_qpos_addr+3]=d.xpos[box_id]+d.xmat[box_id].reshape(3,3)@np.array([0,0,.11])
        mujoco.mj_forward(m,d)
        row={'seed':seed,'pose':'tipped_box_goal_float',**score(env)}
        result['poses'].append(row); print('pose',row,flush=True)
        # Independent settled trials; restore before every drop so layout drift cannot accumulate.
        reset(env,seed); m,d,ev=env.model,env.data,env._task_evaluator
        original_q=d.qpos.copy(); box_id=m.body('grab_clutter_target_box').id
        rng=np.random.default_rng(930_000_000+seed)
        for i in range(args.settles_per_seed):
            d.qpos[:]=original_q; d.qvel[:]=0
            d.ctrl[env._ctrl_indices] = d.qpos[env._qpos_indices]
            mujoco.mj_forward(m,d)
            local = np.array([rng.uniform(-.13,.13),rng.uniform(-.12,.12),rng.uniform(.17,.27)])
            d.qpos[ev._target_qpos_addr:ev._target_qpos_addr+3]=d.xpos[box_id]+d.xmat[box_id].reshape(3,3)@local
            quat = rng.normal(size=4); quat/=np.linalg.norm(quat)
            d.qpos[ev._target_qpos_addr+3:ev._target_qpos_addr+7]=quat
            mujoco.mj_forward(m,d)
            for _ in range(args.steps): mujoco.mj_step(m,d)
            kind,loc,contacts,speed=classify(env)
            row={'seed':seed,'trial':i,'category':meta['target_category'],'class':kind,'local_origin':loc,
                 'contacts':contacts,'qvel_norm':speed,**score(env)}
            result['settles'].append(row)
            if i<2: print('settle',row,flush=True)
        print('settles done seed',seed,flush=True)
    result['summary']={
        'reset_tries':len(result['resets']),
        'reset_errors':sum('error' in r for r in result['resets']),
        'reset_nonzero_reward':sum(r.get('reward',0)>0 for r in result['resets']),
        'reset_success':sum(r.get('success',False) for r in result['resets']),
        'same_side':sum(r.get('same_side',False) for r in result['resets']),
        'settle_classes':{},
    }
    for r in result['settles']:
        key=f"{r['class']}|success={r['success']}"
        result['summary']['settle_classes'][key]=result['summary']['settle_classes'].get(key,0)+1
    (args.out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    # Variable-length qpos by model seed -> save separately.
    np.savez_compressed(args.out/'poses.npz',**dict(states))
    print(json.dumps(result['summary'],indent=2),flush=True)
    env.close()
if __name__=='__main__': main()
