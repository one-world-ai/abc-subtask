import numpy as np, mujoco, abc_sim
from abc_sim.task_eval.stagekit import ObjectScene, hold_object, set_object_pose, T, inv, axis_angle_quat, qmul
env = abc_sim.make_env(task="put_relative", render_cameras=False)
R_DOWN = np.diag([1.0, -1.0, -1.0])      # link6 z -> world -z
def topdown(env, scene, k, yaw, arm, height_above_com=0.0):
    d=env.data; c=d.subtree_com[scene.bodies[k]].copy()
    Rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    Rw=Rz@R_DOWN; p6=c+np.array([0,0,0.1347+height_above_com])
    T6=np.eye(4); T6[:3,:3]=Rw; T6[:3,3]=p6
    return {"arm":arm,"T_obj_link6":inv(scene.object_T(d,k))@T6,"fingers":np.array([0.035,0.035])}
res={"held_in_zone_success":0,"held_tries":0}
for seed in range(1,40):
    try: env.reset(seed=seed)
    except RuntimeError: continue
    ev=env._task_evaluator; g=ev._goal; md=env._last_randomization.metadata
    names=[o["name"] for o in md["objects"]]
    joints=[(n, mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_JOINT,n+"_joint")) for n in names]
    scene=ObjectScene(env.model, joints); k=names.index(g["mover"])
    ref=env.data.qpos[g["reference_adr"]:g["reference_adr"]+3].copy()
    target=ref.copy(); target[:2]+=g["axis"]*0.14; target[2]=g["spawn_z"]+0.012
    a=scene.qpos_adr[k]; quat=env.data.qpos[a+3:a+7].copy()
    arm = 0 if target[1]>0 else 1
    ok=False
    for yaw in np.linspace(0,np.pi,6,endpoint=False):
        set_object_pose(env, scene, k, target, quat)
        ok,info=hold_object(env, scene, k, topdown(env,scene,k,yaw,arm))
        if ok: break
        env.reset(seed=seed); scene=ObjectScene(env.model, joints)
    res["held_tries"]+=1
    if not ok: print(seed, g["mover"], "no hold:", info.get("why")); continue
    r=env.evaluate_task(); suc=bool(np.asarray(r.success).ravel()[0])
    obs=scene.observe(env.data, np.zeros(len(names)))
    res["held_in_zone_success"]+=suc
    print(seed, g["mover"], "held by arm", int(obs["grasp_arm"][k]), "| stock success:", suc, "| along/perp", np.round(r.metrics["mover_along_m"],3), np.round(r.metrics["mover_perp_m"],3))
    if res["held_tries"]>=8: break
print(res)
