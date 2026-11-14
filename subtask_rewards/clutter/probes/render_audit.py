"""CPU Mesa renders of representative clutter scorer probes."""
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import abc_sim

out=Path(__file__).parent/'renders';out.mkdir(exist_ok=True)
env=abc_sim.make_env(task='grab_specific_object_from_clutter',render_cameras=False)
images=[]
for seed,kind in [(0,'above_rim_float'),(0,'below_floor'),(0,'settled_inside'),(1,'settled_inside')]:
    env.forget_arm_state();env.reset(seed=seed)
    m,d,ev=env.model,env.data,env._task_evaluator
    box=m.body('grab_clutter_target_box').id
    local=[0,0,{'above_rim_float':.28,'below_floor':-.04,'settled_inside':.08}[kind]]
    d.qpos[ev._target_qpos_addr:ev._target_qpos_addr+3]=d.xpos[box]+d.xmat[box].reshape(3,3)@local
    d.qvel[:]=0;mujoco.mj_forward(m,d)
    if kind=='settled_inside':
        d.ctrl[env._ctrl_indices]=d.qpos[env._qpos_indices]
        for _ in range(1500):mujoco.mj_step(m,d)
    target=m.body('target_object').id
    for g in range(m.ngeom):
        if m.geom_bodyid[g]==target:
            m.geom_rgba[g]=[1,.02,.55,1];m.geom_matid[g]=-1
    cam=mujoco.MjvCamera();cam.lookat[:]=d.xpos[box]+np.array([0,0,.12]);cam.distance=.60
    cam.azimuth=105 if d.xpos[box,1]>0 else -105;cam.elevation=-45
    renderer=mujoco.Renderer(m,480,640);renderer.update_scene(d,camera=cam)
    im=Image.fromarray(renderer.render()).convert('RGB');renderer.close()
    result=env.evaluate_task();success=bool(result.success[0])
    draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,38),fill='white')
    draw.text((8,6),f'seed {seed}  {kind}  repaired success={success}',fill='black')
    path=out/f'{seed}_{kind}.png';im.save(path);images.append(im)
contact=Image.new('RGB',(1280,960),'white')
for i,im in enumerate(images):contact.paste(im,((i%2)*640,(i//2)*480))
contact.save(out/'contact_sheet.png');env.close()
