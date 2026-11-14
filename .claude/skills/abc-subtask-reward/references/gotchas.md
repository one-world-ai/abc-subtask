# Sim gotchas (ABC on Torch)

- **Physics on CPU**: `MUJOCO_GL=disable`; `make_env(..., render_cameras=False)`; `env.step(a, render_obs=False)`.
  Run on a cs CPU node (128 cores), not a login node, for anything parallel.
- **CPU rendering** for review images works with Mesa: `E=../envs/mesa-cpu`,
  `__EGL_VENDOR_LIBRARY_FILENAMES=$E/share/glvnd/egl_vendor.d/50_mesa.json MUJOCO_GL=egl EGL_PLATFORM=surfaceless
  LD_LIBRARY_PATH=$E/lib`, then `mujoco.Renderer`. Without it, importing mujoco with MUJOCO_GL=egl fails on CPU nodes.
  Paint the object of interest a bright colour (`geom_rgba`, `geom_matid=-1`) — dark meshes are unreadable.
- `reset()` carries the previous arm pose over, which changes the randomized scene → `forget_arm_state()` first
  (done inside `reset_to_snapshot`).
- Scenes with variable object counts swap the MjModel and the evaluator on reset → re-fetch
  `env._task_evaluator` after every reset/restore.
- Task prompts are lookup keys: every subtask prompt must normalize to a string no other task uses.
- Assets are per task: `python prepare.py --sim-task <task>` in the abc checkout before first use.
- Object frames are not object-centred: ShapeNet meshes are bounding-box-centred, so a mug's cup axis is 0.6–2 cm off
  its body origin. Measure geometry (axis, radius, handle) from the collision mesh per model.
- Freejoint qpos order in multi-object scenes follows body order, not names — resolve via the evaluator's
  `*_qpos_addrs`, and `mj_name2id` returns −1 silently for a wrong name (→ index −1 = last body!).
- Bit-exact replay: rebuild from seed, step recorded actions, compare `env.get_state()` with each server-trace call
  state (`drift == 0`). If it drifts, the abc checkout differs from the eval commit — stop and report.
- Don't `pkill -f` with a pattern that also matches your own command.
- **Never run with `python -I`**: it ignores PYTHONPATH and imports `../simulation/abc` (installed in the venv, older
  scorers) instead of the `abc` checkout under test — results look plausible and are wrong. Check the printed
  `abc_sim.__file__`.
