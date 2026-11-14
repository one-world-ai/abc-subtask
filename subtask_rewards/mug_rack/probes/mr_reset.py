"""Reset probe for the current scorer: N seeds -> reward 0 / success False; also times the per-reset handle+hole
measurement (configure_from_randomization re-measures every dealt mug).

    MUJOCO_GL=disable OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD ../simulation/.venv/bin/python \
        subtask_rewards/mug_rack/probes/mr_reset.py --seeds 40
"""
import argparse, os, time
os.environ.setdefault("MUJOCO_GL", "disable")
import numpy as np  # noqa: E402
import abc_sim  # noqa: E402
print(f"abc_sim from {abc_sim.__file__}", flush=True)
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=40); a = ap.parse_args()
env = abc_sim.make_env(task="hang_mug_on_mug_rack", render_cameras=False)
bad, meas = [], []
for s in range(a.seeds):
    env.reset(seed=s)
    r = env.evaluate_task()
    if float(np.asarray(r.reward).ravel()[0]) != 0 or bool(np.asarray(r.success).ravel()[0]):
        bad.append(s)
    t = time.time(); env._task_evaluator._measure_mug_handles(env.model); [env._task_evaluator._hole(i) for i in range(len(env._task_evaluator.mug_names))]; meas.append(time.time() - t)
print(f"resets {a.seeds}: nonzero/success {bad}; handle measurement + (lazy) hole of every mug {np.mean(meas):.3f} s (max {max(meas):.3f})")
