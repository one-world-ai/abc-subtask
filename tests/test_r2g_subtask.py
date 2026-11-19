"""Real2Gym "in a container" stage envs (abc_sim/task_eval/r2g_subtask.py): gate G2. CPU, no renderer.

  (a) a built start classifies as its stage           (b) idle earns nothing and does not end the episode
  (c) every listed failure gives -1                    (d) shaping does not depend on how the motion is chunked
  (e) a saved start restores exactly                   (f) stages chain in place and the place goal is the obj check
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("NO_RENDER", "1")
import sys
import traceback
from pathlib import Path

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval.bottles_starts import idle_action, move_link
from abc_sim.task_eval.r2g_starts import SEED_BASE, over_container_pose
from abc_sim.task_eval.r2g_subtask import R2G_STAGE_TASKS
from abc_sim.task_eval.stagekit import ARMS, STATE_SPEC, T, arm_ids, set_object_pose
from abc_sim.task_specs import get_task_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "subtask_rewards/r2g/tools"))

TASK = "put_the_screwdriver_in_the_bin"
STAGES = ("reach", "pick", "carry", "place")
_ENVS: dict = {}
_STARTS: dict = {}


def env_for(stage, task=TASK):
    key = (task, stage)
    if key not in _ENVS:
        _ENVS[key] = abc_sim.make_env(task=f"r2g_{task}__{stage}", render_cameras=False)
    return _ENVS[key]


def start_for(stage, task=TASK, i=0):
    key = (task, stage, i)
    if key not in _STARTS:
        env = env_for(stage, task)
        for j in range(i, i + 12):
            start, log = env.subtask.build_start(env, SEED_BASE + j)
            if start is not None:
                break
        assert start is not None, log
        _STARTS[key] = start
    return _STARTS[key]


def begin(stage, task=TASK, i=0):
    env, start = env_for(stage, task), start_for(stage, task, i)
    env.reset_to_stage(start["seed"], start)
    return env, env.subtask, start


def run_chunks(env, act, n=1, per=32):
    total, done = 0.0, False
    for _ in range(n):
        for _ in range(per):
            env.step(act, render_obs=False)
        r, done = env.subtask.chunk_reward()
        total += r
        if done:
            break
    return total, done


def link6(env, arm):
    ids = arm_ids(env.model, ARMS[arm])
    return T(env.data.xpos[ids.link6], env.data.xquat[ids.link6])


def test_registered_for_every_task():
    for task in R2G_STAGE_TASKS:
        for stage in STAGES:
            spec = get_task_spec(f"r2g_{task}__{stage}")
            assert spec.env_task == f"r2g_{task}" and spec.evaluator_kwargs()["stage"] == stage, spec


def test_starts_classify_restore_and_idle():
    for stage in STAGES:
        env, ev, start = begin(stage)
        k = start["target"]
        assert ev.LABELS[int(start["labels"][k])] == stage, (stage, start["labels"])
        back = np.zeros(mujoco.mj_stateSize(env.model, STATE_SPEC))
        mujoco.mj_getState(env.model, env.data, back, STATE_SPEC)
        assert np.array_equal(back, start["state"]), stage                       # (e)
        assert ev.stage_prompt() and "screwdriver" in ev.stage_prompt(), ev.stage_prompt()
        total, done = run_chunks(env, idle_action(env), n=10)                     # (b)
        assert not done and abs(total) < 0.05 and ev.last["label"] == stage, (stage, total, done, ev.failure, ev.last)


def test_shaping_does_not_depend_on_chunking():
    totals = []
    for per in (32, 4):  # the same 64-action motion scored in 2 chunks and in 16
        env, ev, start = begin("carry")
        arm = start["arm"]
        goal = link6(env, arm)
        goal[:3, 3] += [0.0, 0.03 if arm else -0.03, 0.0]
        ids = arm_ids(env.model, ARMS[arm])
        act = idle_action(env)
        act[arm * 7 + 6] = 0.0
        q0 = env.data.qpos[ids.qadr].copy()
        from abc_sim.task_eval.stagekit import ik_link6
        q1, pe, _ = ik_link6(env.model, env.data, ARMS[arm], goal)
        assert q1 is not None and pe < 5e-3, pe
        total, phi0 = 0.0, ev.last.get("phi")
        for i in range(64):
            act[arm * 7:arm * 7 + 6] = q0 + (q1 - q0) * (i + 1) / 64
            env.step(act, render_obs=False)
            if (i + 1) % per == 0:
                r, done = ev.chunk_reward()
                assert not done, ev.failure
                total += r
        totals.append(total)
    assert abs(totals[0] - totals[1]) < 1e-6, totals                           # (d)


def test_failures_fire():
    # pushed: the resting target moved 10 cm without a grasp
    env, ev, start = begin("reach")
    sc, k = ev.scene, start["target"]
    a = int(sc.qpos_adr[k])
    pos = env.data.qpos[a:a + 3].copy()
    away = pos[:2] - env.data.xpos[sc.cont_body[0]][:2]  # away from the bin: a teleport into it is a contact fault
    pos[:2] += 0.10 * away / np.linalg.norm(away)
    set_object_pose(env, sc, k, pos, env.data.qpos[a + 3:a + 7].copy())
    total, done = run_chunks(env, idle_action(env), n=3)
    assert done and ev.failure == "pushed" and total <= -1.0 + 0.5, (ev.failure, total)
    # off the table
    env, ev, start = begin("reach")
    sc, k = ev.scene, start["target"]
    a = int(sc.qpos_adr[k])
    set_object_pose(env, sc, k, [0.2, 0.9, 0.3], env.data.qpos[a + 3:a + 7].copy())
    total, done = run_chunks(env, idle_action(env), n=3)
    assert done and ev.failure and total < 0, (ev.failure, total)
    # container moved
    env, ev, start = begin("pick")
    f = ev.scene.full
    c = ev.scene.containers[0]
    j = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f._objs[c]["free_joint"])
    adr = int(env.model.jnt_qposadr[j])
    env.data.qpos[adr] += 0.08
    mujoco.mj_forward(env.model, env.data)
    total, done = run_chunks(env, idle_action(env), n=3)
    assert done and ev.failure == "container moved or tipped", (ev.failure, total)
    # opened while carrying: released outside the container
    env, ev, start = begin("carry")
    act = idle_action(env)
    act[start["arm"] * 7 + 6] = 1.0
    total, done = run_chunks(env, act, n=4)
    assert done and total < 0 and ev.failure, (ev.failure, total)


def test_chain_reaches_the_obj_check():
    """reach start -> scripted reach, pick, carry, place with env.advance_stage in between; the last +1 is the
    obj env's own success."""
    import scripted_rollouts as S

    done_any = False
    starts = [start_for("reach", i=i) for i in range(8)]  # before any advance_stage: build_start builds env.subtask's stage
    env = env_for("reach")
    for start in starts:
        env.reset_to_stage(start["seed"], start)  # also after an earlier episode advanced this env to a later stage
        assert env.subtask.stage_name == "reach"
        ev, sc = env.subtask, env.subtask.scene
        k, arm = int(ev.target), int(sc.arm[ev.target])
        stages = []
        try:
            if not S.reach_with(env, sc, k, arm):
                continue
            run_chunks(env, idle_action(env), n=2)
            if not env.subtask.success:
                continue
            stages.append("reach")
            env.advance_stage()
            S.hold(env, 25, arm, 0.0)
            S.go(env, arm, S.shifted(S.link6(env, arm), [0, 0, 0.08]), 0.0, 64)
            act = idle_action(env); act[arm * 7 + 6] = 0.0
            run_chunks(env, act, n=2)
            if not env.subtask.success:
                continue
            stages.append("pick")
            info = env.advance_stage()
            assert info["subtask"] == "carry", info
            goal, _ = over_container_pose(env, env.subtask.scene, k, arm, np.random.default_rng(0))
            if goal is None or not S.go(env, arm, goal, 0.0, 96):
                continue
            act = idle_action(env); act[arm * 7 + 6] = 0.0
            run_chunks(env, act, n=2)
            if not env.subtask.success:
                continue
            stages.append("carry")
            info = env.advance_stage()
            assert info["subtask"] == "place", info
            assert env.subtask.last.get("pinch_s", 0) >= 0.2 or env.subtask.tracker.obs["pinch_s"] >= 0.2
            act = idle_action(env); act[arm * 7 + 6] = 1.0
            for _ in range(8):
                total, done = run_chunks(env, act, n=1)
                if done:
                    break
            if env.subtask.success:
                assert env.subtask.last["stock_ever_success"], env.subtask.last  # the obj env's own check passed
                done_any = True
                break
        finally:
            print("  chain", start["seed"], stages, flush=True)
    assert done_any, "no seed chained reach -> place with the scripted solver"


def main():
    tests = [fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("ok", fn.__name__, flush=True)
        except Exception:
            failed += 1
            traceback.print_exc()
            print("FAIL", fn.__name__, flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed", flush=True)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
