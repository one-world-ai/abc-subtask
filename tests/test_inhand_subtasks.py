"""Inhand transfer subtask reward contract (gate G2): starts classify, idle is neutral, every listed failure fires,
shaping is potential-based, snapshot restore is exact. CPU, no renderer. Plain script: prints ok/FAIL per test."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "disable")
import traceback

import mujoco
import numpy as np

import abc_sim
from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.inhand_stages import C, DONE, FALLEN, H, P, PL, PR, R, STAGES, StageConfig, classify
from abc_sim.task_eval.inhand_starts import SEED_BASE, build_start, current_action
from abc_sim.task_specs import get_task_spec

TRAINABLE = STAGES[:-1]
_ENVS: dict = {}
_STARTS: dict = {}


def env_for(stage):
    if stage not in _ENVS:
        _ENVS[stage] = abc_sim.make_env(task=f"inhand_transfer_item_to_other_side__{stage}", render_cameras=False)
    return _ENVS[stage]


def starts(stage, n=3, max_seeds=40):
    """The first n starts built on SEED_BASE + i (cached)."""
    if stage not in _STARTS:
        env, got = env_for(stage), []
        for i in range(max_seeds):
            s, _ = build_start(env, SEED_BASE + i, stage)
            if s is not None:
                got.append(s)
            if len(got) == n:
                break
        _STARTS[stage] = got
    return _STARTS[stage]


def begin(stage, k=0):
    env = env_for(stage)
    s = starts(stage)[k]
    env.reset_to_stage(s["seed"], s)
    return env, env._task_evaluator, s


def updates(env, n=6):
    for _ in range(n):
        env.evaluate_task()
    return env._task_evaluator.chunk_reward()


def fake_obs(**kw):
    o = dict(com=np.array([[0.55, 0.0, 0.9]]), lift=np.array([0.1]), f_give=np.array([2]), f_recv=np.array([0]),
             touch_give=np.array([True]), touch_recv=np.array([False]), gap_give=np.array([0.0]),
             gap_recv=np.array([0.3]), env_touch=np.array([False]), speed=np.array([0.0]), ys=np.array([0.0]),
             on_table=np.array([True]), stock_done=np.array([False]), com_above_table=np.array([0.15]),
             far_gap=np.array([0.13]), m_gap=np.array([0.0]), low=np.array([0.1]), xy=np.array([[0.55, 0.0]]),
             open_give=np.array([1.0]))
    o.update({k: np.atleast_1d(np.asarray(v)) for k, v in kw.items()})
    return o


# ---------------------------------------------------------------------------------------------------- lookups

def test_lookup_preserves_stock():
    assert get_task_spec("inhand_transfer_item_to_other_side").evaluator_name == "inhand_transfer_other_side"
    for stage in TRAINABLE:
        spec = get_task_spec(f"inhand_transfer_item_to_other_side__{stage}")
        assert get_task_spec(spec.prompt).name == spec.name
        assert spec.evaluator_kwargs() == {"stage": stage}
        assert "env.reset_to_stage" in spec.description


# ---------------------------------------------------------------------------------------------------- handoff (task-independent)

def test_handoff_stage_tests_and_progress():
    cfg = hs.HandoffConfig()
    a_only = fake_obs()
    assert hs.handing(a_only, np.array([False]), cfg)[0] and not hs.received(a_only, cfg)[0]
    both = fake_obs(f_recv=2, touch_recv=True)
    assert hs.handing(both, np.array([True]), cfg)[0] and not hs.received(both, cfg)[0]
    let_go_near = fake_obs(f_give=0, touch_give=False, f_recv=2, touch_recv=True, gap_give=0.02)
    assert not hs.received(let_go_near, cfg)[0], "giver still within clear_m: not received"
    done = fake_obs(f_give=0, touch_give=False, f_recv=2, touch_recv=True, gap_give=0.08)
    assert hs.received_in_air(done, cfg)[0]
    assert not hs.received_in_air(fake_obs(f_give=0, touch_give=False, f_recv=2, touch_recv=True, gap_give=0.08,
                                           env_touch=True), cfg)[0], "touching the table is not in the air"
    p = [float(hs.progress(o, cfg)[0]) for o in (a_only, both, let_go_near, done)]
    assert p == sorted(p) and abs(p[-1] - cfg.span) < 1e-9, p
    outside = fake_obs(com=np.array([[0.55, -0.25, 0.9]]))
    assert not hs.handing(outside, np.array([True]), cfg)[0]
    assert hs.handing(fake_obs(com=np.array([[0.55, -0.18, 0.9]])), np.array([True]), cfg)[0], "exit hysteresis"
    assert hs.failure(fake_obs(touch_give=False, f_give=0), 0, cfg) == "dropped during the handoff"
    assert hs.failure(fake_obs(env_touch=True), 0, cfg) == "touched the table during the handoff"


def test_classify_chain():
    cfg = StageConfig()
    prev = np.array([R])
    assert classify(fake_obs(f_give=0, touch_give=False, gap_give=0.3, lift=0.0, m_gap=0.2,
                             com=np.array([[0.55, 0.25, 0.78]])), prev, cfg)[0] == R
    assert classify(fake_obs(f_give=0, touch_give=False, gap_give=0.02, lift=0.0,
                             com=np.array([[0.55, 0.25, 0.78]])), prev, cfg)[0] == P
    assert classify(fake_obs(f_give=0, touch_give=False, gap_give=0.02, lift=0.0, open_give=0.1,
                             com=np.array([[0.55, 0.25, 0.78]])), prev, cfg)[0] == R, "closed fist is not near"
    assert classify(fake_obs(f_give=1, gap_give=0.02, lift=0.0, open_give=0.1,
                             com=np.array([[0.55, 0.25, 0.78]])), np.array([P]), cfg)[0] == P, "closing keeps near"
    assert classify(fake_obs(com=np.array([[0.55, 0.25, 0.9]])), prev, cfg)[0] == PR
    assert classify(fake_obs(), np.array([PR]), cfg)[0] == H
    assert classify(fake_obs(f_recv=2, touch_recv=True), np.array([H]), cfg)[0] == H, "both hold: still handoff"
    recv = dict(f_give=0, touch_give=False, f_recv=2, touch_recv=True, gap_give=0.1)
    assert classify(fake_obs(**recv), np.array([H]), cfg)[0] == C
    far = dict(recv, ys=-0.2, com=np.array([[0.55, -0.2, 0.85]]))
    assert classify(fake_obs(**far), np.array([C]), cfg)[0] == PL
    assert classify(fake_obs(**dict(far, lift=0.2)), np.array([C]), cfg)[0] == C, "too high to enter place"
    assert classify(fake_obs(**dict(far, stock_done=True)), np.array([PL]), cfg)[0] == DONE
    released_far = dict(far, f_recv=0, touch_recv=False, speed=0.5)
    assert classify(fake_obs(**released_far), np.array([PL]), cfg)[0] == PL, "free item keeps its label"
    assert classify(fake_obs(com_above_table=-0.05), np.array([C]), cfg)[0] == FALLEN


# ---------------------------------------------------------------------------------------------------- starts (a, b, e)

def test_built_starts_classify_idle_and_restore_exactly():
    for stage in TRAINABLE:
        got = starts(stage)
        assert len(got) == 3, (stage, "fewer than 3 starts in 40 seeds", len(got))
        env = env_for(stage)
        for s in got:
            env.reset_to_stage(s["seed"], s)
            ev = env._task_evaluator
            assert env.evaluate_task().metrics["label"] == stage, (stage, s["seed"], ev._last)
            total = 0.0
            for _ in range(10):
                act = current_action(env)
                for _ in range(32):
                    env.step(act, render_obs=False)
                r, done = ev.chunk_reward()
                assert not done, (stage, s["seed"], ev._last)
                total += r
            assert abs(total) < 0.05, (stage, s["seed"], total)
            runs = []
            for _ in range(2):
                env.reset_to_stage(s["seed"], s)
                st0 = np.empty(mujoco.mj_stateSize(env.model, int(mujoco.mjtState.mjSTATE_INTEGRATION)))
                mujoco.mj_getState(env.model, env.data, st0, int(mujoco.mjtState.mjSTATE_INTEGRATION))
                assert np.array_equal(st0, s["state"]), (stage, "restored state differs")
                act = current_action(env)
                for _ in range(32):
                    env.step(act, render_obs=False)
                runs.append(env.data.qpos.copy())
            assert np.array_equal(*runs), stage


# ---------------------------------------------------------------------------------------------------- failures (c)

def test_push_and_off_table_fail_in_reach():
    env, ev, s = begin("reach")
    env.data.qpos[ev.scene.qpos_adr[0]] += 0.05
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True) and ev._failed == "pushed", ev._failed
    env, ev, s = begin("reach")
    a = ev.scene.qpos_adr[0]
    env.data.qpos[a + 2] -= 0.20  # below the tabletop, same xy (moving it sideways is caught as "pushed" first)
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True) and ev._failed == "fell off the table", ev._failed


def test_receiver_touch_fails_in_reach_and_pick():
    for stage in ("reach", "pick"):
        env, ev, s = begin(stage)
        o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ev.tracker.obs.items()}
        o["touch_recv"] = np.array([True])
        assert ev.step_failure(0, o, ev.stage) == "receiver arm touched the item"


def test_closed_gripper_touch_fails_only_in_reach():
    env, ev, s = begin("reach")
    o = dict(ev.tracker.obs)
    hit = {**o, "f_give": np.array([1]), "touch_give": np.array([True])}
    assert ev.step_failure(0, {**hit, "open_give": np.array([0.1])}, R) == "touched the item with a closed gripper"
    assert ev.step_failure(0, {**hit, "open_give": np.array([0.9])}, R) == ""
    env, ev, s = begin("pick")
    o = dict(ev.tracker.obs)
    assert ev.step_failure(0, {**o, "f_give": np.array([2]), "open_give": np.array([0.1])}, P) == ""


def test_pick_retreat_regresses():
    env, ev, s = begin("pick")
    from abc_sim.task_eval.stagekit import ARMS, arm_ids
    ids = arm_ids(env.model, ARMS[s["arm"]])
    env.data.qpos[ids.qadr] = env.get_init_q()[s["arm"] * 7:s["arm"] * 7 + 6]
    mujoco.mj_forward(env.model, env.data)
    assert updates(env) == (-1.0, True) and ev._failed == "back to reach", ev._failed


def test_present_failures():
    env, ev, s = begin("present")
    o = dict(ev.tracker.obs)
    assert ev.step_failure(0, {**o, "touch_give": np.array([False]), "touch_recv": np.array([False]),
                               "lift": np.array([0.1])}, PR) == "dropped"
    assert ev.step_failure(0, {**o, "ys": np.array([-0.12])}, PR) == "giver carried it past the middle"
    # real: open the giver in the air -> dropped
    act = current_action(env)
    act[s["arm"] * 7 + 6] = 1.0
    for _ in range(40):
        env.step(act, render_obs=False)
    assert ev.chunk_reward() == (-1.0, True) and ev._failed in ("dropped", "back to pick"), ev._failed


def test_handoff_failures():
    env, ev, s = begin("handoff")
    act = current_action(env)
    act[s["arm"] * 7 + 6] = 1.0  # giver opens before the receiver holds
    for _ in range(40):
        env.step(act, render_obs=False)
    assert ev.chunk_reward() == (-1.0, True) and ev._failed == "dropped during the handoff", ev._failed
    env, ev, s = begin("handoff")
    o = dict(ev.tracker.obs)
    assert ev.step_failure(0, {**o, "env_touch": np.array([True])}, H) == "touched the table during the handoff"


def test_carry_failures():
    env, ev, s = begin("carry")
    o = dict(ev.tracker.obs)
    assert ev.step_failure(0, {**o, "touch_give": np.array([True])}, C) == "giver touched the item again"
    act = current_action(env)
    act[s["arm"] * 7 + 6] = 1.0  # receiver drops it in the middle
    for _ in range(40):
        env.step(act, render_obs=False)
    assert ev.chunk_reward() == (-1.0, True) and ev._failed == "dropped", ev._failed


def test_place_failures():
    env, ev, s = begin("place")
    o = dict(ev.tracker.obs)
    assert ev.step_failure(0, {**o, "touch_give": np.array([True])}, PL) == "giver touched the item again"
    ev._was_touched = True
    free = {**o, "touch_give": np.array([False]), "touch_recv": np.array([False]), "env_touch": np.array([False])}
    assert ev.step_failure(0, {**free, "low": np.array([0.08])}, PL) == "released too high"
    ev._was_touched = True
    assert ev.step_failure(0, {**free, "low": np.array([0.03])}, PL) == "", "low release allowed"
    ev._was_touched = False
    landed = {**free, "env_touch": np.array([True]), "speed": np.array([0.0]), "stock_done": np.array([False])}
    assert ev.step_failure(0, landed, PL) == "landed outside the far band"
    # real: open the receiver where the start holds it (lowest point usually 3-10 cm up) -> too high or allowed
    env, ev, s = begin("place")
    low = float(ev.tracker.obs["low"][0])
    act = current_action(env)
    act[s["arm"] * 7 + 6] = 1.0
    for _ in range(64):
        env.step(act, render_obs=False)
    r, done = ev.chunk_reward()
    if low > StageConfig().release_low_m + 0.01:
        assert (r, ev._failed) == (-1.0, "released too high"), (low, ev._failed)


def test_holding_low_never_succeeds():
    env, ev, s = begin("place")
    for _ in range(10):
        act = current_action(env)
        for _ in range(32):
            env.step(act, render_obs=False)
        r, done = ev.chunk_reward()
        assert not done and r < 0.5
    assert not ev._success


# ---------------------------------------------------------------------------------------------------- shaping (d)

def test_big_move_equals_eight_small_moves():
    for stage in TRAINABLE:
        env, ev, s = begin(stage)
        initial = ev._phi_chunk
        totals = []
        for count in (1, 8):
            ev._phi_chunk = initial
            total = 0.0
            for value in np.linspace(initial, initial + 0.06, count + 1)[1:]:
                ev._last["progress"] = value
                r, done = ev.chunk_reward()
                assert not done
                total += r
            totals.append(total)
        assert abs(totals[0] - totals[1]) < 1e-6, (stage, totals)


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
