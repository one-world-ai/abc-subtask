# Drawer subtasks (open_drawer / close_drawer) for single-stage RL

Tasks: `open_drawer` (env task `drawer_open`) and `close_drawer` (`drawer_close`) from the OpenWAM overlay. Stage
table approved by the user on 2026-11-14 (G1, all default options). Gate numbers, repairs and evidence paths are in
`STATUS.md`; this file explains how the subtasks work.

## Use

```python
import abc_sim
env = abc_sim.make_env(task="open_drawer__pull", render_cameras=False)
obs, info = env.reset_to_stage(seed)          # builds a start (< 5 s CPU), or pass a saved start row
prompt = info["prompt"]                       # e.g. "pull the bottom drawer open"
ev = env._task_evaluator
done = False
while not done:
    for action in policy(obs, prompt)[:32]:   # one chunk = 32 executed actions
        obs, _, _, _, info = env.step(action)
    reward, done = ev.chunk_reward()
```

- Specs: `open_drawer__{reach,grasp,pull,release}`, `close_drawer__{reach,push,release}` (evaluators
  `open_drawer_subtask` / `close_drawer_subtask`). Fixed spec prompts are lookup keys (`"open drawer subtask: …"`);
  `ev.stage_prompt()` names the episode's drawer (`"grasp the top drawer handle"`); no other task resolves from it.
- Saved starts: `../simulation/outputs/subtask_rewards/drawer/starts/<mode>_<stage>.npz` (seed, target, arm, full
  MuJoCo state, episode start openings as `rest_height`, labels). Seeds: open 980_000_000 + i, close
  985_000_000 + i, i < 100.
- CPU, single world. Contacts are read through 3- and 5-step windows kept in the scene, so mid-episode starts begin
  with an empty window (a start can never be "released" before 5 steps have passed).

## Stages (one target drawer per episode: the one the prompt names; `drawer_stages.py`)

Notation: **bar** = crossbar centre (`*_handle` site + 1.9 cm along the pull axis); **pinch** = finger-pad centre
(grasp site + R·(4.4, 0, −0.5) cm); **hand** = the gripper's palm-to-fingertip segment (grasp-site x 0–6.5 cm);
**q** = the target's slide joint (+ = open, travel 0–20 cm); **others** = the larger q of the other two drawers;
**on handle** = a finger touches the stem/bar in the last 3 control steps; **hand on drawer** = any robot geom touches
the target drawer in the last 3 steps; **released** = no robot contact with the target for the last 5 steps.

open_drawer:
| stage | in it when (= start) | goal (= next stage) | progress / span |
|---|---|---|---|
| reach | none of the below | pinch ≤ 3 cm from the bar, or a finger on the handle with the pinch ≤ 5 cm | −dist(pinch, bar) / 0.45 m |
| grasp | near (exit 5 cm), not gripped | both fingers of one gripper on the handle with the gripper commanded closed (< 0.5); **or** q ≥ 2 cm with a hand on the drawer (hooks count, D2) | 0.25·fingers + 0.5·closure·[both on] / 1.0 |
| pull | gripped, or q ≥ 2 cm with a hand on it; stays while a hand is on it | q ≥ 12 cm and others ≤ 5 cm (stock success) | clip(q / 12 cm) / 1.0 |
| release | stock success (stays while q ≥ 11 cm) | stock success **and** released **and** hand ≥ 5 cm from the bar | min(dist(hand, bar), 5 cm) / 5 cm / 1.0 |

close_drawer:
| stage | in it when (= start) | goal | progress / span |
|---|---|---|---|
| reach | none of the below | pinch ≤ 3 cm from the bar, or a hand on the drawer with the pinch ≤ 5 cm | −dist / 0.45 m |
| push | near (exit 5 cm) or (once pushing) any hand on the drawer | q ≤ 1.5 cm and others ≤ 5 cm (stock success) | (start − q) / (start − 1.5 cm) (stock progress) / 1.0 |
| release | stock success (stays while q ≤ 2.5 cm) | stock success and released and hand ≥ 5 cm from the bar | min(dist(hand, bar), 5 cm) / 5 cm / 1.0 |

Failures (−1, latched per control step, before the generic ones):
- **wrong drawer opened**: others > 5 cm (stock `other_max_m`, D4), every stage of both tasks.
- **pulled further open** (close, D5): q > start opening + 3 cm, every stage. Mostly a reach/push guard; the travel
  ends at 20 cm (+~2 cm soft limit), so from starts above 17 cm it cannot fire.
- **back to an earlier stage** (generic, debounced 5 steps): grasp — the hand moved > 5 cm away; pull — no hand on the
  drawer for 3+5 steps before 12 cm; release — pushed back below 11 cm (open) / pulled back above 2.5 cm (close);
  push — no contact and the pinch > 5 cm away.
No "fallen" label (drawers cannot fall); the kit's FALLEN index is never produced.

Why these thresholds: 12 cm / 1.5 cm / 5 cm are the stock evaluator options (eval set v5) and the pull/push goals
reuse the stock success, so a chained open (reach→grasp→pull) is scored exactly like the eval; the release stage
adds "let go and step back", which the eval does not check. Near 3 cm: the solver grasp puts the pinch 0.4–1.5 cm
from the bar, a fingertip push 2.5–2.9 cm. Contact windows: phase 0 saw a hand–drawer contact at only 2 of 6 step
boundaries in a steady push. Release distance is measured from the whole hand: with an open gripper the bar can sit
at the palm, 5 cm behind the pinch (repair round 2).

## Reward (`stagekit.SubtaskEvaluator.chunk_reward`, unchanged)

+1 at the goal, −1 at a failure (both end the episode). In between, `reward_mode="progress"`:
`0.5 × Δprogress / span[stage]` per chunk, potential-based (one big move = eight small moves to 1e-6, G2 test d).

Worked examples (default mode):

**open reach** (span 0.45 m): pinch 35 cm from the bar → 20 cm: +0.17; → 2 cm: **+1** (solver mean return +1.34).

**open grasp** (span 1.0): open fingers straddling the bar, none touching: 0 → one finger touching: +0.125 →
both touching, command 0.8: +0.125 + 0.5·0.4·0.5 = +0.1 → command < 0.5: **+1**. Closing 4 cm in front of the
bar: 0 (no contact, no terminal).

**open pull** (span 1.0): q 0 → 6 cm: +0.25 → 10 cm: +0.17 → 12 cm: **+1** (mean solver return +1.48: 0.5 of
shaping plus the +1). Let go at 4 cm and move away: **−1** (back to grasp). Hold at 10 cm: +0.42, no end.

**open release** (span 1.0): hand on the bar (dist ~1.5 cm, progress 0.3) → open, back 10 cm: progress 1, then 5
steps without contact: **+1**. Push the drawer back below 11 cm: **−1**.

**close push**: start 16 cm open; q 16 → 11 cm: +0.17 → 1.5 cm: **+1**. Stop at 30 %: +0.16, no end. Push some, then
grab and pull it out again: **−1** (back below / past the band) in 19/19.

## Starts (`drawer_starts.py`)

`build_start(env, seed, mode, stage)`: `reset(seed)`, set the slide q directly (no velocity; the drawer is
quasi-static: damping 5, no spring), IK-teleport a gripper to a pose relative to the bar, hold it 6 actions, check
the stage label. A pose is kept only if (a) it touches nothing but the target drawer (the bottom handle is 4 cm
above the table: the gripper opening drops 1.0 → 0.6 → 0.4 until the lower finger clears it), and (b) the stage's
own motion from it has IK (pull to 15 cm, back off 10 cm, push to closed; wrist pitch ≤ 20°), so a start is not a
kinematic dead end.

| stage | setup |
|---|---|
| open reach, close reach | the reset scene |
| open grasp | open gripper, pads above/below the bar, approach 15/30/22° below horizontal, pinch 0.4–1.5 cm in front, ±5 mm sideways; no contact |
| open pull | as grasp, then close for 20 actions: both fingers on the handle, q < 1 cm |
| open release | q ~ U(12.5, 18) cm, then gripped as pull |
| close push | closed fingertips (stacked vertically) on the bar's front face, 30/22/40/15°, at the reset opening (14–19 cm) |
| close release | q ~ U(0, 1.4) cm, fingertips on the bar face |

Closing the fingers vertically for close is deliberate: the overlay's horizontal closing (wrist rolled 90°) has no
IK once the drawer is within ~8 cm of closed, which is why the overlay close solver stalled on 7/7 bottom drawers.
Hook starts for pull were tried and dropped (a static hook loses contact; 10/12 classify as grasp).

## Checking the reward (phase 4, `../simulation/experiments/subtask_rewards/drawer/scripted_rollouts.py`)

Scripted behaviours run from the saved starts like an RL loop (physical joint actions, `chunk_reward()` every 32
actions, stop at the first terminal, then up to 4 still chunks). Motions are Cartesian pinch moves with per-action
IK; an endpoint with no IK is "unplanned" (counts as a solver miss). Results: see `STATUS.md` (filled below).

| stage | status | starts /100 | solver +1 | hacks (as EXPECT among planned) | margin |
|---|---|---|---|---|---|
| open reach | PASSED | 100 | 20/20 (each target 20/20) | idle, move away, wrong handle: 0; open the wrong drawer: −1 18/18 | 1.02 |
| open grasp | PASSED | 100 | 20/20 | idle, close 4 cm in front: 0; hook (alt): +1 13/13 | 1.00 |
| open pull | STOPPED at G5 | 100 | 19/20 | hold at 10 cm: 0; let go early: −1; **other arm opens a neighbour: 3/11** (arms collide) | 1.01 |
| open release | PASSED | 88 | 20/20 | keep holding: 0; push back in: −1 19/20 | 1.00 |
| close reach | PASSED | 100 | 18/20 (18/18 planned) | idle, move away: 0; grab and pull (alt): +1; open the far drawer: −1 15/15 | 1.14 |
| close push | STOPPED at G5 | 87 | 20/20 | stop at 30 %: 0; push then pull open: −1 19/19; **other arm opens a neighbour: 0/15** (knocks the target) | 1.10 |
| close release | PASSED | 100 | 20/20 | stay: 0; pull open while leaving: −1 18/20 | 1.00 |

Per target, every solver earns +1 on ≥ 15/20 starts of each drawer (bottom included). The two stops are about the
wrong-drawer hack not being executable with two arms on stacked handles; the reward showed no fault (details and the
decision left to the user in `STATUS.md`).

Things the checks taught (kept in the code):
- A contact rule needs a distance bound: an open finger of a gripper at the neighbouring handle grazed the target bar
  with the pinch 6.8 cm away and counted as "reached".
- "Away" must be measured from the whole hand: with an open gripper the bar slides to the palm, 5 cm behind the pinch.
- Starts must be completable: a grasp pose whose pull to 15 cm has no IK made the pull solver 0/20 plannable.
- Near the table the gripper cannot open fully (bottom handle 4 cm above it).

## Codemap

| file | what |
|---|---|
| `abc_sim/task_eval/drawer_stages.py` | `StageConfig` (thresholds + reasons), `DrawerScene` (slide q, bar/pinch/hand geometry, contact windows), `classify_open/close`, `progress_open/close`, prompts |
| `abc_sim/task_eval/drawer_subtask.py` | `DrawerOpenSubtaskEvaluator`, `DrawerCloseSubtaskEvaluator` (kit `SubtaskEvaluator`; target from randomizer metadata; wrong-drawer / pulled-past failures; stock `DrawerOpenCloseEvaluator` as `stock_success`) |
| `abc_sim/task_eval/drawer_starts.py` | `build_start`, handle poses, `ik_pinch`, reachability checks, `idle_action` |
| `abc_sim/task_specs.py`, `abc_sim/task_eval/registry.py` | the 7 specs and 2 evaluators (own blocks) |
| `tests/test_drawer_subtasks.py` | G2 tests (plain script) |
| `../simulation/experiments/subtask_rewards/drawer/build_starts.py` | 100 starts per stage + 10-chunk idle check (G3) |
| `../simulation/experiments/subtask_rewards/drawer/scripted_rollouts.py`, `print_summary.py` | behaviours, EXPECT, `summary_*.json` (G4/G5) |
| `subtask_rewards/drawer/probes/` | phase-0 scorer audit probes |
| `subtask_rewards/drawer/tools/` | snapshot of the three tooling scripts at the final run (2026-11-14) |

## Open questions
- Hooks: a hook opening counts for the grasp and pull goals (D2), but no pull starts are hooks (not stable at rest).
- Close reach: reaching any drawer next to an open target passes the target's front (that is the reach goal state);
  only the far drawer can be opened without first scoring the reach goal.
- Not checked: batched/MJWarp evaluation; renders and a Space tab (CPU rendering works, publishing needs the org
  token).
