# Clutter subtasks for single-stage RL: design + codemap

Task `grab_specific_object_from_clutter` (env task `grab_clutter`): a prompted target among 8–16 household
objects must go into a small open box (interior 14 × 13 cm, rim 10.4 cm) placed near a table side.
Ledger with every number and evidence path: [STATUS.md](STATUS.md). Approved table: [stage_table.md](stage_table.md).

## Quick start
```python
import abc_sim, numpy as np
env = abc_sim.make_env(task="grab_specific_object_from_clutter__place", render_cameras=True)  # __reach/__pick/__handoff/__carry
z = np.load("../simulation/outputs/subtask_rewards/clutter/handoff_r3/starts/place.npz", allow_pickle=True)
start = dict(seed=int(z["seed"][0]), target=0, arm=int(z["arm"][0]), state=z["state"][0],
             rest_height=z["rest_height"][0], labels=z["labels"][0])
obs, info = env.reset_to_stage(start["seed"], start)   # exact restore; info["prompt"] names the target
ev = env._task_evaluator
while not done:
    for a in policy(obs, info["prompt"])[:32]:
        obs, *_ = env.step(a)
    reward, done = ev.chunk_reward()
```
`reset_to_stage(seed)` without a start builds one (~1–10 s). Scoring needs real contacts and velocities:
`evaluate_qpos_batch` raises on purpose.

## Stages (`clutter_stages.py`)
Target = the prompted object (`target_object`, index 0) for every stage; it is never replaced by a distractor.

| stage | prompt (spec / live) | start = precondition | goal = next precondition |
|---|---|---|---|
| reach | approach the prompted target / move a gripper to the {label} in the clutter | fresh reset | grasp site ≤ 5 cm from target CoM or a finger touching it, **with the target undisturbed** (speed ≤ 5 mm/s, moved < 3 cm) |
| pick | grasp and lift … / grasp the {label} and lift it from the clutter | open gripper around the target | both fingers of one gripper on it, lifted ≥ 3 cm |
| carry | carry … over the box / carry the {label} over the box | held, lifted | target origin inside the scorer's box-local XY goal, lowest collision point ≥ rim + 1 cm, still held |
| place | lower … and release / place the {label} inside the box and let go | held above the opening | repaired stock success **and** no finger **and** at rest **and** floor-support contact |

Hysteresis: near exit 7 cm; opening exit 1 cm beyond the XY goal. Debounce 5 actions. A contact-free moving target
keeps its previous label (no terminal mid-flight).

Why the reach entry rule: in round 1 a closed gripper shoving the target scored +1 (contact = reach). Entering
pick now needs the target still: resting targets read 0.0 mm/s at all 14 reach-solver goals, a slow shove 14–15 mm/s.
Later stages keep plain `near`, so a grasp squeezing the target does not regress.

## Handoff (amendment 2026-11-14; `handoff_stage.py` reused unchanged)
Chain: reach → pick → **handoff** → carry → place. The box (y ≈ ±0.515) is reachable only by the arm on its side
(bases at y = ±0.31): in the direct pilot 53/53 scenes picked by the other arm had no carry path. So in every scene
**giver = non-box arm, receiver = box arm**; a scene needs a handoff iff its picking arm (nearest at reset) is the
giver (53/100). Labels (state only, `classify`):
- `handoff`: the giver holds it with both fingers, lifted ≥ 3 cm (entering: touching nothing but the robot), or
  `handoff_stage.handing` (in M: x 0.45–0.70, |y| ≤ 0.10, CoM z 0.83–1.15, exit 10 cm; a robot on it; giver holds or
  receiver ≥ 1 finger).
- `carry` from handoff only via `handoff_stage.received_in_air` (receiver both fingers, giver off and its pinch centre
  ≥ 6 cm from the surface, lifted, touching nothing else); from pick (box arm picked) as before. In carry the giver
  touching it again → back to handoff (−1); `place` needs no giver contact.
- Progress: `handoff_stage.progress` (0–1.25) − min(CoM distance to M, 0.5), span 1.75. Failures:
  `handoff_stage.failure` (dropped; touched table/box/clutter) + the scene failures.
- Observation: `touch_give/touch_recv` = any arm geom in contact or a finger ≤ 2 mm away (a closed receiver pinch
  loses contact for single physics steps); pinch centre link_6 (0, −0.031, 0.125).
Worked example (solver): start 0 − 0.25 (CoM 25 cm from M) → in M: 0 → receiver closes 0.5 → giver opens 1.0 →
giver clear 1.25: shaping 0.5·1.5/1.75 = 0.43, then +1 (measured mean 0.90 including 6/20 misses). both_hold reaches
0.5 → ≤ 0.29, never +1 (measured 0.15). set_down_regrasp: −1 the moment it touches the clutter.

Starts: `handoff` — giver holds the target lifted 7–12 cm above its reset spot (sampled grasp, creep check), or a
pick start + scripted lift. `carry` in handoff scenes — that start + the scripted handoff (`plan_handoff`: a
presentation pose in M with a verified receiver grasp; `exchange`: receiver pre-grasp, close, monitored giver release,
giver backs out), else the receiver holding it alone in M (source 3); both need a carry path to the box (`carry_plan`,
now also with hand + target yawed). `place` fallback: a carry start + the scripted carry.

## Reward (`ClutterSubtaskEvaluator.chunk_reward`, kit `SubtaskEvaluator`)
+1 goal, −1 failure (latched per physics step, reported at the next chunk), else 0.5·Δprogress/span.

Failures (all stages): box moved > 3 cm or tilted > 20°; a distractor off the table; a distractor lifted ≥ 3 cm
(wrong object); regression to an earlier stage; target off table. Reach/pick: target pushed > 3 cm without a grasp.
Carry/place: release outside the opening, release with > 4 cm clearance above the floor, landing at rest outside the box.

Progress: reach −dist (span 0.30 m); pick 0.25·fingers + 0.5·clip(lift/3 cm)·grasped (span 1);
carry −3-D gap to the above-opening region (span 0.5 m); place −clearance above floor + 0.05 once released (span 0.15 m).
Example: place start with 12 cm clearance, lowered to 3 cm → +0.5·0.09/0.15 = +0.30 in shaping, then +1 at rest
on the floor (solver mean return 1.40). Hovering lowered without releasing (lower_hold) keeps the 0.33 and ends 0.

## Starts (`clutter_starts.py`, 100 seeds from SEED_BASE 940 000 000)
- reach: the reset scene (100/100).
- pick: sampled native-pad grasp opened around the target, −30° tilt and 2 cm offset alternatives, must not touch
  or move the target (100/100).
- handoff (handoff scenes): 96 of the first 100 eligible seeds (+0..199).
- carry: direct scenes — target teleported 7–12 cm up and held, kept only with a collision-checked carry path
  (`carry_plan`); handoff scenes — after a scripted handoff (28) or held by the receiver in M (24). 98/100
  (before the handoff amendment: 45/100).
- place: target posed over the opening (1–4 cm above rim + clearance), jaw aligned with a box side, level or −35°
  lean, and kept only if `place_plan` finds a collision-free straight lowering to 3 cm above the floor (xy shift
  inside the goal, ±90° yaw about the CoM). 86/100, + 6 chained from the carry solver = 92/100.
- End-to-end: 90/100 scenes have a valid start for every stage of their chain (39/100 before).
- Every held start passes a 32-action creep check and the 10-chunk idle check (100 % retention).

## Checking the reward
- Unit tests `tests/test_clutter_subtasks.py` (15/15, run as a script): target identity, qpos-only rejection, box
  move/tip, push and distractor loss, wrong-object lift, release opening/height, angular motion ≠ landing, floating
  target cannot finish, two real floor-supported placements finish, partition-independent shaping, idle + exact
  restore for every stage, numerical warnings, real high release, shoving contact ≠ reach, verified lowering.
- Scripted policies (`tools/scripted_rollouts.py`, 20 starts per stage, round 3):

| stage | solver +1 (planned) | return | hacks (mean return, outcome) |
|---|---|---|---|
| reach | 12/20 (12) | 0.96 | idle 0, away −0.02, wrong_object 0.28 (0), reach_push −0.05 (11/12 executed −1), box_move/tip −1, distractor_loss −1 (16/20) |
| pick | 15/20 (15) | 1.14 | close_only 0.25 (0), push −1, lift_open −1, wrong_lift −1, box −1, distractor −1 |
| carry | 18/20 (19) | 1.14 | release_early −1, wrong_way 0, shake 0, idle 0, box −1, distractor −1 (16/20) |
| place | 18/20 (20) | 1.40 | release_high −1, lower_hold 0.33 (0), leave_opening −1, box −1, distractor −1 (19/20) |

Handoff amendment (2026-11-14, `scripted_h1`–`h3`): handoff 14/20 (15 planned), return 0.90, best hack both_hold
0.15; carry after a handoff 20/20, 1.45, regrasp_giver −1 17/20; regressions pick 15/20, carry direct 19/20,
place 19/20 — all G4/G5 PASS, no hack +1.

No hack earns +1 on any start. Reach misses G5 only because 8/20 scripted shoves had no IK (see STATUS).

## Codemap
| path | what |
|---|---|
| `abc_sim/task_eval/clutter_stages.py` | `StageConfig`, `ClutterScene.observe` (box-local gap, clearance, floor support, box motion, speeds), `classify`, `progress` |
| `abc_sim/task_eval/clutter_subtask.py` | evaluator: fixed target, step failures, prompts, numerical-state guard |
| `abc_sim/task_eval/clutter_starts.py` | handoff: `TargetView`, `transport`, `plan_handoff`, `receiver_candidates`, `exchange`, `handoff_solver`, `build_handoff_start`, `place_from_carry`; `build_start`, `plan_link`/`move_link` (collision-checked IK paths), `carry_plan`, `place_candidates`/`place_plan` |
| `abc_sim/task_specs.py`, `task_eval/registry.py` | four `grab_specific_object_from_clutter__<stage>` specs, `clutter_subtask` evaluator |
| `tests/test_clutter_subtasks.py` | G2 tests |
| `subtask_rewards/clutter/tools/` | snapshot of `../simulation/experiments/subtask_rewards/clutter/{build_starts,scripted_rollouts}.py` |
| `../simulation/outputs/subtask_rewards/clutter/` | `handoff_r3/starts/` (current full set), `scripted_h1`–`h3` (handoff amendment), `final/`, `scripted_r3/` (before) |

## Open questions
1. Accept reach despite the G5 letter (harness could not execute 8/20 shoves)?
2. (Resolved 2026-11-14: handoff stage added; review its decisions in STATUS.md.)
3. Place solver leaves an opened finger resting on tall targets (2/20 ended 0): a retract move would finish them;
   a policy must learn to back off.
4. Renders / HF Space tab not made (needs the org token).
