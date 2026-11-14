# put_relative subtasks for single-stage RL: design + codemap

Second task through the subtask pipeline (`.claude/skills/abc-subtask-reward`), after mug flip. Ledger with every gate,
number and repair: `STATUS.md`. Built on the shared kit `abc_sim/task_eval/stagekit/`.

## Quick start
```python
import abc_sim
env = abc_sim.make_env(task="put_relative__carry", render_cameras=False)   # or __reach / __pick / __handoff / __place
obs, info = env.reset_to_stage(seed)        # builds this stage's start (~1 s): here the mover is held top-down
prompt = info["prompt"]                     # e.g. "carry the lemon behind the coffee cup" (per episode)
ev = env._task_evaluator
done = False
while not done:
    for a in policy(obs, prompt)[:32]:      # one chunk = 32 actions
        obs, _, _, _, info = env.step(a)
    reward, done = ev.chunk_reward()        # +1 goal, -1 failure, potential-based shaping in between
```
Prebuilt starts: `../simulation/outputs/subtask_rewards/put_relative/starts/<stage>.npz`
(`reset_to_stage(seed, row)` restores one exactly): reach 94, pick 92, carry 117 (same-side), place 82;
handoff and cross-table carry 100 each in `starts/handoff_v1/{handoff,carry_cross}.npz`.

## Task
"put the {mover} {in front of / behind / to the left of / to the right of} the {reference}". 4–6 household objects
(12–30 g) on a 3×4 slot grid. Stock success (`PutRelativeEvaluator`): the mover's xy offset from the *live*
reference is 6–30 cm along the prompted axis and ≤ 8 cm across it, and the mover is ≤ 3 cm above its spawn height.

## Stages (the mover only; `put_relative_stages.py`)
| stage | prompt (live) | in it when (= start precondition) | goal (= next stage's start) |
|---|---|---|---|
| reach | move a gripper to the {mover} | none of the below | grasp site ≤ 5 cm from the mover's CoM, or a finger touches it |
| pick | grasp the {mover} and lift it | near, or grasped but lifted < 3 cm | both fingers of one gripper on it, lifted ≥ 3 cm |
| carry | carry the {mover} {phrase} the {reference} | grasped and lifted | xy inside the stock zone while held or touched |
| place | put the {mover} down {phrase} the {reference} | in the zone and held/touched | stock success **and** no finger on it **and** at rest (< 2 cm/s) |

- Stage from sim state only (contacts, poses, speeds); hysteresis on near (exit 7 cm) and on the zone (exit 1 cm).
  Debounce 5 actions. A mover moving freely (no finger, not at rest) keeps its previous stage.
- Spec prompts (lookup keys) are fixed "put relative subtask: …" strings; the episode prompt is filled with names.

## Reward (`PutRelativeSubtaskEvaluator.chunk_reward`, kit default "progress" mode)
- **+1** goal; **−1** failure, both end the episode:
  - **dropped** (carry, place): no finger on the mover while it is > 4 cm above rest (checked every step).
  - **pushed** (reach, pick): the mover moved > 3 cm while not held, or reached the zone without being carried.
  - **moved the reference** (any stage): the reference moved > 3 cm.
  - back to an earlier stage; off the table.
- In between: 0.5 × Δprogress / span. Progress: reach −dist (span 0.30 m), pick 0.25·fingers + 0.5·lift fraction
  (1.0), carry −xy gap to the zone (0.30 m), place −height above rest + 0.05 if released (0.10 m).

## Starts (`put_relative_starts.py`)
- Synthesized top-down grasps (link_6 straight down, grasp site 1 cm below the CoM, random yaw), no recordings.
- reach: the reset scene. pick: open gripper around the resting mover (at CoM height), not touching it.
  carry: held 5–15 cm above its spawn spot. place: held 3–10 cm above a random zone point, by the arm on that side.
- Rejected: seeds whose reset raises or already satisfies the directive; held starts whose object sinks > 1 mm in
  one extra chunk (creep); **carry: episodes whose zone is across the table** (57 % of usable seeds; one top-down
  arm can't reach across — they need a handoff subtask, not built yet).
- Kept only starts still in their stage after 10 idle chunks.

## Checks
- Phase 0 (stock scorer): held-in-zone counts as success (8/8); 1.3 % of resets already succeed; 1 % of resets
  raise. Handled in the subtask goals/starts; stock scorer and randomizer unchanged (user decision).
- Tests `tests/test_put_relative_subtasks.py` 11/11: starts in stage, idle ≈ 0, mid-air open = dropped, push and
  push-into-zone = −1, reference moved = −1, carry reward = scaled gap change, 1 big move = 8 small, scripted place
  +1 with stock success, exact restore.
- Scripted behaviours (20 starts per stage, `../simulation/experiments/subtask_rewards/put_relative/scripted_rollouts.py`):

| stage | solver +1 | hacks (expected outcome) |
|---|---|---|
| reach | 14/20 (14/14 planned) | move away 0 (20/20), idle 0 (20/20) |
| pick | 20/20 | close only 0, lift 2 cm + open 0, push to zone −1 (9/9), idle 0 |
| carry (same side) | 12/20 (12/12 planned) | wrong way 0 (8/9), open early −1 (12/12), shake 0, idle 0 |
| place | 19/20 (19/19 planned) | open 10 cm up −1 (17/17), lower and hold 0, leave the zone −1 (17/17), idle 0 |

  Solver − best hack mean return ≥ 1.0 in every stage; no hack reached +1. Unplanned starts are the scripted
  solver's IK limits (straight-down gripper near the wrist's range), not the reward.

## Codemap
| file | owns |
|---|---|
| `abc_sim/task_eval/put_relative_stages.py` | `StageConfig`, `PutRelativeScene.observe` (zone geometry from the live reference), `classify`, `progress`, prompts |
| `abc_sim/task_eval/put_relative_subtask.py` | evaluator: directive from randomization metadata, failures, live prompt, `build_start` |
| `abc_sim/task_eval/put_relative_starts.py` | `topdown_grasp`, `reset_scene` (seed filters), `build_start`, `hold_lifted`, `build_handoff_start`, `build_cross_carry_start` |
| `abc_sim/task_eval/put_relative_handoff.py` | cross-table sides, `MoverView`, box gap, giver/receiver grasps, handoff planner + scripted handoff |
| `abc_sim/task_eval/handoff_stage.py` | task-independent handoff tests (shared with inhand; not changed) |
| `abc_sim/task_specs.py`, `task_eval/registry.py` | `put_relative__{reach,pick,handoff,carry,place}`, `put_relative_subtask` |
| `tests/test_put_relative_subtasks.py` | G2 tests |
| `../simulation/experiments/subtask_rewards/put_relative/` | `build_starts.py` (start sets + idle check, `--sides`), `merge_starts.py` (G3), `relabel_starts.py`, `scripted_rollouts.py`, `scripted_handoff.py` (handoff + cross-table carry), `probes/` |
| `subtask_rewards/put_relative/probes/` | phase 0 scorer probes |

## Handoff for cross-table episodes (2026-11-14 amendment; table in `stage_table.md`)
Chain reach → pick → **handoff** → carry → place for the 171/290 usable seeds whose zone point is across the table;
same-side episodes skip handoff and are classified exactly as before (code path untouched; saved starts and the
scripted runs reproduce bit-for-bit, see STATUS).

- Giver G = arm on the mover's side, receiver V = the other one. Tests = `handoff_stage.py` unchanged:
  `HandoffConfig(region = whole work space)`, so "lifted and held by G" (the pick goal) is already a handoff state;
  goal = `received_in_air` (V both fingers, G off and its pinch centre ≥ 6 cm from the collision box, lifted, touching
  only V). The scene adds `com, f_give, f_recv, touch_give, touch_recv, gap_give, env_touch, mid_gap` only in
  cross-table episodes (`PutRelativeScene.handoff_obs`).
- Progress: `handoff_stage.progress` + 0.25 · (1 − min(d_mid / 0.25, 1)) (span 1.5) — the middle-box term shapes
  bringing the mover to where V can take it (the user's chain has no separate "present" stage).
  Worked example (solver): start 0.17–0.25 (G holds, 8 cm from the middle box) → in the box 0.25 → V closes 0.75 →
  G opens 1.25 → G 6 cm clear 1.5; shaping 0.5·Δ/1.5 ≈ 0.42, then +1 → measured mean return 0.90 incl. misses.
  both_hold tops out at 0.75 progress → ≈ 0.17 return, never +1.
- Failures: dropped; touched table/other objects (in the air only); **G carried it into the zone**; carry/place:
  **G touched the mover again**.
- Cross-table carry/place: carry = received in the air (stays carry while received), goal = in the zone with a V
  finger on it and G not touching; place starts for cross-table episodes are held by V over the zone's part on V's
  side.
- Starts: handoff — G holds the mover top-down with the **pinch centre** on its axis (the grasp site is 3 cm beside
  the pads; the old top-down starts hold small movers by their edge: pad centre 3.1 cm from the CoM, so the giver
  lost 3/5 movers on the way to the middle in the first probe), lifted 5–15 cm, kept only if a kinematic handoff plan
  exists. Cross-table carry — chained handoff, else V alone in the middle (source 3).
- Scripted handoff (`put_relative_handoff.py`): G lifts, then carries along a planned joint path to a handoff pose
  (point x 0.45–0.65 / y 0–15 cm toward G, yaw 0/±90/180, wrist tilt 0/±45/±90 about its closing axis; points in
  the zone skipped); V side-on from its side (closing vertically between G's fingers, or horizontally on the lower
  part), checked open and closed for collisions; G opens in steps (monitored), backs off.

Results (20 starts): handoff solver 14/20 (14/18 planned), return 0.90, best hack release_stay 0.27 (margin 0.63);
cross-table carry solver 20/20, return 1.17, regrasp_giver 1/20 +1 (G shoved V's hand into the zone). Coverage:
258/290 scenes end to end (89 %, was 40 %).

## Open
- Cross-table episodes without a handoff plan (18/171): flat/round movers (sponges, tomatoes) where G's top-down
  grasp leaves V no free face within reach.
- Re-measure the 5 cm "near" threshold once policy grasps exist (side grasps sit farther from the CoM).
- Renders / Space tab not made (CPU Mesa rendering works; publishing needs the OneWorld-AI token).
