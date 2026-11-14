# inhand_transfer_item_to_other_side — single-stage RL subtasks

Status, numbers and the repair log: `STATUS.md`. Approved stage table (G1, 2026-11-14): `stage_table.md`.

## Idea

One kitchen item (10 categories) spawns on one side of the table. The arm on that side (**giver A**) picks it up and
brings it to the middle, hands it in the air to the other arm (**receiver B**), which carries it across and sets it
down on the far side. Six trainable stages, each starting from a saved state in which its precondition holds; the
goal of stage k is the precondition of stage k+1, so the trained subtasks chain back into the whole task:

    reach -> pick -> present -> handoff -> carry -> place -> done (fixed stock success)

Everything is decided from simulator state: per-arm finger contacts, any-robot contacts, item pose/speed, and the
distance from each arm's **pinch centre** to the item's surface.

**Pinch centre, not grasp site.** The YAM finger pads span link_6 y −0.047…−0.017 and z 0.07…0.15 (both arms), so
the `<arm>_grasp_site` at link_6 (0, 0, 0.1347) sits 3 cm beside the pads. A weightless close (stagekit
`hold_object`) hides this because the fingers pull the item in; a receiver closing on an item the giver holds rigidly
misses it entirely. All inhand grasps, "near" and "clear" use `PAD = (0, −0.031, 0.125)` in link_6.

## Stages (`abc_sim/task_eval/inhand_stages.py`, handoff tests in `handoff_stage.py`)

s = +1 for a left (+y) spawn; "far" uses the item origin like the stock scorer.

| stage | prompt | in it when | goal (= next stage) | progress (span) | failures (−1) |
|---|---|---|---|---|---|
| reach | `inhand transfer subtask: reach the item with the near arm` | none of below | A's pinch centre ≤ 4 cm from the surface (exit 6 cm) or an A finger on it | −gap_A (0.30) | pushed > 3 cm while not held by A; B touches it; A touches it with a closed gripper (< 50 % open); off table |
| pick | `… grasp and lift the item with the near arm` | near | both A fingers on it, lifted ≥ 3 cm | 0.25·f_A + 0.5·lift/3 cm·[f_A = 2] (1.0) | back to reach; pushed; B touches |
| present | `… bring the item to the middle` | held by A, lifted | CoM in M (x 0.45–0.70, \|y\| ≤ 0.10, z 0.83–1.15), held | −dist(CoM, M) (0.30) | dropped (no robot contact, > 4 cm up); A touches it with its origin in the far band (y·s ≤ −0.10) |
| handoff | `… hand the item to the other arm` | in M (exit 10 cm), lifted, A holds or B has a finger on it | B both fingers, A none and its pinch centre ≥ 6 cm from the surface, lifted, touching nothing but B | 0.25·f_B + [f_B = 2]·(0.25·(2 − f_A) + 0.25·[A let go]·min(gap_A/6 cm, 1)) (1.25) | dropped; item touches the table (decision 2); leaving M by > 10 cm = back to present |
| carry | `… carry the item to the other side` | B holds alone, A clear | origin y·s ≤ −0.13 (exit −0.12), on the table, ≤ 15 cm above rest, B holds alone | −gap to the far zone (0.30) | dropped (any release > 4 cm up); A touches it again |
| place | `… set the item down on the other side` | over the far zone, A off, B touching | fixed stock success: band, ≤ 10 cm up, on table, no robot contact, < 2 cm/s | −clip(lift, 0, 0.3) + 0.05·[released] (0.10) | released with its lowest point > 5 cm above the table and nothing else under it (decision 3); comes to rest outside the band; A touches |

A free item (no robot contact) that is still moving keeps its label (falling, sliding). Fallen = CoM > 2 cm below
the tabletop or the origin off the table. Debounce: a label must hold 5 actions.

## Reward

Kit `SubtaskEvaluator`, `reward_mode="progress"`, 32-action chunks: +1 on the goal, −1 on a failure (both end the
episode), else 0.5·Δprogress/span (potential-based: wobbling earns nothing).

Worked examples:
- handoff, solver: start 0 (A holds). B closes: 0.25·2 + 0.25·(2 − 2) = 0.5; A opens: 0.5 + 0.5 = 1.0; A backs off
  6 cm from the surface: 1.25 → shaping 0.5·1.25/1.25 = 0.5, then +1 at the goal: return ≈ 1.5 (measured mean 1.21
  including the 4/20 misses).
- handoff, both_hold (B grasps, A never opens): progress 0.5 → return 0.2, never +1.
- handoff, release_stay (A opens but stays): progress ≤ 1.0 + a little → return ≤ 0.4 + ε, no +1 (A not clear).
- place, hold_low (lowers to the table, never opens): progress −0.09 → 0 → return ≈ +0.45, never +1 (stock needs
  release); place solver: same + 0.05 release bonus + +1.
- carry, dropped over the far zone before the label commits: −1 (carry never pays for a drop).

## Starts (`abc_sim/task_eval/inhand_starts.py`; tooling `../simulation/experiments/subtask_rewards/inhand/`)

SEED_BASE = 1 100 000 000. Every start: reset(seed), settle (the item drops from z = 0.82; settled = < 5 mm/s for
20 actions), then:
- reach: the settled scene. pick: A's open top-down grasp at the CoM section / upper part, raised 0–3 cm until the
  open fingers touch nothing, 10 open actions must not move the item.
- present: A holds it 5–15 cm above its rest spot (|y| pushed out to ≥ 0.14), kit `hold_object` + creep check.
- handoff: item moved weightless to M at a yaw from {90, 60, 120, 30, 150, 0}°; A holds it with the first grasp
  family (ends top-down / top + side / ends + side / side + side) for which B has a collision-free grasp *with A's
  gripper in place*.
- carry: chained — a handoff start + the scripted handoff solver; fallback (source 3) B teleported onto its grasp,
  then the same monitored release.
- place: B holds it 4–10 cm above a random far-zone point.
- Exclusions (approved decisions 4/5): spawn-overlap seeds (item > 5 mm inside the table at reset) for all stages;
  cream-cheese sticks and cheese graters for handoff/carry/place. A start set = the first 100 eligible seeds.

## Results (20 scripted starts per stage; details and repair log in STATUS.md)

| stage | status | starts valid/100 | solver +1 | solver return | best hack return | margin |
|---|---|---|---|---|---|---|
| reach | STOPPED (G5 letter: push hack unplannable on 6/20; no hack +1) | 100 | 20/20 | 1.46 | 0.00 | 1.46 |
| pick | PASSED | 100 | 19/20 | 1.17 | 0.25 (close_only) | 0.93 |
| present | PASSED | 99 | 19/20 | 1.15 | 0.00 | 1.15 |
| handoff | PASSED | 87 | 16/20 | 1.21 | 0.36 (release_stay) | 0.85 |
| carry | STOPPED (G5 letter: regrasp_A obstructed, 13/20; 2/20 +1 by shoving B's gripper) | 97 | 20/20 | 1.14 | 0.00 | 1.14 |
| place | PASSED | 91 | 19/20 | 1.17 | 0.27 (hold_low) | 0.90 |

Open questions: accept reach/carry despite the G5 letter, or add a carry failure "giver touches the receiver arm"
(closes the 2/20 shove); a kit-level pinch-centre constant (the grasp-site offset affects every task); renders and a
Space tab need the org token.

## Checks

- G2: `tests/test_inhand_subtasks.py` (13 tests: lookups, task-independent handoff tests, classify chain, starts
  classify + 10 idle chunks + exact restore per stage, every listed failure, hold-low never succeeds, shaping).
- G3/G4/G5: see STATUS.md (start sets and scripted solvers/hacks per stage).

## Codemap

| file | what |
|---|---|
| `abc_sim/task_eval/handoff_stage.py` | task-independent handoff: region test, received/in-air tests, progress, failures |
| `abc_sim/task_eval/inhand_stages.py` | `StageConfig`, `InhandScene` (per-arm contacts, pinch-centre gaps, stock success), `classify`, `progress` |
| `abc_sim/task_eval/inhand_subtask.py` | `InhandSubtaskEvaluator` (side from metadata, per-step failures) |
| `abc_sim/task_eval/inhand_starts.py` | grasp planner (pinch centre), motion helpers, monitored handoff solver, `build_start` |
| `abc_sim/task_specs.py`, `task_eval/registry.py` | `inhand_transfer_item_to_other_side__<stage>`, `inhand_subtask` |
| `../simulation/experiments/subtask_rewards/inhand/build_starts.py` | start sets + G3 idle check |
| `../simulation/experiments/subtask_rewards/inhand/scripted_rollouts.py` | solvers + hacks, G4/G5 summary |
