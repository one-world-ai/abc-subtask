# hang_mug_on_mug_rack — single-stage subtask rewards (design and results)

Status 2026-11-14: **pick ships**. reach is stopped by the letter of G5 (its push hack reaches the approved
"finger touches" goal). align, insert and release are stopped at G3 (start sets below 80/100 after 3 repair rounds).
Ledger with every number and evidence path: `STATUS.md`. Approved table: `stage_table.md` (G1, 2026-11-14).

## Idea
One target mug per episode: the threadable mug nearest a gripper that is not hanging yet (mug_1/7/10 have
handle holes narrower than a peg and are never targets; seeds whose only mugs are those are skipped). Everything is
decided from sim state. The stock scorer (with the phase-0 threading rule: a tree piece must pass through the
handle's measured hole) is the final goal. Goal of stage k = precondition of stage k+1.

## Stages (`abc_sim/task_eval/mug_rack_stages.py`)
| stage | prompt (`hang_mug_on_mug_rack__<stage>`) | in this stage when | goal |
|---|---|---|---|
| reach | mug rack subtask: move a gripper to the target mug | none of the below | grasp site ≤ 10 cm from the CoM, or a finger touches it |
| pick | ... grasp the target mug away from its handle and lift it | near / touched | both fingers of one arm, no finger contact on the handle region, lifted ≥ 3 cm |
| align | ... line the target mug's handle up with a free peg | held (handle free), lifted | hole centre ≤ 1.5 cm from a free peg's axis line, 0–4 cm beyond its tip, hole axis ≤ 30° from the peg, touching nothing but fingers |
| insert | ... slide the target mug's handle onto the peg | aligned (exit: 2.5 cm / ±6 cm / 40°, peg contact allowed) or threaded shallow, grasped | threaded on a free peg with the hole centre ≥ 2 cm past the tip (exit 1 cm), a finger on it |
| release | ... let go of the target mug on the rack | threaded ≥ 2 cm, finger on it | stock per-mug hanging, no finger, < 2 cm/s, for 5 actions |

Pegs = the 6 tree pieces within 45° of horizontal (axis by PCA of the piece, pointing out of the trunk, tip = outer
end); the trunk pieces are never pegs. "Free" = no other mug threaded on it. Per-peg threading uses the scorer's
hole points and the piece's face planes (bounding-sphere pre-filter).

Failures (−1, latched per action): off the table (kit); fallen back to an earlier stage (kit); knocked over (reach,
pick: cup > 60° from vertical, no finger); dropped (align, insert: no finger for 8 actions while not threaded
≥ 1 cm); released, not hanging (release: no finger, at rest, stock says not hanging); a mug hanging at the start
stops hanging for 5 actions (any stage).

## Reward (kit `SubtaskEvaluator.chunk_reward`, `reward_mode="progress"`, shaping total 0.5)
+1 goal, −1 failure, else 0.5 × Δprogress / span per 32-action chunk:

| stage | progress | span |
|---|---|---|
| reach | −dist(grasp site, CoM) | 0.30 m |
| pick | 0.25 × fingers + 0.5 × min(lift / 3 cm, 1) × grasped-with-handle-free | 1.0 |
| align | −min over free pegs (radial + 0.5 × axial gap outside [0, 4 cm] + 0.02 m × angle / 30°) | 0.40 |
| insert | −min over free pegs (radial + (2 cm − depth)+) | 0.06 |
| release | 0.025 × (2 − fingers) − 0.5 × min(depth lost since the start, 0.1 m) | 0.10 |

Worked example (pick): a start has fingers 0, lift 0 → progress 0. Closing both fingers: progress 0.5 → chunk reward
0.25. Lifting 1.5 cm while grasped: progress 0.75 → +0.125. Reaching 3 cm for 5 actions → +1, episode over. A
policy that closes and wobbles collects at most 0.25 (close_only mean return 0.23).

## Starts (`abc_sim/task_eval/mug_rack_starts.py`, SEED_BASE = 1 200 000 000)
- Grasps are synthesized rim pinches (source 3): link_6 y closes radially across the cup wall 1.5 cm (2.5 cm for
  hang poses) below the rim, link_6 z into the cup, tilted 0/±20°/40°, rim angles ≥ 60° from the handle, both wrist
  mirrors. The mug-flip library grasps were not used: the phase-0 pilot found 198/224 fail IK or the hold in hang
  orientation, and the library holds tilted mugs.
- reach: reset scene. pick: open pinch around the resting mug, must not touch it or move it > 1 cm. align: pick start
  + pick solver (close, lift 8–15 cm) + a one-chunk creep check. insert / release: per (peg, side), place the hole
  on the peg axis 3–4.5 cm inside the tip, handle up, settle 150 actions with the arms home; keep it only if the
  stock scorer says hanging on that peg at rest; move it to a contact-free pose (release: in place, lifted to the
  middle of the free range; insert: hole 0.5–3.5 cm beyond the tip); IK + kinematic-collision filter over the
  pinch candidates; `stagekit.hold_object`; one-chunk creep check.
- Setup steps run with the evaluator detached (`_no_eval`): the evaluator only reads state, and the stock threading
  test costs ~20 ms per action on a hanging mug.

| stage | valid / 100 | G3 |
|---|---|---|
| reach | 93 (7 skipped seeds) | passed |
| pick | 87 | passed |
| align | 66 (best round 73) | stopped |
| insert | 36 (best 50) | stopped |
| release | 51 (best 66) | stopped |

## Checks
- G2: `tests/test_mug_rack_subtasks.py` 15/15; `tests/test_mug_tree_scorer.py` 8/8.
- G4/G5 (`../simulation/experiments/subtask_rewards/mug_rack/scripted_rollouts.py`, 20 starts, 32-action chunks):

| stage | solver | hacks (outcome counts / mean return) | margin |
|---|---|---|---|
| reach | 20/20, 1.43 | idle 0 (−0.01), away 0 (−0.02), push **+1 20/20** (1.35) | 0.08 ✗ |
| pick | 18/20, 1.19 | idle 0, close_only 0 (0.23), lift 2 cm + open 0 (0.0), handle grasp −1 18 / +1 1 (−0.82) | 0.97 ✓ |
| align (pilot, 16) | 4/16 (5 planned) | open mid-air −1 16/16, shake 0, rotated 90° (3 planned) 0/−1 | – |
| insert (pilot, 16) | 8/16 (15 planned) | open early −1 13, push sideways −1 10 / 0 5, carry over trunk top +1 3/16 (threaded while held) | – |
| release (pilot, 16) | 6/16 | yank −1 15/16, open at the tip +1 2 / −1 11 | – |

## Codemap
| file | what |
|---|---|
| `abc_sim/task_eval/mug_rack_stages.py` | `StageConfig` (thresholds + sources), `MugRackScene` (pegs, holes, handle contact, other contact, stock hanging), `classify`, `progress` |
| `abc_sim/task_eval/mug_rack_subtask.py` | evaluator: target choice, failures, release progress, tree re-measure on in-place resets |
| `abc_sim/task_eval/mug_rack_starts.py` | rim pinches, hang settle, clear pose, grasp feasibility, `build_start`, `pick_solver` |
| `abc_sim/task_specs.py`, `task_eval/registry.py` | `hang_mug_on_mug_rack__<stage>` specs, `mug_rack_subtask` |
| `../simulation/experiments/subtask_rewards/mug_rack/{build_starts,scripted_rollouts}.py` | start sets (CPU-time budget) and behaviours |

## Open questions for the user
1. reach: accept it (push = touch = the approved goal, same as clutter) or change the reach goal to the grasp-site
   distance only (needs a new G1).
2. align/insert/release need sturdier hang-orientation holds: a handle-side or body grasp family designed for
   the cup axis horizontal, or a sim change (finger friction/pads) — both outside the approved table.
3. Insert "carry over the trunk top" threads a peg while held on 3/16: real insertion by the goal's definition;
   keep, or require the approach to come along the peg axis.
