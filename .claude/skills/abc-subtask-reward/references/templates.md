# Stage templates

Start from the closest template, then adapt. Each lists stages, progress, failures, and the scripted behaviours
phase 4 must run (S = solver, expect +1; others are hacks/mistakes with their expected outcome).

## A. Pick-and-place (per object): bottles, lego, nuts/bolts, clutter, put_relative
| stage | precondition | goal | progress | failures |
|---|---|---|---|---|
| reach | none of below | grasp site ≤ r_near of the object (measure r_near from grasps) or a finger touches | −dist | object off table / knocked into goal region by pushing |
| pick | near | both fingers of one gripper on it, lifted ≥ 3 cm | 0.25·fingers + 0.5·lift/3cm·grasped | regressed to reach |
| carry | grasped + lifted | object above the goal region (xy inside, z above rim/clearance) still held | −xy dist to goal | dropped outside the goal (no finger, > 4 cm up) |
| place | above goal, held | stock per-object success, no finger touching | −height above goal + release bonus | dropped from > drop height (if task says "put"), lands outside |

Behaviours — reach: S, move_away (0), idle (0), push_object (0 or −1; must never reach +1 in later stages).
pick: S grasp_lift, lift_open (−1), close_only (0), one_finger_drag (0). carry: S, release_early (−1), wrong_goal
(0/−1: wrong bin for sorting), shake (0). place: S, release_high (−1 for "put", +1 allowed for "throw"), hold (0).
Multi-object: target = one object per episode (closest to a gripper, or the prompted one); other objects only matter
via failures (knocked off table). Sorting/counting: wrong bin or over-count is a failure.

## B. Reorient in hand: mug flip (done) — reach / pick / rotate (goal: upright while held) / place.

## C. Articulated: open/close drawer
reach handle (grasp site ≤ 3 cm of handle) → grasp (fingers on handle) → pull/push (joint position to target) →
release (gripper off, joint stays). Progress: −dist, fingers, joint fraction. Failures: wrong drawer moved > 5 cm,
handle slip (regress). Behaviours: S, pull_wrong_drawer (−1), push_without_grasp (counts only if the task allows
it — decide in phase 1), idle.

## D. Handoff: inhand transfer
reach → pick (arm A) → handoff (both grippers on it, then only arm B) → place far side. Progress in handoff:
fingers_B − fingers_A·held_B. Failures: dropped mid-air, regrasp by A after handoff. Behaviours: S, drop_between (−1),
place_with_A_only (0/−1: the task requires the transfer).

## E. Tool use: sweep
grasp tool → bring tool to scraps → sweep (scraps moved into bin region) → release tool. Progress: scraps' distance
to bin (sum, clipped). Failures: tool dropped, bin tipped (also fix the bin-frame scorer first). Behaviours: S,
push_bin_over (must be −1 or 0, never +1), hand_sweep_without_tool (decide), idle.

## F. Insertion / hanging: hang mug, dish rack
reach → pick (handle-side grasp for mugs; rim/edge grasp for plates) → carry (handle hole or plate edge aligned at
the peg/slot, ≤ 2 cm) → insert (stock success while still held: handle around peg / plate in slot upright) →
release (stock success, no finger). Progress in insert: −distance of handle-hole centre to peg axis + depth.
Failures: dropped, knocked rack/tree, object resting on rack in a non-scoring way for > N chunks (reward hack:
balancing on top). Behaviours: S, drop_onto_rack (must be 0/−1 — the mug-tree audit found 57/63 random drops
rest on the tree without hanging), lean_on_outside (dish rack false positive), idle.
Expect solver trouble: these are the likely STOP candidates; stop per stage, keep reach/pick/carry if they pass.
