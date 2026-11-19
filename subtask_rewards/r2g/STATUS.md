# Real2Gym "in a container" stages: status (2026-11-16)

Tasks: `put_the_screwdriver_in_the_bin`, `place_and_organize_the_fake_fruits_in_the_fruit_bowl`,
`organize_the_makeup`. Env ids `r2g_<task>__<reach|pick|carry|place>`. Code: `abc_sim/task_eval/r2g_stages.py`,
`r2g_subtask.py`, `r2g_starts.py`. Stage table (G1, approved with the first option of each decision):
`stage_table.md`. Numbers: `validation.json` (`tools/gates.py`).

| gate | result |
|---|---|
| G0 scene / scorer | not run as a separate audit: the final check is the existing `Real2GymEvaluator` obj check, unchanged |
| G1 design | approved 2026-11-16 |
| G2 code | `tests/test_r2g_subtask.py` 5/5 |
| G3 starts | 12/12 stages >= 80 of 100 seeds, <= 7.7 CPU s per start, all kept starts idle-stable for 10 chunks |
| G4 solver | 11/12 (screwdriver place 13/20) |
| G5 reward order | 11/12 (makeup pick: other-arm hack planned on 9 of 20 starts, 10 needed; all 9 gave -1) |

| task | stage | starts | solver +1 (of 20) | hacks that earned +1 | verdict |
|---|---|---|---|---|---|
| screwdriver | reach | 100 | 20 | none | usable |
| screwdriver | pick | 100 | 17 | none | usable |
| screwdriver | carry | 100 | 20 | none | usable |
| screwdriver | place | 99 | 13 | none | **not usable (G4)** |
| fake fruits | reach | 100 | 20 | none | usable |
| fake fruits | pick | 100 | 16 | none | usable |
| fake fruits | carry | 97 | 18 | none | usable |
| fake fruits | place | 97 | 20 | none | usable |
| makeup | reach | 100 | 20 | none | usable |
| makeup | pick | 97 | 20 | none | **G5 by the planning rule only** |
| makeup | carry | 84 | 18 | none | usable |
| makeup | place | 83 | 20 | none | usable |

Starts: `policy-finetuning/abc_rl_data/r2g_starts/r2g_<task>/<stage>.npz` (1,154 rows; not uploaded).
Rollouts: `policy-finetuning/abc_rl_data/r2g_rollouts/` (first run in `r2g_rollouts_v1`).

## Findings
- **Screwdriver place (G4 fail).** In 6 of 20 solver runs (4 of 20 when the gripper just opens) the screwdriver
  stays on the fully open fingers: 19 cm long, it rests across the finger bodies and the episode never ends. One
  run pushed the bin > 5 cm. Not repaired: stage table decision 6 keeps the object's physics as imported.
- **Opening the gripper at the place start is enough** (`drop_high`: makeup 20/20, fruits 20/20, screwdriver
  15/20). That is decision 3 (the stock check decides; no lowering required) and makes place a short stage.
- **Place = the obj check.** Every place +1 (108 runs) coincides with the obj check's own success. First run: 4 of
  28 screwdriver +1 came 0.4 s before the check's, because a released object keeps its label while it rocks and the
  hold bridged the wobble; `goal_reached` now uses the check's condition at each step.
- **Pinch time.** The obj check needs 0.2 s of pinch before the goal counts; the timer is evaluator state, not sim
  state. It is stored in the last row of `rest_height` and carried through `advance_stage`.
- **Grasp aim.** The finger pads are 4.4 cm to the side of the grasp site: starts aim the pads at the centre of
  mass. Where the container cannot be reached from straight above the approach is tilted 15 / 30 / 40 deg.
- **Dropped starts.** Makeup carry / place lose 16-17 of 100 seeds: container out of the named arm's reach
  (decision 5: no handoff stage yet).
- **Retreat in pick is -1** ("back to reach" when the open gripper leaves the object by > 7 cm): stagekit's rule.

## Repair rounds
1. Carry solver retries its target point (carry +1: 17 / 11 / 10 -> 20 / 18 / 18 of 20); other-arm reach hack
   plans without a pre-grasp point (reach other_arm planned 18 / 13 / 16 -> 20 / 14 / 20).
2. Place goal tied to the obj check's instantaneous condition (see above).

## Open
- Makeup pick: accept as an exception (as nuts/bolts reach), or run the other-arm hack on 40 starts.
- Screwdriver place: accept as not usable, or change finger friction / the hold orientation for this object.
- The other seven "in" tasks of batch 1b: add to `R2G_STAGE_TASKS` with their un-graspable objects in the exclude list.

## Batch 1b (2026-11-16): seven more "in" tasks

Added to `R2G_STAGE_TASKS`: condiment bottles, hair-cutting tools, tools, stationery, beverages onto shelf, canned
foods onto counter, wine bottle onto rack. Same code; 100 seeds per stage, 20 starts per behaviour. Full table:
`validation.json` (`tools/gates.py`). No hack earned +1 on any stage (drop_high on place is expected "any").

| task | reach | pick | carry | place |
|---|---|---|---|---|
| condiment bottles | pass | rule only (other-arm hack planned 0/20) | pass (87 starts, 20/20) | pass (95 starts, 20/20) |
| hair-cutting tools | pass | rule only (8/20) | no starts (7/100) | no starts (7/100) |
| tools | rule only (7/20) | FAIL G4: solver 9/20 | no starts (0/100) | no starts (0/100) |
| stationery | pass | pass (14/20) | FAIL G3: 59 starts | FAIL G3: 51 starts |
| beverages onto shelf | pass | rule only (7/20) | no starts (0/100) | no starts (0/100) |
| canned foods onto counter | pass | rule only (8/20) | no starts (0/100) | no starts (0/100) |
| wine bottle onto rack | FAIL G4: solver 5/20 | FAIL G3: 76 starts | no starts (1/100) | no starts (1/100) |

"rule only" (accepted as passed by the user, 2026-11-17) = G3 and G4 pass, every hack run behaved as expected, the solver-hack gap is >= 0.5, but the other-arm
hack could be planned on fewer than 10 of 20 starts (the other arm cannot reach the object). Same case as makeup pick.

Findings
- Carry / place as defined ("held over the container's footprint, above its top") only fit low open containers
  (bin, bowl, tray). Drawer cabinet (26 cm), clipper box (21 cm), counter board (31 cm), shelf (35 cm, top at 1.32 m)
  and wine rack are tall or entered from the side: the pose over the top is out of reach or collides. These need
  their own stage table (side entry), not a threshold change. Stationery holders are reachable from above in about
  half of the scenes only.
- Condiments: `bottle03_grinder` excluded (no collision-free hold over the tray in 24 of 24 scenes).
- The objects the first grasp probe could not lift (pens, markers, small wrench, pocari_1, congee_1, slim) do get
  pick starts; no exclusion was needed for reach / pick.
- Scripted pick solver now lowers the pads to the object before closing (starts are up to 3 cm above it): stationery
  pick 3/20 -> 14/20, screwdriver 17 -> 19, hair tools 11 -> 17. Flat wrenches / pliers still slip (tools 9/20).
- Wine rack reach: the scripted top-down approach knocks the standing bottles over ("pushed") in 7 of 20 runs.

Space: Real2Gym tasks are a separate group ("Real2Gym tasks (real -> gym)") on
`OneWorld-AI/subtask-reward-visualization`; entries from `tools/space_entries.py`, videos from
`slurm/torch/r2g_render.bash` (all 258 videos replay their saved rewards exactly). The rollout tool now saves
`<out>/<task>/<stage>/<behaviour>_<seed>.npz` (actions, chunk rewards) for that.

## Start-builder fixes (2026-11-17)

- Screwdriver place: 5 of 6 stuck solver runs started from a hold with the screwdriver wedged across the open gripper
  (fingers 8.2 cm apart, both touching). `build_start` now rejects a hold whose fingers did not close to the object's
  width (`WEDGE_SLACK_M`). Place: solver 13/20 -> 20/20, starts 100/100.
- Tall containers: 300 sampled gripper poses per arm showed reachable collision-free poses exist, but with the gripper
  50-90 deg from vertical and often inside the container's height. The builder now uses side holds for containers
  taller than 10 cm (`hold_tilts`) and falls back to sampled, collision-checked poses in the container
  (`_sampled_pose_in_container`). Carry starts kept: hair 7 -> 100, stationery 59 -> 100, shelf 0 -> 85, counter 0 -> 88,
  rack 1 -> 90, tools 0 -> 85. The correction to the earlier finding: "no pose exists" was wrong.
- Accepted by the user 2026-11-17: the CPU-limit-only stages (stationery place, tools carry). 33 of 40 stages usable.
- Still failing (see validation.json): shelf place (61 starts, solver 10/20), counter place (52, 13/20), rack place
  (solver 1/20), tools carry / place and stationery place only on the 10 s CPU limit per start (max 11.5-15.6 s) and
  tools place 78 starts. Place starts on shelf / counter mostly drop at "idle: place" (38 and 18 of 100): the object
  does not stay in the place stage through the idle hold.
