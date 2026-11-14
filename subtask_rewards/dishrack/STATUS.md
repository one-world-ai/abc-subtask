# load_plates_into_dish_rack — subtask rewards ledger

short name: `dishrack` · template: F (insertion) · seeds: SEED_BASE = 1_300_000_000 (950M is nuts_bolts; recorded 2026-11-14) · baseline: exp12 canon 14B 3/60 recorded → **1/60 re-scored**
status: **STOPPED** (2026-11-14) — reach/pick STOPPED at G1 (approved); carry, insert STOPPED at G3; release STOPPED at G4/G5 (and G3 75/100). No stage ships.

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASSED after scorer fix | resets 0/40 score; 2 false positives in exp12 (lean on the rack's outside) fixed; re-score 60/60 bit-exact: 3/60 → 1/60; random settles 0 outside accepted after fix (94 before); tests 4/4 | `tests/test_dishrack_scorer.py`, `probes/dr_*.py`, `../simulation/outputs/subtask_rewards/dishrack/` | 2026-11-13 |
| 1 stage table | G1 | PASSED (user approved all 5 default options as written) | reach/pick STOPPED; carry/insert/release built from synthesized upright top-edge grasps | `stage_table.md` | 2026-11-14 |
| 2 code + tests | G2 | PASSED | `tests/test_dishrack_subtasks.py` 14/14 (starts classify, idle, every failure, 1 vs 8 moves, exact restore, stock-rule equivalence); scorer tests pass; mug flip 9/9; stagekit unchanged | `abc_sim/task_eval/dishrack_{geom,stages,subtask,starts}.py` | 2026-11-14 |
| 3 starts | G3 | FAILED → carry, insert STOPPED | valid/100: carry 65, insert 72, release 80 (before the G4 start check) → 75 (after) | `../simulation/outputs/subtask_rewards/dishrack/starts/` (+ `old_r0`, `r0`, `r1`, `r2`, `release_r_g3pass`, `starts_r*test`) | 2026-11-14 |
| 4 scripted | G4/G5 | FAILED → release STOPPED | release: solver +1 8/20 (8/16 planned, needs 70 %), yank −1 17/17 planned (3 unplannable), idle 0 20/20; margin 0.07 (needs 0.5). Carry/insert not run (stopped at G3). | `../simulation/outputs/subtask_rewards/dishrack/scripted/summary_release.json` | 2026-11-14 |
| 5 report | – | done | design doc | `subtask_rewards/dishrack/design.md` | 2026-11-14 |

## Phase 0 findings (`DishrackEvaluator`, randomizer `DishRackRandomizer`)
- **Scene**: 1–4 plates (21 models, 17.5–23 cm across, 1–5.5 cm deep, rim edge 0.5–2 cm above the table, 25–75 g), one of
  12 rack models (dish_rack_10 excluded upstream), scaled 0.95–1.05, rack 9–24 cm tall, footprint ≈ 28–50 × 20–46 cm.
  The rack is shifted −10..+9 cm in x and ±15 cm in y with ±0.25 rad yaw, and mirrored in y in half the scenes.
  Plates are spread over the other half of the table (`probes/dr_reset_geom.py`, `reset_geom.json`).
- **Stock rule**: every plate's origin inside the rack's xy box (built from bounding spheres, so 5–8 cm too large) plus
  a 5 cm margin, inside a height band, touching the rack, no gripper contact, \|normal_z\| ≤ 0.55. Checked at a
  single instant. It does not check whether the plate is over the rack or whether it is at rest.
- **Resets**: 40/40 score 0, no success.
- **exp12 re-score, bit-exact** (`rescore_replay.py`, all 60 worlds, no drift): recorded 3/60 (worlds 12, 49, 54).
  - world 12 (dish_rack_3) and world 54 (dish_rack_4): the plate leans on the outside of the rack's long side with its
    lower edge on the table, 5–7 cm outside the footprint. **False positives.** World 12 also slides past the upright
    limit within 200 steps. Renders: `replay_final/world-{12,54}_{final,hold}.png`.
  - world 49 (dish_rack_6): the plate is upright in a slot, pinched at its top edge. Success fired during a 50-step
    gripper-contact flicker while the plate was still held. When the grippers open in place and the arms retreat
    slowly (`probes/dr_w49_release.py`), the plate stays seated and scores for 400 steps. **Real insertion.**
  - New scorer: 1/60 (world 49). Mean final reward 0.113 → 0.079; mean max 0.136 → 0.103. Worlds 12/54 were cut short
    by the old false positive, so their new 0 is not a policy verdict.
- **Fix** (`abc_sim/task_eval/dishrack.py`): a plate also needs its centre and its lowest collision point inside
  the rack's tight footprint. The footprint is the rack-frame xy box of the rack's collision geometry (box corners /
  mesh vertices), plus 1 cm (`footprint_tolerance_m`). New metrics: `plate_over_rack_footprint_mask`,
  `plate_bottom_rack_xy`, `rack_footprint_xy`. Behaviour on true successes is unchanged.
- **Slot probe** (`probes/dr_slots.py`; one plate, plate_13 1.3 cm deep, every rack model, 6 lines per rack, upright
  plate lowered kinematically, released, settled 1.5 s): every rack model has positions where the released plate stays
  upright and scores (52–193 per model), with slot pitch 2–5 cm and play 0–35 mm. Old and new rules disagree on
  26 of ≈3600 releases. All 26 are plates that fell outward and rest against the rack's outside with the bottom
  2–9 cm beyond the footprint (renders `slots_v2/disagree/*.png`), so the new rule is correct there.
- **Random settles** (`probes/dr_settle.py`; 150 random drops per rack model, 1800 in all): upright, touching the
  rack and outside the footprint: 94 (old geometric rule accepts all 94, new rule 0). Upright, touching the rack and
  over the footprint: 86; the new rule accepts 65, and the other 21 touch the idle gripper and are rejected by both
  rules. The probe's "old" column omits the gripper check, so some of the 94 may also touch a gripper. Accepted
  poses include plates leaning diagonally inside the rack rather than in a slot (`settle/rack7_004.png`): see decision 2.
- **Gripper** (`probes/dr_fingers.py`): the jaws close along link_6 x, open 9.46 cm, 2 cm thick, pad z
  0.121–0.1465. The pinch centre is (0, −0.043, 0.134) in link_6, 4.3 cm off the axis that `stagekit`'s grasp-site
  constant assumes (harmless for bottles, matters for a 3 mm rim).
- **Pick feasibility** (`probes/dr_rim_grasp.py`):
  - Straight-down rim pinch: reaches the rim (≤ 6 mm error) and closes with both jaws in 3/4 scenes. Then:
    - 10 cm lift: plate pivots, far edge on the table, centre of mass +4 cm, ~35° tilt.
    - 16 cm lift: plate lost in 4/4 scenes.
    - 25 cm lift: out of IK reach in 24/24 attempts.
  - 30°/50° lean: the jaw heel hits the table; 0/48 attempts touched the plate.
  - No scripted pick lifted a plate clear of the table. Debug renders: `rim_debug/`.
- Other checkouts with the old scorer: `../simulation/abc`, the RoboEnv patch (not synced).

## Stages
| stage | status | why (if stopped) |
|---|---|---|
| reach | STOPPED (G1, approved 2026-11-14) | no grasp source: no working scripted pick from the table (see pick); 1/60 baseline picks to replay |
| pick | STOPPED (G1, approved 2026-11-14; capability stop) | rim pinch can't lift a flat plate clear: 0 lifts (3/4 pinch then drag, 0/4 hold at 16 cm, 0/48 tilted); needs a regrasp planner (separate project) |
| carry | STOPPED at G3 (2026-11-14) | 65/100 valid starts after 3 repair rounds (needs 80): in 30 seeds no upright held plate at or near its spot is in IK reach with a grasp that also reaches a slot entry. See repair log. |
| insert | STOPPED at G3 (2026-11-14) | 72/100 valid after 3 repair rounds: the slot-entry pose (plate 1.5 cm above the tines) is out of collision-free IK reach for the remaining seeds within the 10 s budget (14 exhaust the fallback search); 16-35 of the valid ones chain from the scripted carry |
| release | STOPPED at G4/G5 (2026-11-14) | after 3 repair rounds the scripted open-and-retreat keeps the plate in the rack on 8/16 planned starts (G4 needs 70 %); margin 0.07. Many kinematic-map slots do not hold a released plate; the start check that rejects them drops G3 to 75/100. |

## Decisions (made with or for the user)
- **G1 approved by the user, 2026-11-14: all default options as written in `stage_table.md`.**
  1. STOP reach/pick (no table pick; recorded above) and build carry/insert/release from synthesized upright top-edge grasps.
  2. "In the rack" = the fixed stock rule (upright, over the tight footprint, touching the rack, no gripper contact).
  3. One plate per episode, the nearest unracked plate; knocking a racked plate out = −1.
  4. Keep deep plates and bowls; starts may fail where no slot fits, and that counts under G3.
  5. Release must open and retreat with the plate staying seated.
- Seed base 1_300_000_000 (2026-11-14; 950M is taken by nuts_bolts).
- Judgement calls made without the user (conservative, for review):
  - Slots are measured kinematically per scene (`dishrack_geom.compute_slot_map`, cached per rack/plate geometry):
    upright plate lowered on 3 lines per rack axis; slot = it sinks >= 3 cm below the nearby rack tops. The insert
    goal's "seated" uses that map's seat depth at the plate's position (the table's "<= 1 cm above its depth from
    dr_slots"). The map does not release the plate (dr_slots did); the release-stage evidence shows many such slots
    do not hold a plate, so the insert goal is weaker than "in a slot" (open issue, insert is stopped anyway).
  - Carry progress uses the plate's bbox centre (≈ its bottom for an upright plate) for the "xy distance to the
    nearest slot centre"; no height term was added.
  - "Hit the table" counts only outside the rack footprint (an open-floored rack may let a seated plate touch it).
  - No stock evaluator is called per step (it rebuilds the footprint every call, ~2-3 ms); the scene computes the
    stock per-plate rule from cached rack-frame geometry, and a test checks it equals `DishrackEvaluator`.
  - Release starts are only kept if the plate stays in the rack with the jaws opened in place (round 2 above).
  - Grasp: pinch at the plate's top with the approach tilted 30-75 deg in the plate plane (round 2 of G3 carry).
  - Not applied (rounds used up): the release "left the rack" check fires on a transient loss of rack contact while
    the released plate settles (seeds 1300000002/06: back in the rack at the chunk's end). Requiring the loss for
    longer, or using only upright + footprint, is a likely fix for a few of the 7 solver failures, not for most.
- Jaw geometry: the pinch centre is link_6 (0, −0.043, 0.134) (`probes/dr_fingers.py`), not the stagekit grasp-site
  axis (0.1347 on the link_6 axis). Handled locally in `dishrack_stages.py` (`PINCH`); stagekit is unchanged.

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
| G0 | 1 | lean-on-outside plates scored (exp12 worlds 12, 54; 94 random settles) | footprint rule (centre + lowest point over the tight rack footprint); `tests/test_dishrack_scorer.py` |
| G3 carry | 0 | 66/100 valid: 12 unbuilt, 16 idle \|return\| >= 0.05, 10 over 10 s | – (first build: rim pinch at theta 30-75 deg off the top, radial approach; `starts/old_r0/carry.json`) |
| G3 carry | 1 | pre-settle 2 chunks before saving: 23/40 valid (worse) | reverted (`PRESETTLE_ACTIONS = 0`) |
| G3 carry | 2 | in-hand creep (gravity torque about the jaw axis, 1-2 deg/chunk) caused the idle leak and slips | pinch at the plate's top (equilibrium) with the approach tilted 30-75 deg in the plate plane (`edge_grasp(beta=)`): 30/40 valid, idle in stage 35/35, leak 1/35 (`starts_r2test/`) |
| G3 carry | 3 | over-budget builds (per-attempt timer); start-pose IK | one CPU deadline per build; start 3-6 cm tried (29/40 built, worse) and kept at 6-10 cm. Full run: **65/100 valid** (70 built, idle in stage 70/70, leak 5, over budget 6) → STOP (`starts/carry.json`) |
| G3 insert | 0 | 57/100 valid: slot fallback searched only the first slot (IK cap inside the per-slot loops); 15 over budget | – (`starts/r0/insert.json`) |
| G3 release | 0 | 67/100 valid, idle in stage 83/83; 22 over budget | – (`starts/r0/release.json`) |
| G3 insert/release | 1 | (see above) | fallback combinations shuffled over all slots, nearer arm first; release chains only if its carry start was found within 2.5 s → insert 69, release 75 (`starts/r1/`) |
| G3 insert/release | 2 | over-budget builds: the chained carry-start search left no time for the fallback | carry-start search capped at 3 s / 2.5 s when chaining → insert 72, **release 80 (G3 passed)** (`starts/r2/`, `starts/release_r_g3pass/`) |
| G3 insert | 3 | 14 seeds exhaust the fallback search (reach) | chain only if its carry start is found in 2.5 s, chained carry capped at 4 candidates → **72/100** → STOP |
| G2 | – | jam fired on a plate resting on the tines during idle (insert starts 31/85 valid in a first build) | the table's force condition was missing: jam now needs plate-rack normal force > 2 plate weights (a fix to the table's definition, not a new threshold) |
| G4/G5 release | 1 | solver +1 5/20: released plates that lean in the slot's play failed my "seated" test | release failure/goal made exactly the table's: failure = falls over (\|nz\| > 0.55) or leaves the stock in-rack geometry; goal = stock success + still + jaws ≥ 5 cm (decision 2). Yank re-planned (straight up unplannable 8/20) → solver still 5/20, 11 fall out |
| G4/G5 release | 2 | plates fall out of kinematic-map slots when the jaws open (the map checks sinking, not holding) | release starts kept only if the plate stays in the rack when the jaws open in place for 1 chunk (scratch copy): G3 → 75/100; solver 8/20 (8/16 planned), margin 0.11 |
| G4/G5 release | 3 | 7 planned runs still fail after opening | gentler retreat (open 2 chunks, 1.5 cm slow back-off, then 10 cm): unchanged 8/20, margin 0.07 → STOP |
