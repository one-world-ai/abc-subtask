# put_relative — subtask rewards ledger

short name: `put_relative` · template: A (pick-and-place, one object) · seeds: SEED_BASE = 910_000_000 · baseline: none
status: DONE — reach, pick, **handoff (new, cross-table)**, carry (same-side + **cross-table**), place PASSED; end-to-end scene coverage 40 % → 89 % (2026-11-14)

## Latest verification — 2026-11-13 continuation
Fresh run of `tests/test_put_relative_subtasks.py`: **11/11 passed**. The five earlier failures came from the
shared test candidate pool, which contained only two same-side carry seeds while checks required three or four.
Kept all original candidates and added offsets 4 and 8 from the already validated same-side carry start set.
All existing assertions and required sample counts are unchanged; task/reward/start-generation code is unchanged.
Evidence: `../simulation/outputs/subtask_rewards/put_relative/seed_pool_regression.log`.
This closes the G2 regression noted in the preceding status review. Full-task cross-table carry still requires
a separate handoff; the previously approved same-side pilot scope is unchanged.


## Handoff amendment — 2026-11-14 (user approved adding a handoff for cross-table scenes)
Chain: reach → pick → handoff → carry → place; handoff only in cross-table episodes (zone point across the table:
171/290 usable seeds of SEED_BASE + 0..299), giver G = mover-side arm, receiver V = zone-side arm. Table:
`stage_table.md` (amendment). Tests: the shared `handoff_stage.py` **unchanged**, configured by `HandoffConfig`
(region = whole work space) and `PutRelativeScene.handoff_obs`. Code: `put_relative_{stages,subtask,starts,handoff}.py`,
spec `put_relative__handoff`. Tooling: `../simulation/experiments/subtask_rewards/put_relative/{build_starts,merge_starts,
relabel_starts,scripted_handoff}.py`, `probes/{cross_census,handoff_probe}.py`.

| phase | gate | result | numbers | evidence |
|---|---|---|---|---|
| 0 feasibility | – | probe | cross-table 171/290; giver top-down at the old grasp (site at the CoM) dropped 3/5 movers on the way to the middle (pads 3.1 cm off the CoM); pinch-centre grasps + tilted handoff poses: 7/9 scripted handoffs | `outputs/.../handoff_probe/{census_*,try0-4}.json` |
| 2 code + tests | G2 | PASSED | `tests/test_put_relative_subtasks.py` 15/15 (11 old + cross-table labels/progress, handoff failures, cross carry giver touch, scripted handoff +1); inhand 14/14 + 5/5 | tests |
| 2 regression | – | same-side unchanged | old start sets relabelled (labels renumbered only, states untouched; originals in `starts/pre_handoff/`); scripted reach/pick/carry re-run bit-identical to the 2026-11-13 summaries (0/260 rows differ); pick push_to_zone on cross seed +1 ends −1 two chunks later (same outcome); place: 8 rows on 2 cross seeds changed → place starts rebuilt (below) | `scripted/regression_handoff/` |
| 3 starts handoff | G3 | PASSED | 153/171 valid (first 100 eligible: 89), idle 10 chunks 153/153, CPU mean 2.6 s, max 8.7 s; 18 without a handoff plan | `starts/handoff_v1/handoff{,_g3}.*` |
| 3 starts cross carry | G3 | PASSED after 3 repairs | 156/171 valid (first 100: 92), idle 155/156, CPU mean 3.6 s, max 7.9 s; sources: 74 chained scripted handoff, 82 source 3 (V alone in the middle) | `starts/handoff_v1/carry_cross{,_g3}.*` |
| 3 starts place (rebuilt) | G3 | PASSED | 84/94 valid, 82 after idle (old: 84); 77 same-side starts bit-identical to the old set; 5 cross seeds now held by V (were held by G when the random zone point fell on G's side), 2 dropped | `starts/place.npz` (old in `starts/pre_handoff/`) |
| 4 handoff | G4/G5 | PASSED (G5 after 1 repair) | solver +1 14/20 (14/18 planned), mean return 0.90; drop_between −1 20/20; both_hold never +1 17/17 planned (return 0.14); release_stay never +1 17/17 (0.27); set_down_regrasp −1 20/20; giver_alone −1 16/20 (15 "giver carried it to the goal", 3 unplannable); idle 0 20/20; no hack +1; margin 0.63 | `scripted/handoff_round2/summary_handoff.json` |
| 4 cross carry | G4/G5 | PASSED (G5 after 2 repairs) | solver +1 20/20, return 1.17; release_early −1 20/20; regrasp_giver −1 19/20 (18 "back to handoff": G within 6 cm, 1 dropped), **1 +1** (seed +24: G's arm shoved V's hand 3.5 cm into the zone, G never within 15 cm); wrong_way/shake/idle 0 20/20; margin 1.17 | `scripted/carry_cross_round3/summary_carry.json` |
| 4 place (rebuilt set) | G4/G5 | PASSED | lower_release +1 20/20 (1.31); release_high −1 18/18; lower_hold 0 20/20 (0.28); leave_zone −1 19/19; idle 0 20/20; margin 1.03 | `scripted/regression_place_v2/` |

**End-to-end coverage** (usable seeds of SEED_BASE + 0..299 = 290): before: same-side only, 117/290 = **40 %**. Now:
same-side 117 + cross-table scenes with a valid handoff start *and* a valid cross-table carry start 141 → 258/290 =
**89 %**. Stricter (the scripted handoff physically completed for the scene, chained carry start): 117 + 89 → 71 %.

### Repair log (handoff amendment)
| gate | round | what failed | what changed |
|---|---|---|---|
| G3 cross carry | 1 | 7/171 starts > 10 CPU s (max 15.2) | chain budget 6 → 4 s → still 8 over (max 12.8) |
| G3 cross carry | 2 | fallback had no time cap | chain 3 s, 2 tries; fallback capped at 8.5 s → 3 over (max 12.6: resets before the timer) |
| G3 cross carry | 3 | timer started late | timer from `build_start` entry, cap 7.5 s, discard builds > 10 s → max 7.9 s, 156/171 valid |
| G5 handoff | 1 | giver_alone unplannable 9/20 (zone point at 14 cm beyond G's reach) | aim at zone points nearest the mover, two heights, tilts to 60° → 16/20 |
| G5 cross carry | 1 | regrasp_giver 5/20 closed on air (V's gripper in the way) | require a touch after closing, else next pose; also required a collision-free open gripper → 9/20 unplannable |
| G5 cross carry | 2 | over-strict pose filter | allow brushing V's hand, side grasps at 3 heights → 19/20, 1 +1 (5 %, ≤ 10 %) |

### Decisions (agent, for review)
- G = arm on the mover's side (pick-start arm), V = the other; cross-table = zone point's y sign differs (the rule the
  same-side carry restriction used). Zones along y can straddle the middle: handoff points inside the zone (+3 cm)
  are skipped, and G holding the mover inside the zone during handoff is −1 "giver carried it to the goal" (inhand
  decision 1 analogue) — so in cross-table episodes the giver may not finish the task alone even where it could reach.
- Handoff region = whole work space (the requested chain has no "present" stage); a middle-box term (+0.25) in
  handoff progress shapes bringing it to the middle. In the air only: table / other-object contact −1 (inhand
  decision 2). "Clear" = G's pinch centre ≥ 6 cm from the mover's collision **box** (exact distance).
- Cross-table place = V finger on it and G not touching; place starts of cross-table seeds held by V on its own half
  of the zone (V reaching over G's half sagged: idle return 0.055 on seed +1).
- **Grasp-site finding matters**: the old synthesized top-down starts put `<arm>_grasp_site` 1 cm below the CoM, so
  the pads hold movers 3.1 cm off-centre (by an edge). Same-side starts were left as they were (passed, semantics
  unchanged); handoff/cross-carry starts use pinch-centre (PAD) grasps. Consider rebuilding the old carry/place
  starts with PAD grasps (more realistic holds; changes those start sets).
- Labels renumbered (handoff = 2); saved pre-handoff start sets relabelled with `OLD_LABELS`.
- Start sets: handoff = first 100 valid cross-table seeds (first 100 eligible: 89 valid); cross carry likewise
  (`starts/handoff_v1/carry_cross.npz`); same-side carry stays `starts/carry.npz`. `reset_to_stage(seed)` on
  `put_relative__carry` now builds a cross-table carry start for cross-table seeds (was: error).
- Open for the user: (a) the 1/20 "giver shoves the receiver's hand" +1 in cross carry (same exploit kind as inhand
  carry) — a "G touches V's arm" failure would close it, not in the table; (b) 18/171 cross-table scenes have no
  handoff plan (flat/round movers); (c) rebuild old starts with pinch-centre grasps?
- Proposed (not made) `handoff_stage.py` change: none needed.

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASSED with findings (handled in the stage goals; stock scorer untouched) | 4/300 resets already successful; 3/300 resets raise; held-in-zone counts 8/8 | `probes/` | 2026-11-13 |
| 1 stage table | G1 | PASSED (approved by user; decisions 1–5 as proposed, baseline skipped) | | `stage_table.md` | 2026-11-13 |
| 2 code + tests | G2 | PASSED | tests 11/11; mug 9/9 + rebuilt starts bit-identical after kit changes | `tests/test_put_relative_subtasks.py`, `abc_sim/task_eval/put_relative_{stages,starts,subtask}.py` | 2026-11-13 |
| 3 starts | G3 | PASSED | carry rebuilt same-side only over 300 seeds: 119 eligible, 118 valid, 117 kept after idle. Valid of 100 seeds: reach 94, pick 92, carry 94 (all sides, superseded), place 86 (6 seeds unusable: reset raises / already solved); 0.6–1.1 s/start; idle 10 chunks in stage: carry 92/94, place 84/86 (≥ 90 %); the 4 that left were dropped from the sets → 94/92/92/84 | `../simulation/outputs/subtask_rewards/put_relative/starts/` | 2026-11-13 |
| 4 scripted | G4/G5 | PASSED (carry after restricting to same-side zones; user chose option 1). Carry same-side: solver 12/20 (12/12 planned), wrong_way 8/9, others all as expected, margin 1.19. Earlier all-sides run: | solver +1 of 20 starts: reach 14 (14/14 planned), pick 20, carry 5 (5/5 planned; 5/7 same-side, 0/13 zone across the table), place 19 (19/19 planned). Every hack as expected except 1/13 carry wrong_way (−1 "back to pick"). Solver − best hack mean return: reach 1.46, pick 1.02, carry 1.26, place 1.01. No hack reached +1 | `../simulation/outputs/subtask_rewards/put_relative/scripted/summary_*.json` | 2026-11-13 |
| 5 report | – | DONE | | `design.md` | 2026-11-13 |

## Phase 0 findings (stock `PutRelativeEvaluator`, randomizer `randomization/tasks/put_relative.py`)
- Success = the mover's xy offset from the *live* reference lies 6–30 cm along the prompted direction and ≤ 8 cm
  across it, and the mover is ≤ 3 cm above its spawn height. Nothing about the robot.
- **Held counts as placed**: a mover held by a top-down grasp 1.2 cm above the table in the zone scores success in
  8/8 seeds (`probes/pr_held.py`). Release and at-rest are not checked.
- **Free successes at reset**: 4/300 seeds (28, 175, 203, 222). The randomizer only rejects a prompt when the mover is
  within 5.5 cm + its radius of the goal *point* (14 cm off the reference), but the scorer's zone reaches 30 cm, so a
  mover two slots (24 cm) along the direction already succeeds (`probes/pr_reset.py`).
- **Reset crashes**: seeds 72, 156, 182 raise "could not place non-overlapping objects".
- Scene: 4–6 objects on a 3×4 slot grid (x 0.43/0.55/0.67, y ±0.18/±0.36), masses 12–30 g; directions right 91,
  behind 88, left 87, front 31 per 300.
- Top-down grasps (link_6 z down, grasp site at the CoM, yaw search) held the mover in 8/8 seeds → grasp source 3
  works; no recorded rollouts needed.
- Handling: starts skip crash seeds and already-successful seeds; the place goal adds release + at rest on top of the
  stock check. Changing the stock scorer/randomizer is left to the user (it changes benchmark episodes).

## Stages
| stage | status | why (if stopped) |
|---|---|---|
| reach | PASSED | |
| pick | PASSED | |
| handoff | PASSED (cross-table; 2026-11-14) | |
| carry | PASSED (same side; cross-table PASSED 2026-11-14) | Cross-table episodes (171/290 usable seeds) excluded from carry starts. Was: 13/20 start episodes have the goal zone on the other side of the table (y sign differs from the mover's); a top-down single-arm carry can't reach across (IK goal 9–15 cm out of reach). Same-side: 5/7 solved. Needs a handoff, or a same-side-only start set |
| place | PASSED | |

## Decisions (made with or for the user)
- Pushing the mover into place without a grasp: −1 in reach/pick (task says "put").
- Moving the reference > 3 cm: −1.
- Other objects: only off-the-table is −1.
- Prompts: per-episode, with the object names.
- Stock scorer and randomizer unchanged; subtask place goal adds release + at rest; starts skip bad seeds.
- Carry: start sets only where the zone is on the mover's side; cross-table episodes need a handoff subtask (user, option 1).
- No baseline run for the pilot (user: "currently we just want to build up substage rewards").

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
| G2 | 1 | idle in reach earned +0.29 | test bug: after reset ctrl targets are 0, so "repeat ctrl" moved the arm; idle = hold current joint positions |
| G2 | 2 | idle in place earned +0.055 / 10 chunks; one place start dropped its mover in chunk 1 | held movers creep through the pinch (median 0.4 mm/chunk, some 5 mm). Starts: grasp 1 cm below CoM (12/12 holds vs 9/12), reject a start whose object sinks > 1 mm in one extra chunk; pick pre-grasp stays at CoM height (lower hit the table) |
| G3 | – | (no repair) 2 seeds' movers slipped out later despite the creep check; 1 low place start (3.8 cm) slid out and landed in the zone → idle +1 | kept the gate; removed the 4 starts that failed the idle check from the sets |
| G4 | 1 | reach 7/20, carry 5/20 planned; idle reach +0.15; wrong_way/release_early behaviours mis-defined; place solver 13/17 | solver IK tolerance 8 mm / 0.2 rad (top-down near wrist limits); other arm holds its joints (ctrl is 0 after reset); wrong_way = away from the zone, release_early = before moving over |
| G4 | 2 | place solver dropped 4/19 at 4–4.6 cm | solver bug: read the mover's lift from ev._last, which reset_to_stage clears → 5 cm default, plans stopped 4.4 cm down. Read it from the tracker → 19/19. (A first guess, actuator ctrlrange clipping, was wrong: limits equal the joint ranges; `stagekit.actuator_limits` kept as an option) |
| G2 | 3 | place starts unreachable when the zone is across the table | place start uses the arm on the zone's side (carry across sides needs a handoff: watch in phase 4) |

## Continuation fixture repair
2026-11-13: expand the test candidate seed list from six to eight, preserving the original six (including the
intentionally rejected cross-table cases). Offsets 4/8 provide the two missing same-side carry starts. The
unchanged suite passes 11/11; no simulator or reward behavior was altered.
