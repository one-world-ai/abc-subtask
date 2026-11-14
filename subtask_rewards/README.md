# ABC subtask rewards — task queue

**Using the rewards for RL: see [QUICKSTART.md](QUICKSTART.md)** (usable stages, minimal loop, saved starts, paths).

Pipeline (paths relative to the abc repo root): `.claude/skills/abc-subtask-reward` (scorer checks: `.claude/skills/abc-scorer-audit`).
One ledger per task: `subtask_rewards/<short>/STATUS.md`. Update the status column when a task changes state.

Shared kit: `abc_sim/task_eval/stagekit/` (extracted 2026-11-13; mug flip verified bit-identical).

| # | task | short | template | baseline (exp12, canon 14B) | status (2026-11-13 evening) |
|---|---|---|---|---|---|
| 1 | turn_mug_right_side_up | mug_flip | B | 20/60 | IMPLEMENTED (reference); pick G4 8/12 and G5 margin 0.38 below current gates; G3 idle check not recorded |
| 2 | put_relative | put_relative | A + D | none | DONE — 5 stages (handoff for cross-table scenes), end-to-end coverage 89 % |
| 3 | put_plastic_bottles_in_bin | bottles | A | none | DONE — all gates; starts 99/83/96/82 |
| 4 | sort_lego_blocks | lego | A + D (handoff) | 1/10 | reach/pick/handoff/carry/drop PASSED (2026-11-14): handoff for cross-side scenes (solver 14/20, margin 0.82), carry 19/20, drop 17/20 on both sides; end-to-end start coverage 100/100 scenes (was 52.7 %) |
| 5 | grab_specific_object_from_clutter | clutter | A | none | reach → pick → [handoff] → carry → place: handoff/carry/pick/place PASSED (handoff added 2026-11-14; carry 98/100 starts, end-to-end 90/100 scenes); reach PASSED by user-accepted G5 exception |
| 6 | sort_nuts_and_bolts | nuts_bolts | A | 0/10 | all PASSED: pick/carry; reach (user G5 exception: push hack 15/20 as expected, 0 hack +1); drop amended to low release only (≤ 5 mm above floor, else −1 "released too high"): solver 15/20, 0 numerical, 0/100 hack +1, margin 0.80; **handoff added 2026-11-14** for cross-side parts (all bolts, 15.7 % of seeds): handoff 16/20 margin 0.88, cross carry 20/20 margin 1.33, cross drop 15/20 margin 0.91, 0 hack +1; cross-side E2E chain 0/30 → 15/30 |
| 7 | hang_mug_on_mug_rack | mug_rack | F | 0/10 | DONE 2026-11-14 — pick PASSED (starts 87/100, solver 18/20, margin 0.97); reach STOPPED by the letter of G5 (push touches = approved goal); align/insert/release STOPPED at G3 (66/36/51 of 100 after 3 rounds: hang-orientation pinches slip, build time); seeds 1200M |
| 8 | open_drawer / close_drawer | drawer | C | none | DONE 2026-11-14: open reach/grasp/release + close reach/release PASSED; open pull + close push STOPPED at G5 (two-arm wrong-drawer hack not executable; user may accept) |
| 9 | sweep_away_paper_scraps_from_table | sweep | E | 10/60 → **7/60** with fixed scorer | chain H (hand pick, user-approved 2026-11-14) DONE — reach/pick/carry/drop all PASSED; starts 100/93/89/85; solver 20/20/15/20; tool chain deferred |
| 10 | inhand_transfer_item_to_other_side | inhand | D | 1/10 (real handoff) | **PARTIAL** (2026-11-14) — reach/pick/present/handoff/place PASSED (reach under the planned-runs G5 rule); carry STOPPED at G5 (giver shoves the receiver's gripper: +1 on 2/20; fix on hold). Task-independent `handoff_stage.py`; grasp site found 3 cm off the finger pads
| 11 | load_plates_into_dish_rack | dishrack | F | 3/60 → **1/60** with fixed scorer | STOPPED (2026-11-14) — reach/pick at G1 (no table pick); carry G3 65/100, insert G3 72/100 (IK reach of upright held plates); release G4/G5 (scripted release keeps the plate in the rack on 8/16 planned; margin 0.07). Code + 14 tests in place. |

Later: put_markers_in_{top,middle,bottom}_drawer (needs a scorer; after drawer). Out: conveyor_pick, multi_drawer_search,
chess, pouring, count_*, spell_*, ball_tray, build_wood_block_tower.

## Status review 2026-11-13

- **Agreed execution:** bottles implementation/validation in parallel with LEGO, clutter, and nuts/bolts
  scorer audits and stage drafts, using separate worktrees. Bottles is now DONE: starts valid/100 are
  100/85/96/82; scripted solvers succeed on 13/20, 20/20, 17/20, 20/20; all reward-order gates pass;
  bottle tests 14/14, mug regression 9/9. Release above the opening is allowed, with supported rest inside
  required. See `bottles/STATUS.md` and `bottles/validation.json`. Other three stage tables need explicit approval
  before parallel stage implementation; the bottle approval does not extend to their task semantics.
- **Parallel audits complete:** LEGO/clutter scorer fixes integrated; main-checkout regressions pass 7/7 and 5/5.
  LEGO ten-world replay is bit-exact, success still 1/10, final reward mean 0.891→0.908. Nuts/bolts regressions pass
  4/4; ten-world replay is bit-exact with 4,410 unchanged checks and success still 0/10. Every nuts/bolts replay
  emits a reproducible numerical-instability warning; see its ledger before building starts or RL workers.
- **Mug flip:** fresh `tests/test_mug_flip_subtasks.py` passes 9/9. The saved scripted results and starts predate the
  current skill gates: no recorded G3 10-chunk idle check over the held-start sets; pick solves 8/12 planned (below
  G4's 70 %), solver mean return 0.602 vs close-only 0.222 (G5 margin 0.380 < 0.5); rotate wrong-way behaves as
  expected on 4/7 planned. Keep it as the implemented reference, but do not claim all current gates pass.
  Evidence: `../simulation/outputs/mug_stages/{starts,scripted}/`, `../mug_substages.md`.
- **put_relative:** fresh `tests/test_put_relative_subtasks.py` passes 6/11. All five failures report only 2 of 6
  candidate seeds building a start; the test requests 3 (or 4). Four candidates are now intentionally excluded from
  carry by the same-side restriction. The saved same-side solver/start results remain available, but G2 needs a
  candidate-pool repair and rerun before restoring DONE. See `put_relative/STATUS.md`.
- **Mug rack:** fresh `tests/test_mug_tree_scorer.py` passes all 6 checks, including accepted handle hangs and
  rejected cup/rim rests. This verifies the scorer regression cases; mug-rack subtask rewards remain queued.

## Next approval gate after parallel audits

No LEGO, clutter, or nuts/bolts stage implementation has been authorized by approving the bottle design. Their
concrete draft tables are ready for review; numerical scorer/audit evidence is in each task's ledger.

| task | proposed defaults to approve | full table |
|---|---|---|
| LEGO | reach/pick/carry/drop; nearest unsorted target; matching-color bin; release above opening allowed; wrong bin, pushing, bin disturbance or loss of sorted contents fails; measure carry coverage before handoff/exclusions | [LEGO](lego/stage_table.md) |
| clutter | reach/pick/carry/place; prompted target; controlled lowering and floor support; wrong-object lift fails; harmless distractor nudges allowed but off-table loss/box disturbance fails; direct reachable-carry pilot with explicit coverage | [clutter](clutter/stage_table.md) |
| nuts/bolts | reach/pick/carry/drop; nearest unsorted part; fixed nuts→teal/bolts→orange; release above assigned opening allowed; wrong bin/push/bin disturbance/loss of sorted contents fails; retain and measure cross-table coverage before deciding on handoff | [nuts/bolts](nuts_bolts/stage_table.md) |

Each drop/place goal additionally requires release, real rest, and physical support rooted in the fixture. All
three use the verified progress reward and 32-action chunks; mid-task starts have no frame history. The next
implementation round can run in parallel once the corresponding G1 table is approved. Keep shared `stagekit`
changes coordinated and reconcile registration blocks during integration.

## Continuation approved — 2026-11-13
The user approved the three proposed stage tables with “lets continue”. LEGO, clutter, and nuts/bolts now
proceed through G2–G5 in parallel in their isolated worktrees, using the stated defaults. The approval-pending
statements in the earlier review above are historical. Shared stagekit changes remain coordinated.

## Existing pilot regression repaired — 2026-11-13
Put-relative now passes 11/11 existing tests after adding two validated same-side candidates to the stale seed
pool. All original cases/assertions remain; no reward/start-generation code changed. Earlier 6/11 review results
above are historical. Carry remains the approved same-side pilot, without cross-table handoff.

## Parallel round — 2026-11-13 afternoon
One agent per task in worktrees `../abc-audit-{lego,clutter,nuts-bolts}` and `../abc-subtask-{mug-rack,drawer,sweep,inhand,dishrack}`
(branches `subtask-<task>`), merged into `substage-mug`. Reports and open decisions are in each task's `STATUS.md`.
Seed bases: lego 930M, clutter 940M, nuts_bolts 950M, sweep 970M, drawer 980M/985M, inhand 1100M, mug_rack 1200M — check the ledger before choosing.
Scorer fixes live only in this abc checkout (user scope rule: subtask work stays in abc). G5 counts hack EXPECT shares over planned runs (≥ 10/20 planned), user decision 2026-11-14 — see the skill's gates.md.
Review videos: `../simulation/experiments/subtask_reward_space/{render_scripted,build_task_page}.py`.
