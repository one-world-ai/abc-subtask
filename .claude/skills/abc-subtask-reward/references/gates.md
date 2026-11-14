# Gates

A gate is a script result, not a judgement. Record the numbers in STATUS.md. Thresholds are calibrated on mug flip
(numbers in brackets); don't loosen them to pass — at most 3 repair rounds of real fixes, then STOP.

| gate | after | pass when | mug flip |
|---|---|---|---|
| **G0 scene** | phase 0 | scene compiles; 20 reset seeds score 0 and success False; every audit probe behaves as expected (`abc-scorer-audit`) | mug_tree failed this twice before the handle rule |
| **G1 design** | phase 1 | user approved the stage table (or "auto-approved" recorded with the decisions) | user chose: placed-not-dropped, strict rotate, no frame history |
| **G2 code** | phase 2 | unit tests pass: (a) each built start classifies as its stage; (b) idle 10 chunks → no terminal, \|return\| < 0.05; (c) every listed failure fires −1; (d) one big move = 8 small moves (shaping totals equal ±1e-6); (e) snapshot restore is exact (state diff 0) | 9/9 |
| **G3 starts** | phase 3 | per stage ≥ 80/100 valid starts; ≤ 10 s per start; held-object stages: ≥ 90 % still in stage after 10 idle chunks (catches slow slips) | 99–100/100, ~2 s; slip check was missing — add it |
| **G4 solver** | phase 4 | per stage, the solver earns +1 on ≥ 40 % of 12+ starts (planning failures count as misses) and ≥ 70 % of the starts it could plan | reach 11/12, pick 8/12, rotate 5/12 (5/5 planned), place 11/12 |
| **G5 reward order** | phase 4 | every hack behaves as its EXPECT on ≥ 80 % of the runs it could plan, and could plan ≥ 10 of its 20 starts (report unplanned counts next to every result); mean solver return − max mean hack return ≥ 0.5; no hack gets +1 on > 10 % of **all** starts | all as expected except place lower-release back-off |

**Unplanned hack runs (user decision 2026-11-14).** A hack run with no IK plan never moved, so it says nothing about
the reward: the EXPECT share is computed over planned runs, with at least 10 of 20 starts planned. Applied
retroactively: clutter reach (11/12 planned), inhand reach (14/14), drawer close reach (15/15) pass under it.

## Stop rule
- A gate failing after 3 repair rounds → STOP that **stage** (later stages that need its end state for starts may
  still use sources 2–3). A task with at least one PASSED stage still ships those stages.
- Hard stops (no repair rounds): the scorer can't be fixed without redefining the task; the scene needs assets that
  aren't available; a stage needs a capability the robot doesn't have (e.g. reach beyond workspace).
- Write the stop as: stage, gate, numbers, the failure seen (with an npz/video path), what was tried.

## What counts as a repair
Fixing a bug, a threshold that contradicts measured physics (e.g. mug "near" 4 cm → 10 cm after measuring 576
grasps), a better start setup, a better solver plan. **Not** a repair: widening a goal, dropping a failure, raising
the hack ceiling, cutting the solver's success bar.
