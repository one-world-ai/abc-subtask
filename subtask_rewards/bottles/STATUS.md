# put_plastic_bottles_in_bin — subtask rewards ledger

short name: `bottles` · template: A (pick-and-place, one bottle at a time) · seeds: SEED_BASE = 920_000_000 · baseline: none for this task (exp12's 7/10 = 41/60 is `throw_plastic_bottles_in_bin`, a different scene — see phase 0)
status: DONE — all four stages PASSED; CPU single-world, progress reward, 32-action chunks

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASSED with findings (put scene) | resets 0/40 score; 0 crashes; settle 900 drops: in-bin ⇔ inside, 9 leaning-on-rim count (CoM ≤ 2.3 cm over rim, accepted); held over the bin counts 4/4 | `probes/bt_{reset,settle,held,reach}.py`, `probes/settle_seed*.{log,json}` | 2026-11-13 |
| 1 stage table | G1 | PASSED (user confirmed next task and draft drop semantics) | reach → pick → carry → drop; release above the opening allowed, settle inside required | `stage_table.md`; user confirmation in status-review conversation | 2026-11-13 |
| 2 code + tests | G2 | PASSED | final bottle tests 14/14; shared stagekit unchanged; mug regression 9/9 | `tests/test_bottles_subtasks.py`; `../simulation/outputs/subtask_rewards/bottles/{tests,mug_regression}.log`; `validation.json` | 2026-11-13 |
| 3 starts | G3 | PASSED all stages | valid/100: reach 100, pick 85, carry 96, drop 82. Held idle: carry 96/96, drop 82/84; all mean construction times <4 s | `../simulation/outputs/subtask_rewards/bottles/starts/*.{json,npz}` | 2026-11-13 |
| 4 scripted | G4/G5 | PASSED all stages | solver +1/20: reach 13, pick 20, carry 17, drop 20; planned successes: 13/14, 20/20, 17/17, 20/20. Solver-minus-best-hack mean return: 1.029, 1.0125, 1.207, 1.000; no hack +1; each hack meets EXPECT on ≥85% | `../simulation/outputs/subtask_rewards/bottles/scripted/summary_*.json` | 2026-11-13 |
| 5 report | – | written | CPU, single-world only; no RL model run | `design.md` | 2026-11-13 |

## Phase 0 findings (stock `PutBottlesInBinEvaluator`, randomizer `WaterBottleRandomizer`)
- Scene: 2–6 RoboCasa water bottles (50 g, lying flat, Ø ≈ 5 cm, 18–24 cm long), tapered tub 23–27 cm tall
  (rim at z ≈ 0.99–1.02, table 0.75), top radius 10–12 cm (scale 1.0–1.4), centre x ≈ 0.57–0.64, y ≈ 0.04–0.13.
  Counts over 40 seeds: 2:8, 3:4, 4:7, 5:8, 6:13.
- Success = every bottle's CoM inside the measured cone, up to rim + the bottle's reach (~10 cm). Nothing about
  the robot or rest.
- Resets: 40/40 score 0, no crash (`bt_reset.py`).
- Random settles (one bottle dropped over the bin, 3 seeds × 300 tries, 900 kept): every bottle resting inside is
  in, every one on the table out; 9 leaning on the rim with CoM 0.5–2.3 cm above it count (accepted: bottom inside).
  No false positive found (`bt_settle.py`).
- **Held counts**: a bottle held 3 or 8 cm above the rim scores in-bin (4/4 holds). → the stage goal adds released +
  at rest (as put_relative).
- **Reach**: above the rim, a straight-down gripper is out of reach (IK error 6–16 cm at grasp-site z 1.05–1.15);
  leaning 40–60° reaches (`bt_reach.py`). Synthesized leaning holds over the bin: 4/20 (`bt_held.py`; IK and
  slips). → carry/place starts need a better grasp source (chain from the pick solver, or leaning grasps).
- **Baseline mix-up**: the README's 41/60 is exp10/12 on `throw_plastic_bottles_in_bin` (Blender worlds: 4 baked
  bottles, 18 cm bin, rim ≈ 0.94, `BottlesInBinEvaluator`), with recorded traces. The put scene has no baseline.

## Stages
| stage | status | why (if stopped) |
|---|---|---|
| reach | PASSED | |
| pick | PASSED | |
| carry | PASSED | |
| drop | PASSED | |

## Decisions (made with or for the user)
- User confirmed: continue `put_plastic_bottles_in_bin` next, then `sort_lego_blocks` (2026-11-13).
- Use the requested **put** scene; the throw scene's 41/60 baseline does not apply.
- User selected **allow dropping from above the opening, as in the current draft**; do not require lowering before
  release. The target must settle inside the bin, with no finger contact; release outside the opening is rejected.
- Approved draft defaults: one not-yet-binned target nearest a gripper; reach → pick → carry → drop; bin moved/tipped,
  pushing the target, or knocking a previously binned bottle out are failures as listed in `stage_table.md`.

## Next work
The verified bottle pipeline can now serve as the next pick-and-place reference.
LEGO, clutter, and nuts/bolts audits/designs run independently; their stage tables require separate G1 approval.
No policy training has been launched. For RL integration use the CPU `evaluate_model_data` path; direct qpos-only
evaluation is rejected because real velocities and contacts are required.

## Validation and start-source scope
The 82 drop starts include successful physical carry chains and fallback sampled over-bin grasps checked under
gravity. Fallback starts do not establish an end-to-end chain from every carry start: the physical carry solver
is separately 17/20 including planning failures. Original bin anchors and bottle rest positions survive snapshot
restore. Fixture-rooted support permits real settled stacks while rejecting unrooted airborne contact cycles.
The final start-set build includes a fresh 100-seed run of every stage after the goal/failure fixes.

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
| G3 | 1 | top-down source: pick 71/100, carry 78/100, below 80; many wrist orientation misses | try equivalent gripper yaw +π as well as the principal-axis yaw; final 85/100 and 96/100, idle stable |
| G3 drop / G4 carry | 1 | positive leaning path was unreachable; carry solver 0/20 | plan IK waypoints continuously from previous solutions; retain actual physical execution |
| G3 drop / G4 carry | 2 | single leaning direction: drop chain 49/100; carry solver 5/20 | try both leaning signs and several approach angles; carry solver 17/20 |
| G3 drop | 3 | chained starts still below 80/100 | use allowed grasp source 3 after chain failure: sampled over-bin poses plus gravity/idle checks; 84 built, 82 retained |
| G2 | 1 | original test used an absolute bin quaternion equal to its normal upright pose; pick test began from an invalid plain reset | compose tip relative to original orientation; explicitly begin the tested pick state before its adversarial transition |
| G2 | 2 | independent review: airborne bottles could support one another; regrasp bypassed release check; low-speed flight looked landed; safe lowering regressed | fixture-rooted stable support graph; validate every release; require landing contact; retain valid in-bin lowering; regression cases added |
| G5 pick | 1 | push policy often did not push: only 10/20 expected failures | push along the gripper closing axis rather than a fixed world direction |
| G5 pick | 2 | one-finger push jammed near table: 13/20 expected failures | raise open jaws 1 cm before pushing farther and waiting for motion; 17/20 expected, no +1; three planning misses count as misses |
