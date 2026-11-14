# sort_nuts_and_bolts — subtask rewards ledger

short name: `nuts_bolts` · template A · SEED_BASE = 950_000_000 · baseline: exp12 0/10, mean final 0.026,
mean max 0.148 (`../simulation/experiments/experiment12/RESULT.md`; replayed bit-exactly, see below).
status (2026-11-14): **all five stages PASSED** — pick, carry PASSED; reach PASSED (user exception to G5);
drop PASSED after the user's low-release amendment (see "2026-11-14 decisions"); **handoff (cross-side) PASSED, and
cross-side carry and drop PASSED** (see "Handoff amendment"; handoff outputs in `$O/handoff/`). Design doc: `design.md`. Outputs: `../simulation/outputs/subtask_rewards/nuts_bolts/` (= `$O` below).

## Phases

| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASSED with required guards (release/rest/support/bin stability, implemented in the stage goal) | 40/40 resets reward 0; 154 valid physical drops: floor-supported 64/64 accepted, wall/rim-only 0/27, outside 0/58; floating / tipped synthetic layouts accepted by the raw scorer (hence the guards); 10/10 exp12 worlds replay bit-exactly, 4410/4410 scores unchanged | `probes/` | 2026-11-13 |
| 1 stage table | G1 | PASSED — user approved | see Decisions | `stage_table.md` | 2026-11-13 |
| 2 code + tests | G2 | PASSED | 13/13 unit tests (incl. numerical-fault latch, floating contact cycle, real velocities, lowered-inside-opening) | `$O/g2_final.log` (13/13, code of commit a4f365d; 2c8331b only changes the start budget clock) ; earlier `g2_round0.log` 11/11, `g2_latency_cap.log` 12/12 | 2026-11-13 |
| 3 starts | G3 | PASSED all four | reach 100/100 (max 2.3 s), pick 100/100 (max 5.2 s), carry 100/100 (max 5.8 s, idle-kept 100 %), drop 90/100 (max 9.3 s CPU, idle-kept 90/92 = 97.8 %) | `$O/final/starts/*.{npz,json}`, `$O/final_*_starts.log` | 2026-11-13 |
| 4 scripted | G4/G5 | see per-stage table | | `$O/final/scripted/summary_<stage>.json`, per-run traces `$O/final/scripted/<stage>/` | 2026-11-13 |
| 5 report | – | written | | `design.md` | 2026-11-13 |

## Per stage (final run, first 20 saved starts; reach/pick/carry code at 2c8331b, amended drop at c672721)

| stage | starts valid | solver +1 (planned) | solver mean return | best hack mean return | hack +1 | G4 | G5 | state |
|---|---|---|---|---|---|---|---|---|
| reach | 100/100 | 16/20 (19) | 1.26 | 0.006 (idle) | 0/80 | PASS | FAIL: push_reach as expected 15/20 = 75 % (< 80 %) — accepted by the user as a documented exception | **PASSED (user exception)** |
| pick | 100/100 | 15/20 (20) | 0.75 | 0.247 (close_only) | 0/100 | PASS | PASS (margin 0.500, at the limit) | **PASSED** |
| carry | 100/100 | 12/20 (12) | 0.70 | −0.000 (idle) | 0/120 | PASS | PASS (margin 0.70) | **PASSED** |
| drop (amended, 2026-11-14) | 91/100 | 15/20 (20); 0/20 numerically invalid | 0.81 | 0.008 (idle) | 0/100 | PASS | PASS (margin 0.80) | **PASSED** (amended stage, repair 1 of 3) |
| drop (original, superseded) | 90/100 | 4/20 (20); 14/20 numerically invalid | −0.34 | 0.008 (idle) | 0/80 | FAIL | FAIL (margin −0.35) | STOPPED, superseded by the amendment |

All hacks of pick, carry and drop behaved as expected 20/20. No hack earned +1 in any stage.

### Reach — G5 exception (was: Stop)
- Numbers: solver 16/20, margin 1.26, no hack +1; but `push_reach` (fast one-finger strike) as expected 15/20:
  15 × −1, 3 × 0 (strike missed), 1 unplanned (IK), 1 numerically invalid. All 15 −1 are **"bin moved or tipped"**
  (the arm strikes the bin), none is "pushed" (`first_push` never set). Traces:
  `$O/final/scripted/reach/push_reach_9500000*.json`.
- Why the hack can't test what the table says: the reach goal is "≤ 5 cm or finger touch", so any push touches the
  part first and earns +1 after the 5-action debounce unless the part moves > 3 cm within those 5 actions. Round 1
  (slow push): 16/20 +1 — a legal reach, not a hack. The reward itself showed no exploit.
- Tried: round 2 ballistic strike (13/20), round 3 rewrite (0/20 planned), round 4 rapid strike with event ordering
  (15/20). Reward code unchanged in all rounds. Budget exhausted.
- User decision (2026-11-14): accept reach with this documented G5 exception → reach PASSED.

### Original drop stop — G4 (superseded by the 2026-11-14 amendment)
- Numbers: solver 4/20 +1, 14/20 **numerical instability** (latched −1 by the approved guard), 1 × 0, 1 × "released
  outside the opening". Traces `$O/final/scripted/drop/solver_*.json`.
- Diagnosis (`$O/round5_release_diagnosis.txt`, scripts `trace_release_qacc.py`, `probe_floor_release.py` in
  `../simulation/experiments/subtask_rewards/nuts_bolts/`): not planning (20/20 planned) and not a slip. While held,
  finger pads penetrate the nut by up to 3.6 mm. When the fingers open in free fall the last finger contacts give
  the nut (6 small boxes, tiny inertia) a huge kick: one step later it has **no contacts and 53,348 rad/s angular
  speed**, MuJoCo raises BADQACC on the nut's own free joint and resets the state (the nut reappears on the table).
  This is the same warning seen in every exp12 replay.
- Predicate bug found and fixed on the way (round 5, commit a4f365d): the rim is 0.5–0.7 cm above the scorer box
  top, so lowering a held part > 1 cm below the rim regressed it to carry (round 4: 9/20 "back to carry"). Fixed with
  a drop-only column inside the walls; carry entry unchanged; unit test added. After the fix 0/20 "back to carry".
- Diagnostic only (not a gate run, after budget): lowering until the nut is 2 mm above the floor and opening over
  160 actions gave +1 on 8/12 starts whose solver run was numerically invalid (3 still invalid, 1 no terminal).
- Tried: round 3 and 4 carry/plan fixes (planned 4 → 8 → 13 of 20), round 5 predicate fix + release from 1.5 cm
  above the floor (planned 20/20, but 15/20 invalid). Budget exhausted.
- User decision (2026-11-14): amend drop to allow only a low release (see "2026-11-14 decisions"); scene numerics
  stay open (kit/scene proposal 2). Note: any RL policy that releases a nut
  in the air will mostly hit the same −1 numerical failure, so (b) matters for training, not just for the gate.

### Cross-table coverage (approved: retain and measure before deciding on handoff)
- The nearest-part rule picks a nut on the bins' side in 87/100 seeds and a bolt in 13/100; **every bolt target is
  cross-table** (on the other side from the orange bin) and no nut target is.
- Carry solver: 4/4 cross-table bolt starts in the first 20 are IK-unreachable for the holding arm (plus 4 same-side
  nuts unreachable). Carry still passes G4 (12/20) only because cross-table is 20 % of the 20.
- Drop starts: the 6 cross-table bolts come only from the sampled fallback that uses the bin-side arm — effectively an
  unmodelled handoff. Valid drop sources: 46 physical carry chain, 44 sampled lowered hold.
- User decision: add a handoff stage, or scope carry/drop to same-side targets (which would remove bolts entirely).

## Handoff amendment (user approved adding a handoff, 2026-11-14)

**Merge:** `git merge substage-mug` was a fast-forward (substage-mug already contained this branch). Chain
reach → pick → handoff → carry → drop; handoff only for cross-side scenes. Stage table amendment: `stage_table.md`
(top). `handoff_stage.py` is reused unchanged (configured via `HandoffConfig` + `NutsBoltsScene.observe` arrays); no
change to it is proposed. Outputs `$O/handoff/` (= `$H` below); tooling in
`../simulation/experiments/subtask_rewards/nuts_bolts/` (`screen_cross.py`, `e2e_chain.py`, `remap_starts.py`, and the
updated `build_starts.py`, `scripted_rollouts.py`; that directory is untracked in ../simulation, as before).

### Coverage measured first (before the handoff)
- Screen of 1000 seeds (`$H/cross_screen.json`): **157 cross-side targets, all bolts** (bolts on the +y side, orange bin
  at −y); 0 nuts; the near arm is never the bin-side arm. First 100 seeds: 13 cross-side.
- Stage level (old final scripted runs, first 20 starts, 4 cross-side): reach 2/4, pick 4/4, **carry 0/4 (IK
  unreachable for the holding arm)**; the 6 valid cross-side drop starts came only from the bin-side-arm fallback.
- Physical end-to-end chain (`e2e_chain.py`: reach → pick → carry → drop solvers in one episode, covered = target
  DONE with no MuJoCo warning), old code: first 100 seeds **0/100** (same-side 0/87: the single-stage solvers do not
  compose — 18 numerical blow-ups at the nut pick, 26 carry IK, 14 reach misses); first 30 cross-side seeds **0/30**
  (17 reached carry, **0/17 carry planned**). `$H/before/e2e_first100.json`, `$H/before/e2e_cross30.json` (old code
  run from a `git archive f2974e3` copy).

### Physics findings (probes, `$H/`, scratch scripts not kept)
- Bolts **stand on their heads**; a top-down grasp centred on the CoM encloses the whole bolt between finger plates
  6.4 cm wide (link_6 y −0.050…0.014) and 8 cm long — nothing sticks out to hand over. A grasp high on the shank
  (CoM at link_6 z ≥ 0.150, fingertips at 0.147) leaves 2–3 cm (head + lower shank) below the fingertips.
- Receiver from the side (link_6 horizontal, plates down, pad strip 3 mm under the giver's fingertips): the bolt
  hangs along its axis and stays held when the giver opens (sag ~1 mm, no warning). Horizontal presentation (bolt
  axis horizontal, receiver top-down on the head stub): the bolt pivots out within 8 actions. Flipping the giver's
  wrist (fingers up) is out of reach.
- Low release from the side grasp: the receiver rolls 180° about its approach axis (bolt head-up, shank 0.6–2 cm
  below its fingers), carries turning about the vertical, and pitches the hand down about its closing axis once the
  hand meets the rim.
- Numerics: **0 numerical-instability trials in 380 scripted cross-side trials** (handoff 120, carry 140, drop 120)
  and none in the start builds' idle checks. Nuts never reach the handoff (none is cross-side), so the nut release
  blow-ups are not exercised by it; the guard is unchanged.

### Per stage (final run `$H/run_final.sh`, first 20 starts, code of this commit)

| stage | starts valid/100 (first 100 cross-side seeds) | solver +1 (planned) | solver mean return | best hack mean return | hack +1 | G3 | G4 | G5 | state |
|---|---|---|---|---|---|---|---|---|---|
| handoff | 100 (max 3.9 CPU-s, idle-kept 100 %) | 16/20 (18) | 1.084 | both_hold 0.208 | 0/100 | PASS | PASS | PASS (margin 0.876; all hacks ≥ 85 % as expected) | **PASSED** (G5 repair 1) |
| carry, cross-side | 100 (max 3.8 CPU-s, idle-kept 100 %) | 20/20 (20) | 1.330 | idle 0.002 | 0/120 | PASS | PASS | PASS (margin 1.328; all 20/20 as expected) | **PASSED** (G4 repair 1, G5 repairs 2–3) |
| drop, cross-side | 93 (max 4.4 CPU-s, idle-kept 93 %) | 15/20 (20) | 0.927 | idle 0.018 | 0/100 | PASS (repair 3) | PASS (repair 1) | PASS (margin 0.908; all 20/20 as expected) | **PASSED** |

Handoff hacks (20 each): idle 0 20/20; drop_between −1 20/20; both_hold 0 17/20 (2 unplanned, 1 dropped −1), never
+1, return 0.21; set_down_regrasp −1 20/20 (11 "touched the table", 9 "back to pick" while lowering); giver_alone
−1 20/20 (carried towards the bin in 4 cm steps, opened where its reach ended: 18 dropped, 2 touched the table).
Solver misses: 2 × no receiver grasp under the giver (unplanned), 2 × dropped while the receiver closed.
Cross carry hacks: release −1, wrong_way 0, shake 0, wrong_bin −1 ("released outside the opening"), move_bin −1,
idle 0, all 20/20. Cross drop hacks: release_beside, wrong_bin, tip_bin, release_high −1, idle 0, all 20/20.
Cross drop solver misses: 5 × released too high (the bolt slipped while the hand pitched down).
Summaries: `$H/final/scripted_cross/summary_{handoff,carry,drop}.json`, traces `$H/final/scripted_cross/<stage>/`.

Same-side: labels unchanged by construction and by test (`test_same_side_labels_unchanged`, frozen copy of the old
`classify` over 4 stages × 40 steps × 5 previous labels). The scripted pick solver now grasps cross-side bolts high
(CoM 0.156 below link_6) and the reach solver pre-grasps them at pad_z 0.150: natural pick set first 20 re-run
15/20 (+1 on all 4 cross-side; same as before), reach — see Coverage after.

### G2 (tests, final code)
`tests/test_nuts_bolts_subtasks.py` **19/19** (13 old + same-side labels unchanged, cross-side roles/pick goal,
handoff failures, both-holding not the goal, cross carry giver touch / receiver-only drop; starts test now also
builds 3 handoff starts), `tests/test_nuts_bolts_scorer.py` 4/4, `tests/test_inhand_subtasks.py` 14/14 (shared
`handoff_stage.py` untouched). Logs `$H/test_{nuts_bolts,nuts_bolts_scorer,inhand}_final.log`.

### Starts
- handoff: first 100 cross-side seeds (`cross_screen.json`), giver holds it lifted 6–12 cm where it lay, grasp
  `HIGH_PAD_Z` (CoM at link_6 z 0.152–0.160). `$H/final/starts_cross/handoff.npz` (= `$H/round0/starts/`).
- carry/drop cross-side: **synthesized post-handoff states** (grasp source 3), not chained: the scripted handoff is
  ~460 actions = 16–19 CPU-s on this node (> 10 s budget). carry: `receiver_hold` (side grasp, plates down, middle,
  giver home); drop: `receiver_hold_over_bin` (bolt head-up over the opening in the rolled side grasp).
  `$H/final/starts_cross/{carry,drop}.npz`.
- Natural-distribution sets (`$H/final/starts_natural/`, `remap_starts.py`): old same-side rows with label indices
  remapped (handoff inserted), cross-side carry/drop rows replaced by the new cross-side rows of the same seed:
  reach 100, pick 100, carry 100 (13 cross replaced), drop 89 (4 replaced, 2 dropped); 0 misclassified on restore.
- Use `starts_natural` for training in the natural mix; `starts_cross` for the cross-side stages.

### Coverage after
| measure | before | after |
|---|---|---|
| physical E2E chain, first 30 cross-side seeds (`e2e_chain.py`, one episode reach → pick → [handoff] → carry → drop, target DONE, no warning) | **0/30** (17 reached carry, 0/17 carry planned) | **15/30** (22 reach ok, 22 pick ok → 18 in handoff; handoff 16, carry 15, drop 15; misses: 8 reach, 4 pick→reach, 2 no receiver grasp, 1 dropped) |
| stage-solver chain, cross-side (product of solver rates on that stage's starts) | reach 2/4 · pick 4/4 · **carry 0/4** = **0** | reach 2/4 · pick 4/4 · handoff 16/20 · carry 20/20 · drop 15/20 = **0.30** |
| stage-solver chain, same-side (unchanged code paths) | reach 14/16 · pick 11/16 · carry 12/16 · drop 12/17 ≈ 0.32 | same (reach/pick re-run 16/20, 15/20 on the natural sets incl. cross) |
| natural mix (13 % cross-side in the first 100 seeds, 15.7 % of 1000) | 0.87 × 0.32 + 0.13 × 0 ≈ **0.28** | 0.87 × 0.32 + 0.13 × 0.30 ≈ **0.32** |
| physical E2E chain, first 100 seeds | 0/100 (same-side 0/87: solvers don't compose — nut pick blow-ups, carry IK) | not re-run: same-side paths unchanged; the 13 cross-side seeds are the first 13 of the cross-side 30 above |

Evidence: `$H/before/e2e_{first100,cross30}.json`, `$H/after/e2e_cross30.json` (round 0 with the old low pick grasp:
0/30, 15 handoffs "sticks out < 1.6 cm" → `$H/after/e2e_cross30_round0.json`), `$H/final/scripted_natural/`.

### Repair log (handoff amendment)
| gate | round | what failed | what changed |
|---|---|---|---|
| handoff G5 | round 0 | giver_alone 0/20 planned (straight move to the bin unreachable — the reason for the handoff) | hack carries in 4 cm steps and opens where its reach ends → 20/20 −1 |
| cross carry G4 | round 0 | solver 0/20 planned: the side grasp cannot keep its heading to the bin | `carry_turned`: raise, turn about the vertical (bolt is symmetric) → 20/20 |
| cross carry G5 | 1 (round 2) | wrong_way 15/20, wrong_bin 13/20 planned | hacks may turn about the vertical → wrong_way 20/20; wrong_bin 0/20 (other bin out of the receiver's reach) |
| cross carry G5 | 2 (round 3) | wrong_bin 0/20 planned | wrong_bin carried in 4 cm steps, opens where it ends → 20/20 −1 |
| cross drop G3 | round 0 | 65/100 valid (old top-down lowered hold: the upright bolt slides out / onto the floor; 14 idle returns 0.05) | – |
| cross drop G3 | 1 | 58/100 (hold 8 mm above the floor: bolt touches the floor during the hold check) | – |
| cross drop G3 | 2 | 57/100 (settle two chunks before saving: budget overruns) | – |
| cross drop G3 | 3 | – | synthesized carry-end state `receiver_hold_over_bin` → 93/100 PASS |
| cross drop G4 | round 0 | solver 0/20: horizontal hand meets the rim with the bolt 1.6–4.9 cm up → released too high | – |
| cross drop G4 | 1 | – | lower in 1 cm steps, pitch the hand down 15° steps, lower again, slow open → 15/20 PASS |

Counting: carry used 3 rounds (one G4 + two G5), drop G3 3 rounds, drop G4 1 round, handoff G5 1 round; none beyond.
G3 rounds 1–2 of cross drop changed start code that is no longer used (reverted in round 3).

### Judgement calls (for review)
1. Handoff region = the whole table volume (x 0.25–0.90, |y| ≤ 0.5, z 0.70–1.40): the handoff precondition is the
   cross-side pick goal, wherever it was picked. Being in the middle is only shaping: progress = `handoff_stage.progress`
   − min(distance to x 0.40–0.55, |y| ≤ 0.05, z 0.82–0.95; 0.30), span 1.55.
2. In carry/drop of a cross-side part the giver touching it again is −1, and drop entry needs the giver off it.
3. Handoff starts hold the bolt high on the shank. **A pick that encloses the whole bolt (CoM-centred) cannot be
   handed off** with these grippers; the pick reward does not prefer a high grasp for cross-side bolts, so a pick
   policy may produce handoff-infeasible holds. Options: accept; or add "≥ 1.6 cm below the fingertips" to the
   cross-side pick goal (changes the approved pick stage — not done).
4. Cross-side carry/drop starts are synthesized post-handoff states (source 3), because the chained handoff costs
   16–19 CPU-s per start here.
5. The scripted reach/pick solvers now pre-grasp / grasp cross-side bolts high; same-side solver paths unchanged.

## 2026-11-14 decisions and amended drop

**Merge:** `git merge substage-mug` (a56077d, no conflicts) before any change.

**Reach — accepted (user decision 1).** Marked PASSED with the documented G5 exception: no hack earned +1 (0/80),
the push_reach strike that misses its EXPECT hits the bin (−1 "bin moved or tipped"); a push necessarily touches the
part first, which is a legal reach, and pushing after touching is −1 in pick (push hack 20/20). No code change.

**Drop — low release only (user decision 2).** Amendment (commit c672721, `nuts_bolts_stages.py`/`_subtask.py`): in
the drop stage, at the first executed action without finger contact on the target, the target's lowest collision
extent (bin frame) must be ≤ `release_max_clearance_m` = **5 mm** above the bin floor top, else terminal −1
**"released too high"**. Checked after the "released outside the opening" rule, before the part can fall, so a high
release reads −1 "released too high" rather than as a numerical failure. Numerical guard (−1) kept. Goal,
precondition, other failures, carry and gate thresholds unchanged. Stage table amended (`stage_table.md`).
- Threshold measurement (`$O/amend/release_sweep.json`, `release_sweep_slow.json`; script
  `release_height_sweep.py`; old reward, 20 final drop starts, release clearance recorded at the first untouched
  step): clearance < 2 mm → 14/20 +1, 3 numerical; 2–5 mm → 1/4 numerical; 5–10 mm → 8/15 numerical; 10–20 mm →
  12/23 numerical; > 20 mm → 10/14 numerical. Judgement call: 5 mm = smallest round value above the ~2 mm release that
  worked (the diagnostic's height), leaving room for a part resting slightly tilted or with contact penetration;
  above it about half of all releases blow up. The sweep also showed that the fingertips stop the hand 0.7–2.8 cm
  above the floor, and that only a slow open lets the part slide down to the floor while still touched
  (open over 160 actions 11/20 +1, 480 → 14/20, 960 → 16/20).
- Starts (G3 rerun; precondition unchanged, rebuilt because the drop code changed): **91/100 valid**, 93 built,
  idle-kept 91/93 = 97.8 %, max 9.6 s CPU → PASS. `$O/amend/starts/drop.{npz,json}`, log `$O/amend/drop_starts.log`.
  The set differs slightly from `final/starts` (the CPU-time attempt budget under node load); the amended run uses
  the new set.
- New hack **release_high** (open over the opening without lowering, EXPECT −1). Solver: lower to target 2 mm
  (fingertips stop it), open at 1/960 per action until 16 actions after release, open fully, idle 4 chunks.
- G2: 14/14 subtask unit tests + scorer tests all ok (`$O/amend/test_*_final.log`) (new `test_drop_release_only_low_in_bin`: 0/2/4.9 mm legal, 5.1/20/50 mm "released too
  high", carry unaffected; two older release tests adjusted to the rule). `$O/amend/g2_amend.log`.

| round | solver +1 | numerical | other solver outcomes | hacks | margin | G4 | G5 | what changed |
|---|---|---|---|---|---|---|---|---|
| 0 (`$O/amend/round0`) | 11/20 (20 planned) | 2 | 6 × released too high (5–9 mm), 1 × 0 | 0/100 +1; release_high 18/20 as expected | 0.44 | FAIL (55 % of planned) | FAIL (margin) | – (first eval of the amended stage) |
| repair 1 (`$O/amend/round1`) | **15/20** (20 planned) | **0** | 3 × released too high (5.5–6.6 mm), 1 × bin moved, 1 × 0 (seed 950000015, part stays wedged in the fingers; release_high also 0 there) | **0/100 +1**; release_beside/wrong_bin/tip_bin/idle 20/20, release_high 18/20 (1 numerical, 1 × 0 on the same wedged seed) | **0.80** | **PASS** (75 % ≥ 70 %, 75 % ≥ 40 %) | **PASS** | dropped the fast loosen to the held aperture; slow ramp from closed as in the sweep |

Drop PASSED after 1 of 3 repair rounds. Note (judgement call): the scripted solver needs 230–520 actions (7–16
chunks; 1184 on the wedged seed), more than the specs' default `max_chunks` = 10; the runner does not enforce
max_chunks (as in all earlier rounds). An RL policy with a 10-chunk horizon has to open faster than this solver.

## Decisions (approved 2026-11-13)
reach/pick/carry/drop; nearest unsorted active part; fixed nuts→teal, bolts→orange; release above the assigned
opening allowed with real rest + fixture-rooted support; wrong bin, push, bin motion/tilt and loss of sorted
contents fail; retain and measure cross-table coverage before deciding on handoff. Required by the user in addition:
numerical-robustness guard — new MuJoCo warnings / time reset / non-finite state → finite −1 "numerical
instability" (implemented, unit-tested; scripted runners count such trials as misses/not-as-expected).

## Repair log

Reward/stage code was unchanged from round 0 (10:28) to round 5; rounds 1–4 changed only scripted behaviours and
start tooling. The interrupted session did not record diffs per round; the 'what changed' column for rounds 0–4 is
reconstructed from the summaries and the current scripts.

| gate | round | what failed | what changed |
|---|---|---|---|
| audit methodology | – | unclassified QACC warnings in first settle search | fresh MjData per trial, warning checks |
| reach G5 | first eval (round 1, push_reach added) | push_reach 16/20 +1 | – |
| reach G5 | repair 1 (round 2) | 13/20 as expected | ballistic one-finger push |
| reach G5 | repair 2 (round 3) | 0/20 planned (script bug) | strike rewrite |
| reach G5 | repair 3 (round 4) | 15/20 (75 %) | rapid strike + event-ordering trace → **STOP** |
| pick G4/G5 | repair 1 (round 2) | round 0: solver 4/20 (14 planned), margin −0.37 | 3.5 mm lower pre-grasp; pick passes since (15/20) |
| carry G5 | first eval (round 0) | wrong_bin, move_bin hacks missing | – |
| carry G5 | repair 1 (round 2) | release/wrong_bin mostly numerical invalid, move_bin 4/20 | numerical trials reported separately |
| carry G5 | repair 2 (round 3) | move_bin 4/20 | release/wrong_bin plans fixed (20/20) |
| carry G5 | repair 3 (round 4) | – | free arm shoves its own-side bin: 20/20 → PASS |
| drop G4 | first eval (round 2) | solver 3/20, planned 4, 16 numerical | – |
| drop G4 | repair 1 (round 3) | 3/20, planned 8, 12 numerical | carry chain / plan fixes |
| drop G4 | repair 2 (round 4) | 3/20, planned 13, 7 numerical, 9 "back to carry" | plan fixes, numerical trials separated |
| drop G4 | repair 3 (round 5) | 4/20 (final starts), 14 numerical | drop column predicate fix (a4f365d) + release 1.5 cm above floor → **STOP** |
| drop (amended) G4/G5 | round 0 | solver 11/20, 6 released too high, margin 0.44 | – (amendment c672721, new solver + release_high hack) |
| drop (amended) G4/G5 | repair 1 | – | slow open from closed → 15/20, margin 0.80: **PASS** |
| G3 measurement | – | under node load ~200/128 cores wall-clock budget gave drop 56/100 | budget measured in CPU time (2c8331b): 90/100, max 9.3 s; contended set kept in `$O/final/contended_wallclock/` (scripted drop there: 5/20, 13 numerical) |

History: `$O/round{0..5}*`, `$O/interrupted/` (run cut off at 11:00: partial pick scripted, wall-clock drop starts).

## Kit / scene proposals (not applied; stagekit untouched)
1. `stagekit/subtask.py`: a generic numerical guard (warning counts / time reset / non-finite → finite latched
   failure) in `SubtaskEvaluator.evaluate_model_data`; this task implements it locally.
2. Scene (`nuts_bolts_sorting` assets): nut collision boxes have tiny inertia and the gripper pads penetrate ~3.6 mm;
   candidate fixes are body armature/inertia floor, softer finger–part `solref`/`solimp`, or a smaller timestep for
   this task. Needed before drop (and any RL release of a nut) is trainable.
3. Kit tooling: a CPU-time start budget helper (wall clock is not a property of the start on shared nodes).

## Other worktree state
`slurm/eval_foundation_task.bash`, `slurm/submit_foundation_all_tasks.bash`, `slurm/test_eval_foundation_put_bottles.bash`
have mode-only changes (100755 → 100644) unrelated to this task; left uncommitted. `abc_sim/models/assets` is an
untracked symlink to the main assets; not committed.
