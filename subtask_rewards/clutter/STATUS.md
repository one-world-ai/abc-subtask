# grab_specific_object_from_clutter — subtask rewards ledger

short name: `clutter` · template: A (targeted pick and place) · SEED_BASE = 940_000_000 (starts on seeds SEED_BASE+i, i<100)
baseline: none; no clutter rollout directory exists in exp12's task outputs (checked 2026-11-13)
status: **DONE** (2026-11-14, handoff amendment) — chain reach → pick → [handoff] → carry → place. handoff PASSED,
carry PASSED with full coverage (98/100, direct + after-handoff), pick/place PASSED (re-run on the new labels),
reach PASSED by user exception. **End-to-end start coverage 90/100 scenes** (was 39/100: direct carry ∩ place).
History: carry was a direct-reach pilot (45 %) before 2026-11-14; reach G5 exception, see below.

## Handoff amendment — 2026-11-14 (user-approved: "add a handoff so those scenes are covered")

Branch `subtask-clutter` merged `substage-mug` first (fast-forward to f2974e3, no conflicts). Stage-table amendment:
`stage_table.md` § "Amendment 2026-11-14". `handoff_stage.py` unchanged (configured by `HandoffConfig` defaults +
clutter observation arrays); `inhand_*` unchanged (imported).

**Needs-handoff test (decided from the IK evidence).** Box at y = ±0.515 m, arm bases at y = ±0.31 m. In the direct
pilot's 100 carry candidates: picking arm (nearest at reset = pick-start arm) ≠ box arm in 53 scenes → **53/53**
"IK unreachable" exclusions; picking arm = box arm in 47 → 45 valid, 2 exclusions (+20 water bottle, +79 bar soap,
same-side; a handoff cannot help those). Rule: giver = non-box arm, receiver = box arm in every scene; a scene needs a
handoff iff picking arm ≠ box arm (53/100). The reward labels from state: giver holding it lifted → `handoff`, box arm
alone → `carry` (so a policy that picks with the box arm skips the handoff).

### Phases (handoff amendment)
Outputs `O = ../simulation/outputs/subtask_rewards/clutter/`.

| phase | gate | result | numbers | evidence |
|---|---|---|---|---|
| 1 table amendment | – | user-approved scope (handoff for the IK-unreachable scenes); decisions below | | `stage_table.md` |
| 2 code + tests | G2 | PASS | `tests/test_clutter_subtasks.py` 19/19 (+4 handoff: roles, classify chain, drop / set-down failures, progress terms + partition); `test_grab_clutter_scorer` 5/5; inhand 14/14 + 5/5 | `O/tests_handoff.log`, `O/tests_handoff_other.log`; commit 8a72c8c |
| 3 starts | G3 | handoff PASS 96/100 · carry PASS 98/100 · place PASS 92/100 | handoff = first 100 eligible of seeds +0..199 (strict first 100 seeds: 52/53 eligible), mean 1.8 CPU-s, idle 99 %; carry 100 seeds: 46/47 direct + 52/53 after a handoff (28 chained from the handoff solver, 24 grasp source 3), mean 5.3 CPU-s, idle 100 %; place 86 synthetic + 6 chained from the carry solver, mean 8.1 CPU-s, idle 100 % | `O/handoff_r1/starts/handoff.*`, `O/handoff_r2/starts/carry.*`, `O/handoff_r3/starts/` (full set) |
| 4 scripted | G4/G5 | handoff PASS (round 1) · carry-after-handoff PASS (round 2) · regressions PASS: pick, carry direct, place | see table | `O/scripted_h1/summary_{handoff,pick,carry_direct}.json`, `O/scripted_h2/summary_carry_handoff.json`, `O/scripted_h3/summary_place.json` |
| 5 report | – | written | | `design.md` |

| stage (20 starts) | solver +1 (planned) | solver return | best hack (mean return) | margin | hacks |
|---|---|---|---|---|---|
| handoff | 14/20 (15) | 0.90 | both_hold 0.15 | 0.75 | drop_between −1 19/20 (1 unplanned); both_hold never +1, ≤ 0.5, 18/20 (2 unplanned); set_down_regrasp −1 20/20; giver_alone never +1 20/20; idle 0 20/20; box_move −1 20/20; box_tip −1 17/20 (3 pushes missed the box); distractor_loss −1 17/20; 0 hack +1 |
| carry, handoff scenes | 20/20 (20) | 1.45 | shake 0.00 | 1.45 | regrasp_giver −1 17/20; release_early −1 20/20; wrong_way 0 20/20; shake 0 20/20; idle 0 20/20; box_move 20/20, box_tip 19/20, distractor_loss 16/20 −1; 0 hack +1 |
| carry, direct scenes (regression) | 19/20 (20) | 1.21 | idle 0.00 | 1.20 | all ≥ 80 %, 0 hack +1 |
| pick (regression, new labels) | 15/20 (15) | 1.14 | close_only 0.25 | 0.88 | all ≥ 90 %, 0 hack +1 |
| place (regression, r3 set incl. chained) | 19/20 (20) | 1.45 | lower_hold 0.33 | 1.12 | all ≥ 95 %, 0 hack +1 |

Handoff solver misses (5 unplanned + 1): receiver closes with one finger (+34, +41, +43), no receiver grasp (+1),
receiver cannot hold it alone (+33); +28: the target touched clutter during the exchange → −1 (honest failure).
Reach is unaffected (labels R/P unchanged) and was not re-run.

### End-to-end coverage (seeds SEED_BASE + 0..99, a scene counts if every stage of its chain has a valid start)
| | direct scenes (47) | handoff scenes (53) | all |
|---|---|---|---|
| before (direct carry ∩ place) | 39 | 0 | **39** (carry alone 45) |
| now (reach ∩ pick ∩ [handoff] ∩ carry ∩ place) | 44 | 46 | **90** |
| without place | 46 | 52 | 98 |
Uncovered: carry +20, +59; place +49, +50, +61, +68, +77, +86, +97, +99 (no collision-free lowering from any
over-box grasp or chained carry). The handoff-scene carry starts from the chained solver (28/52) are physical
evidence that the whole handoff can be done in those scenes; the other 24 rely on grasp source 3.

### Repair log (handoff amendment)
| gate | round | what failed | what changed |
|---|---|---|---|
| G2/G3 | – (phase-2 calibration) | top-down transport to M had no IK; receiver's open gripper hit the giver's finger links | transport turns hand + target about its CoM (closing axis aligned with y first, then x); presentation pose planned with a receiver-grasp check before moving |
| G2 | – | one contact-free physics step inside the receiver's closed pinch read as "dropped" | handoff touch = contact or finger ≤ 2 mm (observation, not the shared test) |
| G3 carry | 1 | 68/100: chain hit the wall-clock budget; 31 receivers had no straight carry path from M | CPU-time budgets (chain 8 s, total 16 s); `carry_plan` also tries hand+target yawed 90/−90/180/±45° after all original plans → 98/100 |
| G3 handoff | 1 | 85/100 (giver lift holds creep / no IK) | grasp source 1 fallback: pick start + scripted lift → 96/100 |
| G3 place | (coverage, not a gate failure) | 86/100 | fallback chained from the carry solver → 92/100 |
| G5 handoff | 1 | box_tip 15/20 (pushes missed the box), giver_alone unplannable 19/20 | box hacks retry the next push pose when the box did not move; giver_alone goes to M first, then towards the box → PASS |
| G5 carry-handoff | 1 | regrasp_giver 14/20 (giver closed on air beside the receiver's gripper) | retry the next free grasp if the giver closed without touching → 17/20 PASS |
No goal widened, no failure dropped, no gate changed.

### Decisions made for the user (handoff amendment, for review)
1. Giver/receiver are fixed by the box side (non-box arm gives) — not by which arm happens to hold it — so the label
   is state-only and a box-arm pick skips the handoff.
2. Handoff happens in the inhand region M (shared defaults, unchanged); receiving outside M before the giver is clear
   = back to pick (−1), as inhand.
3. Handoff span 1.75 = handoff_stage's 1.25 + a transport term −min(CoM distance to M, 0.5), since the stage starts
   where pick ends (no separate "present" stage, per the requested chain).
4. Pick's goal in a handoff scene additionally needs the lifted target to touch nothing but the robot (entry to
   handoff); direct scenes unchanged.
5. Handoff touch observation uses a 2 mm finger margin (measured contact flicker); finger counts use contacts.
6. Handoff starts: sampled giver grasp lifted 7–12 cm first (cheap, the old carry hold), pick-chain as fallback —
   the reverse of the kit's preference order, for CPU budget; both are pick-goal states.
7. `place` also requires no giver contact; `carry_plan` yaw options are tried only after every original plan.
8. Place starts got a chained-from-carry fallback (coverage only); direct-reach carry/place starts unchanged otherwise.
9. Handoff start set = first 100 eligible seeds (as inhand); carry/place sets stay the strict seeds +0..99.

## Approval recorded — 2026-11-13
The user said **“lets continue”** in response to the request to approve the three concrete tables and their
proposed defaults. G1 is approved, not auto-approved. Selected decisions: reach/pick/carry/place; fixed prompted
target; controlled placement with floor support; wrong-object lift fails; harmless distractor nudges allowed but
off-table loss or moved/tipped box fails; direct reachable-carry pilot with explicit coverage; raw XY benchmark
reward unchanged. All final goals require real release/rest/support; CPU single-world scoring; 32-action chunks.

## Phases

Outputs root `O = ../simulation/outputs/subtask_rewards/clutter/`; tooling `../simulation/experiments/subtask_rewards/clutter/`
(snapshot in `tools/`).

| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASS | 40 resets: 0 reward/0 success; 9 airborne/below-floor false successes rejected after repair; 41/41 supported settles kept | `probes/`; `tests/test_grab_clutter_scorer.py` 5/5 | 2026-11-13 |
| 1 stage table | G1 | PASS (user-approved) | see above | `stage_table.md` | 2026-11-13 |
| 2 code + tests | G2 | PASS | 15/15 (`tests/test_clutter_subtasks.py`, run as script) | `O/tests_final.log`; commits 74cd4c3, 233bf86 | 2026-11-13 |
| 3 starts | G3 | reach PASS 100/100 · pick PASS 100/100 · place PASS 86/100 · carry FAIL strict 45/100 (scoped pilot 45/45 eligible) | mean build 0.7 / 2.3 / 9.9 / 11.5 s; held-start 10-chunk retention 100 % in every stage | `O/final/starts/*.{npz,json}`, `O/final_<stage>.log` | 2026-11-13 |
| 4 scripted | G4 | PASS all four | solver +1: reach 12/20 (12/12 planned), pick 15/20 (15/15), carry 18/20 (18/19), place 18/20 (18/20) | `O/scripted_r3/summary_<stage>.json` | 2026-11-13 |
| 4 scripted | G5 | pick PASS · carry PASS · place PASS · reach FAIL (letter) | margins 0.68 / 0.88 / 1.14 / 1.06 (reach/pick/carry/place); no hack +1 anywhere in round 3 | `O/scripted_r3/` | 2026-11-13 |
| 5 report | – | written | | `design.md`, this file | 2026-11-13 |

## Stages

| stage | status | starts valid/100 | solver x/20 (planned) | solver return | best hack return | why / note |
|---|---|---|---|---|---|---|
| reach | **PASSED (user-accepted G5 exception, 2026-11-14)** | 100 | 12 (12) | 0.96 | 0.28 wrong_object | reach_push as-expected 11/20 < 80 %: 8/20 push runs had no IK for the scripted shove; 11/12 executed shoves end −1 ("pushed target"), 0/20 earn +1. Reward behaviour is right on every executed run; failure is harness coverage after 3 rounds. User decision below. |
| pick | PASSED | 100 | 15 (15) | 1.14 | 0.25 close_only | |
| handoff (2026-11-14) | PASSED | 96 (first 100 eligible) | 14 (15) | 0.90 | 0.15 both_hold | handoff scenes only; see the handoff amendment |
| carry | PASSED — full coverage since 2026-11-14 | 98 (46 direct + 52 after a handoff) | direct 19 (20) · handoff 20 (20) | 1.21 · 1.45 | 0.00 | was the 45/100 direct-reach pilot (old row: 18/20, 1.14, all 55 exclusions IK-unreachable) |
| place | PASSED | 92 (86 + 6 chained from carry, 2026-11-14) | 19 (20) | 1.45 | 0.33 lower_hold | 2 planned misses: tall targets (cereal, water bottle) in box, stock success, but an opened finger still rests against them (solver does not retract). |

## Repair log

| gate | round | what failed | what changed |
|---|---|---|---|
| G0 | 1 | Site-only success accepted objects wholly above the rim or below the floor | Physical floor/rim height check (scorer) |
| G3 | 1 | pick 73, carry 36, place 84 (initial 100/73/22/53) | reversed wrist yaw and leaning over-box grasps |
| G3 | 2 | pick 79, carry 38 | native-pad anchoring, fixture collision checks from moving-link roots, numerical invalidation |
| G3 | 3 (last) | pick, carry < 80 | −30° approach tilt + 2 cm pregrasp offset → pick 100/100; carry 45/100 (55 IK-unreachable direct carries). Place (its round 3): the stage table's "grasp that can lower without hitting the rim" was not implemented → added `place_plan` verification, box-aligned jaw, −35° lean variants → 86/100 (scripted_full_round1 place solver had planned only 3/20 because of rim collisions) |
| G4/G5 | 1 (`O/scripted_final_r1/`) | reach_push +1 6/12 (shove counted as reach); pick box_move +1 (hack closed the pregrasp on the target — a real pick); box/tip hacks mostly unplanned; place lowering | reach→pick entry requires target at rest and moved < 3 cm; place solver uses `place_plan`; box hacks search sides/yaws/leans, place box hacks drag the box with the held target |
| G4/G5 | 2 (`O/scripted_r2/`) | reach_push still +1 3/12 (slow 14–15 mm/s shove under the 20 mm/s at-rest bar); distractor sweep with open fingers left objects on table | fast pushes now keep commanding the final target (they had stopped where the arm lagged); pick box hacks use the free arm first |
| G4/G5 | 3 (last, `O/scripted_r3/`) | — | reach entry speed bar 5 mm/s (resting targets measured 0.0 at all 14 solver goals); distractor drag with closed fingers toward the nearest table edge. Result: no hack earns +1; reach fails only on reach_push execution coverage |

Not done as repairs: no goal widened, no failure dropped, no gate threshold changed. The G5 script counts an
unplanned hack as not-as-expected (strict).

## Evidence notes
- Box-motion failure predicate exercised by real pushes: reach 20/20 and 18/19, place 20/20 and 17/20 (box_move/box_tip).
  In pick/carry many box hacks end −1 earlier by regression (the pick arm leaving the target, "back to reach/pick") —
  still −1, but the box predicate itself is evidenced in reach/place and the unit tests.
- Carry solver seed 940000016: −1 "box moved or tipped" — the held target bumped the box on the direct path (honest failure).
- Distractor-loss: drags end −1 via "distractor off table" or "wrong object lifted" (both approved failures).

## User decisions needed
1. **Reach G5: ACCEPTED by the user 2026-11-14.** (Was: accept reach (every executed hack behaves; G5 as-expected computed over planned runs = 11/12) or keep it STOPPED. Reach starts are plain resets, so pick training does not depend on it.
2. **Carry coverage: RESOLVED 2026-11-14** — user approved the handoff stage; carry now 98/100, end-to-end 90/100.
   Review the handoff-amendment decisions above.
3. Publishing renders to the HF Space needs the user's org token (not done).

## Scene and scoring rule

The randomizer chooses 8–16 distinct categories from 24 household-object categories, a target from among them,
and an upright 14 × 15 cm interior box near y = ±0.515 m, with random yaw. The prompt identifies the target by its
category; `target_object_joint` identifies it in simulator state. The randomized model is rebuilt on reset.

Stock success: target **body origin** is inside the box-local goal AABB. Its half-size is (6, 5.5, 18) cm and its
centre is 11 cm above the box origin: vertical range −7 to +29 cm. Actual collision floor top is +1.2 cm and rim is
+10.4 cm. Stock reward depends only on world XY distance to the goal centre and can be 1 even when success=False.
It does not require release, rest, support contact, or an upright box.

The repair keeps the existing goal region and XY reward, and additionally requires (a) target CoM above the
physical box floor, within 5 mm tolerance and (b) some target collision geometry at or below the physical rim,
within 5 mm. Geometry is measured per randomized target model and transformed into the live box frame; tall
objects may protrude above the rim. Curved primitive geoms use conservative outer bounds, not a sampled visual
mesh. This is a narrow height fix, **not** a contact/rest or full-volume containment guarantee.

## Audit evidence

| probe class | stock successes / probes | repaired successes / probes |
|---|---|---|
| Reset, seeds 0–39 | 0/40 | 0/40 |
| Unsupported target origin at box-local z=20 cm | 3/3 | 0/3 |
| Unsupported target origin at z=28 cm | 3/3 | 0/3 |
| Target origin below floor at z=−4 cm | 3/3 | 0/3 |
| Above entire goal at z=40 cm | 0/3 | 0/3 |
| Beside box, local x=14 cm | 0/3 | 0/3 |
| At side rim, local y=7 cm | 0/3 | 0/3 |
| Wrong object placed in box, target left on table | 0/3 | 0/3 |
| Tipped box + target in its local goal | 3/3 | 3/3 — intentionally left to explicit stage box-motion semantics |
| Physically settled, independently classified inside + box contact | 41/41 | 41/41 |
| Physically settled on table | 0/93 | 0/93 |
| Physically settled rim supported | 0/14 | 0/14 |
| Physically settled outside but touching box | 0/2 | 0/2 |

- Settle trials: seeds 0/1/2 (yogurt, tomato, sponge), 50 random drops each, 1500 MuJoCo physics steps per trial.
  Classifier uses contacts and physical interior dimensions, independently of the evaluator goal site. It uses
  target-origin geometry, not a guarantee of full collision-mesh containment. Representative renders inspected.
- Stock run held no arm control targets; repaired run explicitly held both arms at reset joint positions. All
  settle class counts remained identical. Neither run is a policy benchmark or bit-exact replay of the other.
- Regressions: 20-reset check, airborne/below-floor rejection on 3 target models, two true placements after settling
  and a further 500 physics steps, wrong-object rejection, and batched per-pose geometry: **5/5 passing**.
- 40-reset geometry initialization covered 20/24 target categories. It is not exhaustive over all asset variants.
- No real gripper-held probe was constructed. The unsupported-pose probes demonstrate why release/support/rest
  must be added to the staged terminal goal; these still need held-state validation in phases 2–4.

## Workspace coverage

Only 13/40 reset seeds put target and box on the same y-sign side; 27/40 are opposite-side by this simple geometric
heuristic. This is **not** an IK proof that every opposite-side case needs a handoff (near-centre objects can be
reachable by both arms). Carry feasibility must be measured. The draft proposes a direct single-arm pilot with
explicit workspace eligibility and coverage reporting; do not silently label it full-task coverage.

## Baseline and other checkouts

No `grab_specific_object_from_clutter` or `grab_clutter` task directory appears under
`../simulation/outputs/openwam/experiment12/step-57000/tasks/`, and exp12's RESULT table has no clutter row. No
recorded baseline could be re-scored, and no model inference was launched in this audit. Baseline unavailable is
recorded rather than a blocker for developing substage rewards (consistent with the user's put-relative pilot).

Only this isolated worktree contains the repair until root integrates it. `../abc`, `../simulation/abc`, and any
RoboEnv/vendor checkout still have their previous scorer; none were synchronized by this track.

## Decisions (history)
- Parallel audit/design authorized by the user, 2026-11-13; G1 approved with "lets continue".
- Root authorized the narrow physical-height scorer repair; support/rest and box-motion semantics are stage goals.
- No `stagekit` changes. Specs/registry: one clutter block each.
