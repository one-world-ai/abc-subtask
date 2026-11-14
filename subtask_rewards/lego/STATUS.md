# sort_lego_blocks — subtask rewards ledger

short name: `lego` · template: A · seeds: SEED_BASE = 930_000_000 (start i = seed SEED_BASE+i) · baseline: exp12 1/10, mean final 0.908 after the scorer repair (0.891 before)
status: **DONE** — reach, pick, handoff, carry, drop PASSED (2026-11-14). The handoff stage (amendment 3,
user-approved 2026-11-14) covers the cross-side scenes, so the same-side drop scope is lifted: end-to-end coverage
**100/100** of the first 100 scenes (eligible 100 %, every stage of the scene's chain has a valid start), was 52.7 %

Paths below: `$O` = `../simulation/outputs/subtask_rewards/lego` (outputs), `$T` = `../simulation/experiments/subtask_rewards/lego` (tooling:
`build_starts.py`, `scripted_rollouts.py`).

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASSED | stock resets 0/40, repaired 0/20; 54 handmade cases; 180 physical drops; 2 near-wall false negatives and tipped/inverted-bin false positives repaired; 10/10 replays bit-exact, success 1/10 unchanged | `probes/`, `tests/test_lego_scorer.py`, `probes/replay_cavity/summary.json` | 2026-11-13 |
| 1 stage table | G1 | PASSED — user approved ("lets continue") | reach/pick/carry/drop; nearest unsorted target; matching-colour bin; release above the opening allowed; wrong bin, ungrasped pushing, bin disturbance, loss of sorted contents fail; measure carry reachability before exclusions/handoff | `stage_table.md` | 2026-11-13 |
| 2 code + tests | G2 | PASSED | 15/15 (`tests_g2_final.log`); rerun after the drop scope 15/15, start test now also builds/idles/restores 3 same-side drop starts (`tests_g2_round5_scoped_drop.log`) | `tests/test_lego_subtasks.py` | 2026-11-14 |
| 3 starts | G3 | reach, pick, carry PASSED; drop FAILED (rounds 0–2), then PASSED scoped to same-side scenes (round 3) | valid/100: reach 100, pick 100, carry 97, drop 100 of the first 100 eligible (round 2 unscoped: 33). Mean s/start 1.3 / 1.6 / 6.2 / 3.9 CPU s. Idle 10 chunks still in stage: 100 % of built for all four; drop 30-chunk idle 100/100. Drop coverage: 118/224 scanned seeds same-side (52.7 %), 106 excluded | `$O/starts/round0/{reach,pick}`, `$O/starts/round1/carry`, `$O/starts/round3/drop` | 2026-11-14 |
| 4 scripted | G4/G5 | reach, pick, carry PASSED; drop PASSED on the scoped starts (19/20, margin 1.01, `$O/scripted/round3/summary_drop.json`, 2026-11-14) | see Stages | `$O/scripted/round5/summary_reach.json`, `$O/scripted/final/summary_pick.json`, `$O/scripted/round1/summary_carry.json`, `$O/scripted/round2/summary_drop.json` | 2026-11-13 |
| 5 report | – | DONE | | `design.md` | 2026-11-13 |
| 1a amendment 3 (handoff) | G1 | APPROVED by the user 2026-11-14 ("add a handoff so cross-side scenes are covered"); table written by the agent, decisions below | handoff for cross-side blocks only; carry scoped (cross-side carry starts held by the receiver); drop scope removed | `stage_table.md` (Amendment 3) | 2026-11-14 |
| 2 code + tests (handoff) | G2 | PASSED | `tests/test_lego_subtasks.py` 18/18 (3 new: handoff labels cross/same-side, handoff failures incl. a physical drop, handoff solver → carry; start test now builds/idles/restores 3 handoff starts and mixed cross/same-side carry/drop); `tests/test_lego_scorer.py` 7/7 (`tests_lego_scorer_round6.log`); `tests/test_inhand_subtasks.py` 14/14 (`tests_inhand_after_lego_handoff.log`) (shared handoff_stage unchanged) | `tests_g2_round6_handoff.log` | 2026-11-14 |
| 3 starts (handoff, carry, drop) | G3 | PASSED | handoff 100/100 of the first 100 cross-side seeds (216 scanned, 112 same-side excluded by design), mean 2.3 CPU s, max 5.1, idle 100 %; carry 100/100 (52 cross: 33 chained handoff solver + 19 source 3; 48 same-side), mean 5.2 CPU s, max 10.3 (4/100 > 10 s), idle 100 %; drop 100/100 (52 cross + 48 same), mean 4.2 CPU s, max 6.9, idle 10 chunks 100 %, 30 chunks 100/100 | `$O/starts/round4/{handoff,carry}`, `$O/starts/round5/drop` | 2026-11-14 |
| 4 scripted (handoff, carry, drop) | G4/G5 | PASSED all three | see Stages | `$O/scripted/round5/summary_{handoff,carry,drop}.json` | 2026-11-14 |
| 5 report (handoff) | – | DONE | | `design.md`, end-to-end coverage `$O/coverage_round5.json` | 2026-11-14 |

## Stages
Scripted checks run from the first 20 saved starts of each stage, scored once per 32 actions with
`chunk_reward()` exactly as in RL. "x/20 (planned y)": solver +1 on x starts, IK plan found on y.

| stage | status | starts valid/100 | solver +1 | solver mean return | best hack (mean return) | margin | G3 | G4 | G5 |
|---|---|---|---|---|---|---|---|---|---|
| reach | **PASSED** | 100 | 20/20 (planned 20) | 1.483 | idle 0.000 (side_push −0.536, 20/20 −1; away −0.006) | 1.48 | pass | pass | pass |
| pick | **PASSED** | 100 | 20/20 (planned 20) | 1.250 | close_only 0.250 (idle 0; push/one_finger_drag −0.84; lift_open −1) | 1.00 | pass | pass | pass |
| handoff (cross-side) | **PASSED** | 100/100 eligible (cross-side) | 14/20 (planned 15) | 1.005 | both_hold 0.189 (never +1, 19/20 planned; idle 0; drop_between −0.88; set_down_regrasp −0.88; giver_across −0.94, 20/20 −1) | 0.82 | pass | pass | pass (round 1) |
| carry (scoped) | **PASSED** | 100 (52 cross-side held by the receiver, 48 same-side) | 19/20 (planned 20; same-side 9/9, cross-side 10/11) | 1.175 | idle −0.0001 (shake −0.0005; wrong_way −0.076; wrong_bin −0.91, 19/20 planned; release −1) | 1.18 | pass | pass | pass (round 1) |
| drop (scope removed) | **PASSED** | 100 (52 cross + 48 same) | 17/20 (planned 20; same-side 8/9, cross-side 9/11) | 0.775 | idle / hold_inside 0.000 (wrong_bin −1.00; release_beside −1.01; tip_bin −1.01) | 0.78 | pass | pass | pass |

Rows for reach/pick are unchanged from 2026-11-13 (their starts and goals are physically unchanged; see the
handoff section). The earlier carry (round 1, 12/20) and scoped-drop (round 3, 19/20) results are superseded.

No hack reached +1 on any start in the final runs.

## Handoff (amendment 3, 2026-11-14)

**What changed.** `STAGES = reach, pick, handoff, carry, drop, done` (label indices shift: handoff 2, carry 3, drop
4, done 5, fallen 6). A block is cross-side when its rest y and its matching bin's rest y have opposite signs (the
scoped drop's `start_side != bin_side` rule, now computed in `LegoScene.observe` from the episode's rest array).
Receiver = the arm on the bin's side, giver = the other. `classify`: same-side blocks unchanged; for cross-side
blocks a lifted giver hold or the shared `handing` test is handoff, the shared `received_in_air` (and, once in
carry, `received` while lifted) is carry, and drop never while the giver touches it. The shared
`handoff_stage.py` is used unchanged, configured by `HandoffConfig(region_lo=(0.40, -0.10, 0.80),
region_hi=(0.60, 0.10, 1.10), span=1.55)` and these observation arrays: `com, lift, f_give, f_recv, touch_give,
touch_recv` (per-arm finger counts / any arm-subtree geom), `gap_give` (giver pad midpoint → nearest point of the
block's collision box), `env_touch` (block touches any non-robot geom). No change to `handoff_stage.py` was needed.

**Measured geometry.** Blocks are 4.1–5.5 × 2.1–2.8 × 1.2–1.7 cm, 13 g; the finger pads are 3 cm wide. Two
top-down grasps do not fit side by side (the inhand cream-cheese stick, a similar size, failed 0/6 that way), so
the giver holds it top-down across its width at the middle and the receiver closes along its length on the free end
faces, approaching from its own base side at 15–55° elevation (`lego_handoff.receiver_T6s`). The nearest block is
the left arm's in every scanned scene, so the giver is always the left arm and the picking arm. Pinch centre: the
scene's measured closed-pad midpoint link_6 (0, −0.044, 0.130) (`LegoScene.pad_link6`, collision pad spheres),
which is where the inhand note's (0, −0.031, 0.125) lands for these pads; not `<arm>_grasp_site`.

**Starts.** handoff = pick's goal: the giver's sampled lifted hold (block 6–12 cm above its rest spot, weightless
close, gravity hold, ≤ 1 mm creep), the same construction carry used (the physical pick is skipped: it was never
pad-centred, G3 drop round 3). Cross-side carry: source 1 = a handoff start + the scripted handoff solver (33/52),
source 3 = block moved weightless into M and the receiver holds it alone (19/52). Cross-side drop: receiver hold in M
(source 3 first, chain second, to leave CPU time for the carry), then the carry solver with the receiver
(`carry_to_opening(..., follow=True)`, tilts 0/−30/30), creep ≤ 1.5 mm. Same-side carry/drop starts are built
exactly as before.

**Scripted solver (handoff).** The giver lifts the block and brings it into M (`plan_region_pose`: first (point,
long-axis yaw) whose giver pose is reachable and where the receiver's open gripper fits, checked by teleporting and
restoring), the receiver moves to a pre-grasp in joint space and straight in, closes; once both its fingers are on
the block the giver opens in monitored steps, lifts 10 cm and backs off 6 cm. Misses on the first 20: 4 ×
"receiver closed with 0 fingers" (only the low 0–15° approaches fitted, the fingers missed the end faces), 1 ×
region pose unreachable for the giver (seed 930000018), 1 × handed over then dropped during the giver's retreat
(seed 930000020, −1). Mean return 1.005.

**End-to-end coverage** (`../simulation/experiments/subtask_rewards/lego/coverage.py`, `$O/coverage_round5.json`):
of the first 100 seeds, 52 cross-side, 48 same-side; every seed is eligible for every stage of its chain (no design
exclusion left) and **100/100** have a valid start for every stage of their chain (cross-side 52/52, same-side
48/48). Before the handoff: 52.7 % (scoped drop). Scanned over the 216 handoff seeds, 104 (48.1 %) are cross-side.
Scripted chain success per stage on the first 20 is in the table above (cross-side: handoff 14/20, carry 10/11,
drop 9/11).

### Earlier history: drop scoped to same-side (superseded)

**Drop PASSED, scoped (2026-11-14).** The user scoped drop to same-side scenes (decision below). The round-3
builder (`build_starts.py --same-side --long-idle 30`) classifies each seed SEED_BASE+i with the same
`start_side`/`bin_side` rule that `_build_start` records, and builds only same-side seeds. It scanned seeds
930000000–930000223: 106 cross-side excluded, 118 same-side built, all 118 valid. The first 100 eligible seeds are
the G3 set (100/100 valid, mean 3.9 CPU s, max 6.95; idle 10 chunks 100 % in stage; an extra 30-chunk idle
100/100 in stage, return 0). Saved: the first 100 valid, `$O/starts/round3/drop.{npz,json}`. Same-side coverage of
the task's scenes is 118/224 = **52.7 %**; the other 47.3 % have no drop start and need a handoff stage. G4/G5 on the
first 20: solver 19/20 (20 planned), mean return 1.013; idle 0, hold_inside 0, wrong_bin −0.95 (19/20 planned, all
as expected), release_beside −1.01, tip_bin −1.01; no hack +1; margin 1.01. Solver miss, seed 930000017: the block
is released into the matching bin (in_bin, no contact, bin moved 7 µm), but for all 10 chunks it keeps jittering
3–5 mm up at 0.7–5 cm/s without the support check holding, so the stage stays open (return 0.25, no terminal).
That is a conservative miss, not a false positive. Evidence: `$O/scripted/round3/drop/solver_930000017.{json,npz}`.

Round-2 history (before the scope): **drop STOPPED — G3, 33/100 valid (rounds 0/1/2: 51, 33, 33).** About half the seeds (52/100) put the target
block on one side of the table and its matching bin on the other. The near arm cannot carry it across: carry
solver 9/9 on same-side starts, 3/11 on cross-side starts (all 8 misses are IK-unreachable approaches over
the far bin); drop starts built on cross-side seeds 2/52, same-side 31/48. Sampled over-bin holds with the
bin-side arm (the fallback) mostly fail the 1 mm/2-chunk creep check or IK. This is a workspace limit, i.e. a
hard stop in `gates.md`; the stage table left the remedy (handoff vs. same-side scope) to the user after the
measurement, so no exclusion was applied. Evidence: `$O/starts/round2/drop.json` (per-seed `chain_tilts`,
`start_side`, `bin_side`), `$O/scripted/round1/summary_carry.json`. The drop reward itself checks out on the 33
starts that exist (G4 19/20, G5 all hacks as expected, margin 0.94); its one solver miss (seed 930000045)
released the block and it landed outside the bin.

Notes the user should see:
- Carry G3: the builder's time check is the mean (6.2 s ≤ 10 s); 20/100 carry builds took 10–12.3 s, because
  sampled holds are retried when they creep 1.0–1.2 mm per 2 chunks. Read strictly per start, this is over.
- Carry G4 passes at 12/20 overall only because planning failures are counted as misses; the misses are the
  same cross-table reach limit as drop.
- **Carry coverage, by side (not scoped; carry stays PASSED as gated on all seeds):** of carry round-1's 100 seeds,
  48 are same-side and 52 cross-side. Valid carry starts are 47/48 same-side and 50/52 cross-side (97 total), so the
  saved carry set is about half cross-side. The scripted carry solver earns +1 on 9/9 same-side and 3/11 cross-side
  of the first 20 starts (all 8 misses: no IK path over the far bin). For training, read the 50 cross-side carry
  starts as mostly beyond one arm's reach; the start's `info.start_side`/`bin_side` tell them apart. Filtering them
  is a user call; it was not asked for, so the carry start set is unchanged.
- Start building has an 8.5 s wall-clock retry budget, so whether a marginal seed builds depends on CPU load
  (a saved start, once built, restores exactly). Use the saved `.npz` starts for training.

## Decisions (made with or for the user)
- **User (2026-11-14): add a handoff so cross-side scenes are covered** (amendment 3). Judgement calls made by the
  agent to apply it (for review):
  1. "Cross-side" = rest-spot y sign of block vs. bin (the scoped drop's rule), so a centre-slot bin with y < 0
     (|y| ≤ 3.5 cm) also needs the handoff, though the left arm could sometimes reach it alone (3/11 old cross-side
     carry successes were such bins). Changing this would need a reachability rule, not a sign.
  2. Handoff region M = x 0.40–0.60, |y| ≤ 0.10, z 0.80–1.10 (inhand's |y| band; x kept off the middle bin's
     opening; z from this table height). Handoff progress = shared progress − min(distance to M, 0.30), span 1.55,
     because the handoff start is pick's goal (no separate "present" stage as in inhand).
  3. New handoff failure "giver carried it across" (giver touches it > 10 cm past M on the receiver's side), the
     analogue of inhand's approved decision 1; the shared failures (dropped, touched the table/bin/another block)
     apply unchanged. Cross-side carry falls back (−1) if the giver comes back within 6 cm or onto the block.
  4. Carry scope: same-side carry starts as before; cross-side carry starts are held by the receiver after a handoff
     (the old round-1 cross-side carry starts, held by the giver, now classify as handoff and are replaced by
     `starts/round4/carry`). Drop: scope removed, `starts/round5/drop` covers both sides.
  5. Pick and reach are physically unchanged (pick's goal "both fingers, lifted ≥ 3 cm" now labels handoff for a
     cross-side block, carry otherwise), so their saved starts (labels ≤ 1) and G3–G5 results were kept, not re-run.
  6. Handoff starts are the giver's sampled lifted hold (grasp source 3), not a physical pick chain: the physical
     pick never pad-centres (drop round 3). G3 judged on the first 100 eligible (cross-side) seeds.
  7. `carry_to_opening(follow=True)` (turn the block with the arm's bearing) for receiver carries; the carry solver
     tries it first on cross-side starts and second on same-side ones (a solver change, not a reward change).

- Approved table (G1) as listed above.
- Amendments made during G5 repair, both **tightenings** of the approved "ungrasped pushing fails" (no goal
  widened, no failure dropped):
  1. Reach goal: the grasp site must be within 5 cm of the block. The approved table also allowed "or finger
     contact"; finger contact now only keeps an episode in pick once it is there. Reason: the outside of a closed
     finger touches the block while the grasp site is 6–11 cm away, which is how a push starts (round 1, seed
     930000005 earned +1 by a palm push that way).
  2. Reach failure: moving the block more than 3 cm during reach fails even if both fingers touch it. A real
     grasp ends reach first (grasp site ≤ 5 cm); two closed fingers dragging the block from outside counted as
     "grasped" and escaped the push check (round 2, seed 930000004: 9 cm drag, 0 instead of −1).
- **User decision (2026-11-14): scope drop to same-side scenes.** Drop starts only come from seeds where the target
  block and its matching bin are on the same side (`start_side == bin_side`); cross-side seeds are excluded and
  counted (coverage 52.7 %), and a future handoff stage (template D) is the remedy for them. Drop goal, failures and
  gate thresholds are unchanged. Judgement calls made to apply it: (1) G3 is judged on the first 100 *eligible*
  seeds of the scan (a re-indexed seed list), with exclusions recorded per seed in `drop.json`; (2) the round-3
  builder repair (below) changed only drop start construction, not reach/pick/carry, whose saved starts were not
  rebuilt; (3) carry was left unscoped (see the carry coverage note).
- Earlier open question, now answered by the decision above (drop, and the cross-table half of carry): (a) restrict LEGO carry/drop starts to seeds
  whose target block and matching bin are on the same side of the table (measured: 48/100 seeds; drop
  starts built on 31/48 of them, so this would also need a re-indexed seed list to reach 80 valid), or (b) add a
  handoff stage (template D) for cross-table blocks, or (c) leave drop stopped.

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
| G5 handoff | 1 | round 4: giver_across planned 3/20 (the straight single-arm carry to the far bin is IK-unreachable on 17/20, the workspace limit itself) → as_expected 15 % | hack steps across 3 cm at a time as far as the giver's IK reaches, then lets go → 20/20 planned, 20/20 −1; receiver approach elevations 15/35/55/−15° added (solver unchanged 14/20) → G5 PASS (`$O/scripted/round5/summary_handoff.json`) |
| G5 carry | 1 | round 4: wrong_bin planned 13/20 (receiver's single −30° plan IK-unreachable on 6/11 cross-side starts) | hack tries the solver's tilts and both wrist modes → 19/20 planned, all −1 → G5 PASS |
| G3 drop | 4 (handoff) | round 4: 94/100 (6 cross-side seeds: receiver holds in M crept 1.1–1.2 mm, checked against carry's 1 mm) | bug: the region hold check now uses drop's own measured 1.5 mm limit (round-3 repair) → round 5 100/100 |
| G0 | 1 | horizontal/inverted containers score; near-wall valid rest rejected; raw exported mesh triangle faces are not reliably convex | upward-opening check; began measured collision-cavity correction |
| G0 | 2 | enumerating all vertex triples is too large for dense stackable-bin meshes | keep enumeration only for small no-graph meshes |
| G0 | 3 | need actual solver hull without another package | read compiled MuJoCo hull triangles and validate enclosure; both near-wall cases now pass and no other saved rest-pose scores change |
| G2 | 1 | start construction used the grasp site, missing thin blocks by the native finger-pad offset; carry built 0/12 in initial test candidate set | measured closed-pad midpoint in link6 coordinates; task-local native-pad grasps, open-table clearance and physical pick alignment. Suite 13/14 → 14/14 |
| G2 | 2 | new synthetic test state was not at rest, so the free-flight hold rule kept the label | test sets zero speed + resting contact (no code change) |
| G3 carry | 1 | round 0: 82/100 valid, 82 % still in stage after 10 idle chunks (slow slips) | pad-axis centring check + 64-action creep check (≤ 1 mm) at build time → round 1 97/100, 100 % idle-kept |
| G3 drop | 1 | round 0: 51/100 valid, 65 % idle-kept | same centring/creep checks → round 1 33/100, all idle-kept (the slips were the starts that had passed) |
| G3 drop | 2 | round 1 spent a full reset + regrasp per carry tilt and hit the 8.5 s budget | all 7 carry tilts from one held snapshot (`lego_starts.py`) → round 2 33/100: same-side 31/48, cross-side 2/52. Stopped: workspace limit, no 3rd round |
| G5 reach | 1 | round 0: palm_push hack planned 0/20 | (previous session) new palm-push geometry → round 1 planned 2/20, one +1 by a bare finger touch at 11 cm |
| G5 reach | 2 | finger touch counted as reach; palm push rarely planned | reach goal requires grasp site ≤ 5 cm (amendment 1); new side_push hack (outside of closed fingers, grasp site kept 10 cm away); solver tries 4 jaw yaws (14/20 → 20/20). Rounds 2–4 tuned the hack's contact (round 2: 3/20 +1, fingertips passed over the block; 1 two-finger drag scored 0) |
| G5 reach | 3 | two-finger drag escaped the push check; top-down fingertips cannot get below the block's top | reach push failure regardless of grasp (amendment 2); side_push tilted 20–35° → round 5: 20/20 −1, G5 PASS |
| G3 drop | 3 (scoped) | round 2 same-side seeds still built only 31/48: the physical pick was never pad-centred (block 1.0–1.2 cm off the pad axis on 0/204 carry and 0/100 drop attempts), and the sampled holds' chain was rejected by a 1.00–1.14 mm sink in the 64-action creep check, although those holds then stayed in stage for 30 idle chunks with return 0 (17/17 measured) | drop skips the physical pick; chain tilts only (−30°, 0°), the only tilts that ever succeeded (positive tilts 0/95 IK-reachable); over-bin fallback only from the 2nd attempt; drop creep limit 1.5 mm (measured settling, carry keeps 1 mm); retry budget in process CPU seconds instead of wall clock (node load). → 118/118 same-side valid, mean 3.9 CPU s |
| G5 drop | 1 | tip_bin dragged the bin only 2.7–5 cm (under the 5 cm failure): 0 instead of −1 on 6/20 | hack pushes 16 cm instead of 9 cm → 20/20 −1 |

## Phase 0 rule and findings

Original rule: score each active block when its joint origin lies inside its matching bin's authored local goal
box; wrong-color box memberships are counted separately; success requires every active block correct and none in
a wrong bin. There is no robot-contact, speed, support-contact, or bin-upright check.

The scene assets compile using the prepared assets from the primary ABC checkout (read-only input symlinks).
No downloads or modifications to the shared assets were needed. The 40 reset seeds contain 9–26 active blocks
(mean 18.45), with each color independently sampled at 3–9 blocks, bin and block scales 0.85–1.15, and bin
positions/colors shuffled across three slots. Each reset rebuilds the model, so the evaluator binds the current
geometry each time.

| probe | original | repaired | interpretation |
|---|---|---|---|
| stock 40 reset seeds / repaired 20 reset seeds | 0/40 nonzero | 0/20 nonzero | no free credit or crashes |
| matching center, floating | 9/9 count | 9/9 count | final subtask must add release/rest; qpos scorer alone cannot verify this |
| matching low interior point | 9/9 count | 9/9 count | ordinary valid bins preserved |
| outside side | 0/9 count | 0/9 count | reject |
| wrong-color bin | 9/9 detected wrong | 9/9 detected wrong | reject for target sorting |
| 90° tipped bin | 9/9 count | 0/9 count | opening must face upward |
| inverted bin | 9/9 count | 0/9 count | reject |

180 physical drops (3 seeds × 60), initial randomized poses near the matching opening, 1.5 seconds of settle:

| physical class | original accepted | repaired accepted |
|---|---|---|
| ordinary interior rest | 86/86 | 86/86 |
| confirmed inner-floor rest near wall | 0/2 | 2/2 |
| rim/outer bin support | 0/13 | 0/13 |
| table/other support | 0/76 | 0/76 |
| still moving (>2 cm/s) | 2/3 | 2/3 |

The two changed resting poses are seed 0/drop 59 and seed 1/drop 35. CPU renders show the blocks on the inner
floor. Collision rays at their center heights bound the physical cavity beyond the authored site: for seed 0,
block x=5.652 cm while the authored halfsize is 5.622 cm, with the positive wall at 7.017 cm. Seed 1's negative
x wall is at −7.622 cm while the block x=−6.247 cm and the authored halfsize is 6.130 cm. These are measured
geometry corrections, not relaxed success gates. The rest poses remain valid after another second of physics in
the regression test. The geometric classification was checked on `interior.png`, `outside.png`, `rim.png`,
`edge0.png`, `edge1.png`, and `inverted.png`.

Repair: retain the authored vertical interval and matching-color accounting. Compute the current randomized
bin's physical cavity from its convex collision hull pieces. A segment from the bin center at the target's local
height to the target center must remain clear of all collision pieces and within the measured outer bounds.
Reject horizontal/downward openings. Mesh hull faces come from MuJoCo's compiled hull graph; small no-graph
meshes use exact supporting planes. No new dependency. Validate that each computed hull encloses its vertices.
See [MuJoCo 3.8 MakeGraph/CopyGraph](https://github.com/google-deepmind/mujoco/blob/3.8.0/src/user/user_mesh.cc).

Six attempted top-down holds inside the bin volume failed setup (IK or gripper/object collision); no actual held
positive was produced. This is a start-construction warning, not a measured held-success rate. The observed
floating/moving positives independently require a released-and-resting stage goal. The draft drop starts should
chain from the carry solver; angled approaches may be necessary, as for bottles.

## Final baseline re-score

All ten exp12 worlds were replayed from their recorded seeds and actions with **zero state drift at every model
call**. The final scorer source SHA256 is
`6785f10b344a933c4f80d1f715bcbfdbe7241e61deb79ffac53e4c12d0c8acf1`.

| metric | original recording | repaired scorer |
|---|---|---|
| successful worlds | 1/10 | 1/10 |
| mean final reward | 0.890899 | 0.907664 |
| mean maximum reward | 0.901782 | 0.907664 |

The successful world is still world-04, first success at step 9050; it was cut short by genuine old success, not
a scorer false positive. Full per-check results and per-world summaries are in `probes/replay_cavity/`; the
aggregate is `summary.json`, and `run_status.json` records all ten successful replay processes. The mean reward
increase is expected from correcting the conservative cavity boundary; no episode's success verdict changed.

The scorer's 23-active-block, all-in-bin probe takes approximately 2.74 ms per evaluation on this CPU with
`OPENBLAS_NUM_THREADS=1`. This is an offline timing observation, not a training-throughput measurement.

## Integration

Only this worktree's LEGO files changed: `lego_blocks.py` (scorer), `lego_stages.py`, `lego_starts.py`,
`lego_subtask.py`, the four `sort_lego_blocks__<stage>` specs in `task_specs.py`, one registry line, tests and
this ledger. `stagekit/` is untouched. `lego_starts.py` reuses `bottles_starts.idle_action/move_link/bottle_yaw`.
