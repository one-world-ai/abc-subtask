# open_drawer / close_drawer — subtask rewards ledger

short name: `drawer` · template: C (articulated) · seeds: SEED_BASE = 980_000_000 (open), 985_000_000 (close) · baseline: none
status: DONE 2026-11-14 — 5 stages PASSED (open reach/grasp/release, close reach/release), 2 STOPPED at G5 (open pull, close push: the wrong-drawer hack cannot be executed; see Stages)

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASSED (no scorer fix needed) | overlay applied, assets prepared; resets 0/40 + 0/40 score 0 (also after one idle chunk, drift 0); scripted open 18/20, close 12/20 succeed; wrong-drawer scripts 0/20 succeed; flings up to 2 m/s move the drawer ≤ 2.1 cm (no coast, no bounce); thresholds exact at ±1 mm | `probes/dr_{reset,scripted,geom,close_touch,summary}.py`, `probes/{reset.json,scripted_s0.json,scripted_s10.json,scripted_summary.txt,geom.log}` | 2026-11-13 |
| 1 stage table | G1 | PASSED (user approved all default options) | open: reach → grasp → pull → release; close: reach → push → release | `stage_table.md` | 2026-11-14 |
| 2 code + tests | G2 | PASSED | `tests/test_drawer_subtasks.py` 6/6 (a starts classify 7 stages x 3 seeds; b idle 10 chunks no terminal, \|return\| < 0.05; c wrong drawer / pulled past start / hand leaves / release band crossed all -1; d 1 vs 8 reward calls equal to 1e-6 on open reach, close reach, open pull; e restore diff 0, two restores identical) + prompt uniqueness; re-run after repair round 1 | `abc_sim/task_eval/drawer_{stages,subtask,starts}.py` | 2026-11-14 |
| 3 starts | G3 | PASSED (all 7 stages, every target ≥ 80 %) | valid/100 (top / middle / bottom): open reach 100 (29/29, 41/41, 30/30), grasp 100, pull 100, release 88 (26/29, 36/41, 26/30); close reach 100 (36/36, 36/36, 28/28), push 87 (29/36, 34/36, 24/28), release 100; idle-kept 100 % of built in every stage (10 chunks; held stages also keep the grip and q within 1 cm / q ≤ 1.5 cm); max build 5.1 s CPU (budget 10 s). Final sets after repair rounds (round 2: completable poses; G3-bottom round 1: narrower gripper openings — open grasp bottom had been 16/30) | `../simulation/outputs/subtask_rewards/drawer/starts/<mode>_<stage>.{npz,json}`, tool `build_starts.py` | 2026-11-14 |
| 4 scripted | G4/G5 | 5 stages pass both; open pull and close push pass G4, STOPPED at G5 | table below; 20 starts per stage for all behaviours + 20 starts per target for every solver | `../simulation/outputs/subtask_rewards/drawer/scripted/summary_<mode>_<stage>[_solver_<target>].json`, tool `scripted_rollouts.py`, `print_summary.py`; earlier rounds in `round0_before_repairs/`, `round1/`, `round2/`, `round3_pre/` | 2026-11-14 |
| 5 report | – | written | | `design.md` | 2026-11-14 |

## Overlay applied (commit 51506b8 on `subtask-drawer`)
Source: `../openwam-sim-eval/abc_tasks` @ d549e43 (read only). Equivalent of its `apply.sh` without assets:
- copied verbatim (cmp-identical): `abc_sim/models/yam_drawer_openclose_scene.xml`,
  `abc_sim/randomization/tasks/openwam_tasks.py`, `abc_sim/task_eval/openwam_tasks.py`,
  `abc_sim/task_runtime/wipe_whiteboard.py`.
- `abc_registry.diff` (sha1 3d07d6a…): applied with `git apply --include` to `randomization/registry.py`,
  `task_registry.py`, `task_runtime/registry.py`, `task_specs.py`. Its `task_eval/registry.py` hunk no longer applies
  (context moved by `bottles_subtask` / `mug_flip_*` / `put_relative_subtask` entries); the same import line and five
  `_TASK_EVALUATORS` entries were added by hand.
- Not applied: `abc_d0832d1_local_changes.diff` (reference only: Blender `live.py` env knobs, an older `mug_tree.py`);
  the headphone asset conversion (needs RoboDojo assets) → `hang_headphones` registers but has no models here.
  All 7 overlay tasks are registered (the overlay modules hold them together); only the drawer tasks were checked.
- Local addition: `prepare.py` `SIM_TASK_PACKAGES` gains `drawer_open` / `drawer_close` → `("drawer", "marker")`. Without
  it `prepare.py --sim-task open_drawer` exits "no asset map"; the scene XML still declares the marker meshes
  (no marker bodies), so the model does not compile without the `marker` package.
- `prepare.py --sim-task open_drawer` / `close_drawer`: installed `drawer` (5 KB) and `marker` (0.8 MB) into
  `abc_sim/models/assets`, which is a symlink to the shared `../abc/abc_sim/models/assets` (additive, two new dirs).

## Phase 0 findings (stock `DrawerOpenCloseEvaluator`, randomizer `DrawerOpenCloseRandomizer`)
- Scene: fixed three-drawer cabinet (position jitter x −6–0 cm, y ±8 cm, yaw ±0.25 rad), three slide joints
  (range 0–0.2 m, damping 5, no spring/frictionloss), drawer subtree mass 53–62 g. Handle = stem (1×3×1 cm) + crossbar
  (5×1.2×2 cm) 1.9 cm in front of the `*_handle` site; ~3 cm gap between bar and drawer front. Handle heights
  z = 0.959 / 0.875 / 0.791 (bottom handle 4 cm above the table). Handles at y −0.23…0.00 → mostly the right arm;
  home grasp-site distance to handles 0.30–0.48 m (right), 0.45–0.64 m (left). Control dt = 0.2 ms × 170 = 34 ms.
- Success rule: open — target slide q ≥ 12 cm and every other drawer ≤ 5 cm; close — target q ≤ 1.5 cm and others
  ≤ 5 cm. Read every step, instantaneous; evals stop at the first success. No check of release, rest or contacts.
  Non-success reward is progress capped at 0.99, halved while another drawer is > 5 cm.
- Resets (`dr_reset.py`, seeds 0–39 each task): 0 crashes, 0 non-zero rewards, 0 after one idle 32-action chunk
  (max drift 0.0). Targets top 10 / middle 12 / bottom 18; prompt = metadata prompt; close starts 14.1–18.8 cm.
- Hand-made states (`handmade`): q 11.9 cm → 0.99/False, 12.1 → True; close 1.6 → False, 1.4 → True; other 4.9 cm ok,
  5.1 cm → False (reward halved). Thresholds behave exactly.
- **Bouncing / momentum**: none possible. The drawer is quasi-static (τ = m/b ≈ 11 ms < one 34 ms step): a 2 m/s
  fling moves it 2.1 cm, 0.3 m/s moves 3 mm; no rebound off either travel limit (`fling`, both tasks). A drawer
  cannot coast past 12 cm or bounce out of the closed band; q only changes while something pushes it.
- **Held at scoring time**: the open success fires while the hand grips the handle in 18/18 scripted successes.
  Not a false positive here: with no spring, letting go leaves q unchanged; retreating with an open hand dragged the
  drawer *further* open (final q 0.20 in 9/18, and 2/20 "pull only 10 cm" rollouts crossed 12 cm during the retreat).
  No scripted success was lost after release + 1 s (`lost_after` empty, ≤ 1 success flip per rollout).
- **Partial opening**: pull 10 cm + release → no success in 18/20 (final 9–11 cm); the 2 successes are real (drag
  during retreat, final 19–20 cm).
- **Wrong drawer**: overlay failure script (open the neighbour) 0/20 success; neighbour reached up to 22.4 cm (the
  soft joint limit lets the 0.2 m range be exceeded by ~2 cm under force — irrelevant to the thresholds).
- **Initial states already satisfying**: none (0/80).
- **No-grasp opening works**: a scripted hook (closed fingertips behind the crossbar, beside the stem) opens the
  target in 12/20 (middle 9/9, top 2/4, bottom 1/7) with stock success. Close is naturally a no-grasp push (overlay
  solver: fingertips on the crossbar face).
- Solver coverage from the overlay scripts (IK, not tuned): open 18/20 (bottom 5/7), close 12/20 (**bottom 0/7**: IK error < 1 cm,
  but the push stalls at 4–15 cm, 4 cm above the table; seed 13 even dragged it to 20 cm). Bottom drawer is the hard case.
- Contact flag chatters: in the close push at 8 cm/s the hand–drawer contact read at step boundaries was on in only
  2 of the 6 steps before success (seed 4; `dr_close_touch.py`); 7/12 close successes had a contact at the first
  success step. Stage "hand on drawer" tests need a window or a distance band, not one-step contact.

Scorer fixes: none. The stock check is correct for this physics; release/at-rest is added in the stage goals
(final `release` stage), not in the stock scorer, so eval numbers stay comparable with eval set v5.

## Baseline
None. No recorded rollouts for these tasks under `../simulation/outputs`, `../simulation_outputs`,
`../abc_eval_records` or `../openwam-sim-eval` (grep for task names / prompts: only this audit's files). The tasks are
in OpenWAM eval set v5 (open: worlds 0,1,2,6,7; close: 0,1,3,6,7; seed 20260511, budget 945 steps), run from
as20482's `abc_tasks_dev` checkout, whose outputs are not readable from this account. There are no expert demos for
either task, so no recorded-rollout grasp source (phase 3 source 2) exists.

## Stages
| stage | status | starts valid /100 | solver +1 (20 starts; per target 20 each t/m/b) | hacks (planned rollouts behaving as EXPECT; +1 share) | margin |
|---|---|---|---|---|---|
| open reach | **PASSED** | 100 | 20/20; 20/20, 20/20, 20/20 | idle 0 20/20; move away 0 20/20; wrong handle 0 20/20 (+0.31); wrong pull −1 "wrong drawer" 18/18 | 1.02 |
| open grasp | **PASSED** | 100 | 20/20; 20/20 ×3 | idle 0 20/20; close in air 0 20/20; hook (alt, D2) +1 13/13 planned | 1.00 |
| open pull | **STOPPED (G5)** | 100 | 19/20; 20/20, 17/20 (3 unplanned), 19/20 | partial 10 cm 0 19/19 (+0.39); let go early −1 "back to" 20/20; **wrong drawer 3/11** | 1.01 |
| open release | **PASSED** | 88 | 20/20; 19/20, 20/20, 19/20 | keep holding 0 20/20; push back in −1 "back to" 19/20 (+1 1/20) | 1.00 |
| close reach | **PASSED** | 100 | 18/20 (18/18 planned); 17/20, 15/20 (15/15 planned), 15/20 (15/17) | idle 0, move away 0 20/20; grab and pull (alt) +1 19/19; wrong drawer (far) −1 "wrong drawer" 15/15; adjacent drawer (info) +1 15/20 | 1.14 |
| close push | **STOPPED (G5)** | 87 | 20/20; 20/20 ×3 | partial 30 % 0 20/20 (+0.16); push then pull open −1 19/19; **wrong drawer 0/15 (round 2), 0/20 planned (round 3)** | 1.10 |
| close release | **PASSED** | 100 | 20/20; 20/20 ×3 | stay 0 20/20; pull open while leaving −1 "back to" 18/20 (+1 2/20 = 10 %, at the ceiling) | 1.00 |

Bottom drawer (D7): passes G3 and G4 in every stage after the narrower-opening start fix (solver 15–20/20), so no
per-target stop.

**Stop reasons (G5, after 3 repair rounds each):**
- **open pull** — the wrong-drawer hack (the other arm opens a neighbour while the first holds the target) cannot be
  executed: both arms work stacked handles at the same y, and in 5/11 planned rollouts the arms collide and knock the
  grasp loose ("back to grasp/reach") before the neighbour moves; in 3 the neighbour is never grabbed (0); 9/20 not
  plannable. In the 3 rollouts where the neighbour passed 5 cm, −1 fired as "wrong drawer opened". Tried: neighbour
  12 cm (r1), pitched pulls + planned-only EXPECT share (r2, passed 11/13 on the earlier start set), holding arm still
  (r3). Evidence `scripted/summary_open_pull.json`, `round2/`.
- **close push** — same hack: the other arm reaching an adjacent drawer knocks the protruding open target, so the bar
  leaves the pushing hand ("back to reach") before any wrong drawer (0/15 for the intended reason, others_max ≤ 9 cm
  in 2); round 3 routed it low to the far drawer, which has no IK for the other arm on any of the 20 starts. Evidence
  `scripted/summary_close_push.json`, `round3_pre/summary_close_push_strict.json`.
In both stages the reward itself showed no fault: the solver earns +1 (19/20, 20/20), every other hack behaves, the
wrong-drawer −1 is covered by G2 test (c) (synthetic neighbour at 6 cm, every stage) and by the single-arm
wrong-drawer hacks of open reach (18/18) and close reach (15/15). **Decision for the user:** accept these two stages
on that evidence, or keep them stopped until a two-arm wrong-drawer hack can be executed.

## Decisions (made with or for the user)
- **For review (taken by me, conservative):**
  1. Pull starts are grasps only (hook starts unstable at rest, see Phase 2–3 notes); hooks still count for the goals.
  2. Contact-based "near" needs the pinch ≤ 5 cm (round 1) — a tightening of the approved "or a finger on the
     handle / hand on drawer".
  3. Release "away" measured from the whole hand (palm–fingertips), not the pinch (round 2).
  4. Starts must be kinematically completable (stage motion has IK from the start pose) and touch nothing but the
     target drawer.
  5. G5 EXPECT shares are over planned rollouts (≥ half the starts planned); the +1 ceiling stays over all starts;
     a −1 hack must fail for the reason it tests. The table's close-reach "grab and pull further open → −1" is run as
     an alt (+1 expected): the reach goal fires on the grab before any pull can happen, by the approved goal
     definition; the pull-past failure is exercised in push (19/19). Close-reach wrong drawer uses the far drawer;
     the adjacent variant is reported as info (+1 15/20: its route touches the target's front, the reach goal).
  6. The pull solver pulls 15 cm (17 cm in the overlay is out of reach near the arm base) and may pitch the wrist
     ≤ 20° about the bar.
  7. Open pull and close push STOPPED at G5 (see Stages) — the user may accept them on the evidence given.
- **G1 approved by the user 2026-11-14: every default option in `stage_table.md` "Decisions for you" accepted as written:**
  1 release stage yes; 2 hook opening counts (pull goal by any contact, pull starts may be hooks); 3 close by pushing,
  no close grasp stage; 4 wrong drawer −1 at 5 cm (stock); 5 close: target pulled > 3 cm past its start → −1;
  6 prompted drawer only; 7 keep all three targets, report gates per target, STOP bottom per target if it fails after
  repairs; 8 keep all seven overlay tasks registered.
- Taken in phase 0: keep the stock scorer unchanged (no false positive found); apply the whole overlay rather than a
  drawer-only subset (the overlay modules hold all seven tasks).

## Proposed kit changes (not made; `stagekit/` is shared)
- `stagekit` assumes free-jointed objects (`ObjectScene(model, [(name, free_joint_id)])`, `hold_object`,
  `set_object_pose`). Drawers need an articulated target: a `JointTarget`/`HandleScene` that observes a slide/hinge
  joint q, handle geoms (contact by finger), and a handle frame; and a `grip_handle(env, arm, T_handle_link6)` that
  IK-teleports the hand, closes on the fixed handle and checks both fingers in contact (no gravity hold check).
  Alternatively keep it task-local in `drawer_stages.py` / `drawer_starts.py` first and promote later.
- A contact-window helper (`touched within last k control steps` from per-substep contacts, or a distance band) for
  quasi-static pushes, where step-boundary contacts chatter.

## Phase 2–3 notes
- Seeds: open SEED_BASE + i = 980_000_000 + i, close 985_000_000 + i, i = 0..99 (same seeds for every stage).
- Kit not changed. Drawers are articulated, so the scene is task-local (`DrawerScene` in `drawer_stages.py`: slide q,
  handle = first two box geoms of each drawer link, contact windows keyed by sim time) — the "proposed kit changes"
  below are still only proposals.
- Close-stage gripper poses use fingers closing vertically (as the open grasp). The overlay's horizontal closing
  (wrist rolled 90°) has no IK once the drawer is within ~8 cm of closed (rot err 0.1–1 rad at q = 0 on 6/6 seeds,
  vertical 0) — this is very likely why the overlay close solver stalled (bottom 0/7).
- Hook starts for pull (approved D2 "pull also starts from hook states") tried and **not used**: closed fingertips
  pressed 2 mm into the back of the bar at q 2.5–4 cm lose contact after settling (the quasi-static drawer is pushed
  off), so 10/12 classify as grasp, not pull. Pull starts are grasps only; hooks still count for the grasp/pull goals.
  Decision for review.
- `build_starts.py` times builds in CPU time (`time.process_time`), as instructed for the shared node.

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
| G5 (open reach, close reach) | 1 | `wrong_handle` hack +1 on 7/7 middle targets: the open upper finger of a gripper at the bottom handle grazed the middle bar with the pinch 6.8 cm away; close `wrong_drawer` +1 18/20: fingers reaching for a neighbour hit the open target's bar with the pinch 7–9 cm away | goal tightened: a finger/hand contact counts for "near" only with the pinch ≤ 5 cm (the exit band); once in push, any hand contact keeps it (`classify_open/close`) |
| G4/G5 (all) | 1 | most "unplanned" rollouts: the planner demanded ≤ 0.05 rad wrist error at every waypoint (open pull solver planned 0/20) | planner falls back to ≤ 0.3 rad endpoint orientation error (the overlay planner checked position only) |
| G5 (open release, close release) | 1 | `push_back_in` / `pull_open_leaving` backed off 3–4 cm first and legitimately met the release goal (+1) before the hack | hacks rewritten to act without first leaving: push in with the open hand; open on the bar face, slide over the bar, close, pull |
| G5 (open reach/pull, close push) | 1 | wrong-drawer hacks pulled the neighbour ≤ 4.8 cm (0, not −1) | neighbour pulled 12 cm |
| G5 (open release) | 2 | `push_back_in` +1 on 9/20: pushing in with the open hand slid the bar 4–5 cm between the fingers to the palm; no contact for 5 steps and the *pinch* ≥ 5 cm from the bar → release goal, though the gripper still surrounds the handle (drawer only moved to 12.3–14.9 cm) | bug fix: release "away" (goal and progress) measured from the gripper's palm-to-fingertip segment (`hand_dist`, site x 0–6.5 cm), not from the pinch; hack pushes 15 cm (bar reaches the palm first) |
| G4 (open pull; open/close release; close push) | 2 | solver unplanned: open pull 19/20 (pull endpoint 17 cm near the arm base), open release 7/20, close release 9/20 (10 cm back-off) | start setup: a start is kept only if its own stage motion has IK from it (`follow_points`: pull to 15 cm, back off 10 cm, push to closed; wrist pitch ≤ 20°); solver pulls 15 cm (goal 12 cm) and may pitch the wrist about the bar ≤ 20° (back-off ≤ 30°) |
| G5 (close reach) | 2 | `wrong_drawer` (adjacent neighbour) +1 15/18: under/over the open target the route touches the target's front with the pinch 3.4–4.9 cm from its bar — the reach goal state itself, not a reward flaw | gated hack uses the farthest drawer (top↔bottom, middle→top); the adjacent variant is reported as `info` (decision for review) |
| G5 (all) | 2 | hack/alt EXPECT shares were divided by all starts, so IK-unplannable rollouts (never executed) counted as misbehaving | EXPECT share over planned rollouts, and only if ≥ half the starts planned; the +1 ceiling stays over all starts (decision for review) |
| G3 per target (open grasp, open release bottom) | 1 | bottom drawer 16/30 (grasp), 21/30 (release): a fully open lower finger hits the table (handle 4 cm above it) and the hand is pushed 3–4 cm off the bar while settling | start setup: reject poses where the gripper touches anything but the target drawer; try openings 1.0 → 0.6 → 0.4 → grasp bottom 30/30, release bottom 26/30 |
| G5 (close reach) | 3 | far-drawer hack still +1 6/20 (top targets: the route down to the bottom handle passes the open top drawer's front) | hack routed out to 30 cm in front of the far handle, then in at 5–15° → 15/15 planned −1 "wrong drawer" |
| G5 (open pull) | 3 | wrong-drawer hack 9/12 (75 %); with EXPECT now also checking the failure reason (a −1 hack must fail for the reason it tests): 3/11 | holding arm kept still (its 4 cm pre-pull slipped the grasp) → still 3/11 → **STOPPED** |
| G5 (close push) | 3 | with the reason check the wrong-drawer hack was 0/15 (all "back to reach": the other arm knocks the open target) | other arm routed low to the far drawer → 0/20 plannable → **STOPPED** |
