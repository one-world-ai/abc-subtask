# hang_mug_on_mug_rack — subtask rewards ledger

short name: `mug_rack` · template: F (insertion / hanging) · seeds: SEED_BASE = 1_200_000_000 (950M is taken by nuts_bolts) · baseline:
exp12 OpenWAM canon 14B **0/10** (bit-exact re-score; 8 of the recorded 8/10 were cut short by old false positives)
status: DONE (2026-11-14) — **pick PASSED**; reach STOPPED by the letter of G5 (push hack reaches the touch goal);
align / insert / release STOPPED at G3 after 3 repair rounds (66 / 36 / 51 of 100 valid starts). G0 PASSED after a
scorer fix (threading rule); G1 PASSED (user approved all 8 default decisions, 2026-11-14).

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | **PASSED after 1 repair round** (scorer fix: a tree piece must pass through the handle hole) | resets 0/40 (before and after fix; lazy hole adds 0.46 s mean per reset only when measured); settles 567 kept (7 scenes, 5 holed models + 2 narrow-hole models): old rule accepted 59, of which 9 do not hang by the handle (15 %); new rule accepts 50, all threaded, 48 hold by the handle alone (2 also lean the body on a peg), rejects all 9; exp12 re-score bit-exact, 0/10 with both rules; tests 8/8 | `probes/mr_*.py`; `../simulation/outputs/subtask_rewards/mug_rack/{scene.json,holes.json,settle/,renders/,rescore_exp12*/,reset_threaded.log,rescore_settles_threaded.log,test_mug_tree_scorer.log}` | 2026-11-13 |
| 1 stage table | G1 | **PASSED** — user approved the table with all 8 default options (2026-11-14) | reach → pick → align → insert → release; SEED_BASE 1200M | `stage_table.md` | 2026-11-14 |
| 2 code + tests | G2 | **PASSED** | `tests/test_mug_rack_subtasks.py` 15/15 (starts classify, 10-chunk idle \|return\| < 0.05, exact restore, every failure −1, 1 big move = 8 small); `tests/test_mug_tree_scorer.py` 8/8 after the threaded-test speed-up | `../simulation/outputs/subtask_rewards/mug_rack/test_mug_rack_subtasks.log` | 2026-11-14 |
| 3 starts | G3 | reach **PASSED**, pick **PASSED**; align / insert / release **STOPPED** (3 repair rounds) | valid/100 (7 seeds skipped by decision 2 count as invalid): reach 93 (max 3.6 s CPU), pick 87 (max 3.1 s), align 73→67→67→66, insert 50→49→22→36, release 46→66→55→51 (round 0→3); idle-kept ≥ 93 % everywhere | `../simulation/outputs/subtask_rewards/mug_rack/starts/{<stage>.npz,.json,.log}`, earlier rounds in `starts/round{0,1,2}/` | 2026-11-14 |
| 4 scripted | G4/G5 | pick **PASSED** (1 hack repair); reach G4 passed, **G5 failed by the letter**; align/insert/release pilot only (stopped at G3) | reach solver 20/20, push hack 20/20 +1, margin 0.08; pick solver 18/20, margin 0.97, handle grasp +1 on 1/20; pilot solvers align 4/16 (11 IK-unreachable), insert 8/16, release 6/16 | `../simulation/outputs/subtask_rewards/mug_rack/scripted/summary_<stage>.json` | 2026-11-14 |
| 5 report | – | written | | `design.md` | 2026-11-14 |

## Phase 0 findings
Stock rule before this audit (`MugOnRackEvaluator`, commit 7b997b2): mug centre in a cylinder around the trunk
(r ≤ 11.5 cm, z band), touches a tree piece above the base within 2 mm, touches nothing else but other mugs, and
some tree touch point is on the handle ≥ 1 cm above the mug CoM. Not checked: whether the handle *holds* the mug,
rest (qpos-only), which peg.

- **Resets**: 40/40 seeds score 0 / success False (`mr_scene.py`, again after the fix with `mr_reset.py`). 1–3 mugs
  (1:10, 2:11, 3:19 of 40), 19 models, random yaw, scale 0.9–1.1, ≥ 10.8 cm from the trunk axis.
- **Scene geometry** (`mr_scene.py`, `mr_holes.py`): 6 pegs Ø 13.2–13.9 mm (median), 7.4–8.5 cm, 19–26° up, tips
  z ≈ 0.91/0.97/1.05; trunk Ø ≈ 27 mm. Handle hole in the handle's mid-plane (inscribed circle, scale 1): 16 models
  18–30 mm; **mug_1 8 mm, mug_7 11 mm, mug_10 12 mm — smaller than a peg**. mug_8's handle sits 4.5 mm off the
  cup's mid-plane. Rim Ø 70–111 mm vs gripper opening 95 mm (mug_1/4/7/10/12 too wide to span at the rim).
- **Regression tests as found**: 6/6 pass.
- **Random settles** (`settle_search.py`, mug_1 dropped near peg tips, 5000 tries × seeds 0,1,2,3,6,18,33 →
  567 rests touching only the tree; classified by `mr_classify.py` (geometric: peg through the measured hole /
  tree point inside the cup / handle touch / body) and by `mr_counterfactual.py` (physical: re-simulate with only
  the handle's collision pieces → stays = hangs by the handle; and without them)):

  | class (geometry) | n | old rule accepts | hangs by handle (physics) | new rule accepts |
  |---|---|---|---|---|
  | peg through the handle hole | 51 | 50 | 49 | 50 |
  | handle touches tree, not threaded | 13 | 7 | 0 | 0 |
  | tree inside the cup | 400 | 2 | 8 ¹ | 0 |
  | body/rim only | 103 | 0 | 0 | 0 |

  ¹ counterfactual artefact: with the body's collision removed the mug drops < 3 cm and the handle catches a peg.
  The 9 old-rule false positives: peg in the cup or body on a peg, handle merely leaning on the trunk/peg above the
  CoM (renders `renders/suspect/`, e.g. mug_7 seed 18 k1816). With the handle's collision removed they move ≤ 4 mm;
  with only the handle they fall 11–29 cm. **The existing regression pose `HANDLE_ON_BRANCH` is one of them**
  (handle only: falls 31 cm; without handle: 1.7 cm) — the old test enshrined a false positive.
  The one threaded hang the old and new rule reject (mug_4 seed 2 k1474) is within 2 mm of the base disc (stock
  "touches nothing else" rule; kept).
- **Fix (repair round 1)**: `abc_sim/task_eval/mug_tree.py` — hanging now also requires `threaded`: some point of the
  handle's hole lies inside a tree piece above the base. Hole per mug model from its collision mesh: 3 slices of
  the handle's mid-plane (through the cup axis, centred on the handle thickness), grid points outside every piece
  from which solid is hit outwards, inwards, up and down. Pieces are convex (face planes, ≤ 1 mm deviation). No
  scipy. Measured lazily (only for a mug already in band, touching the tree and touching it with the handle above
  the CoM), cached per reset: 0.02–1.2 s per model; resets cost nothing extra. New metric `mug_threaded_mask`.
  Independent check: on all 567 settles the scorer's threaded verdict equals `mr_classify.py`'s (different code:
  scipy hulls + flood fill).
- **Tests** `tests/test_mug_tree_scorer.py` 8/8: `HANDLE_ON_BRANCH` moved to rejections with 5 more handle-lean rests
  (seeds 0, 1, 6, 18, 33; models mug_12/9/10/7/8); 4 threaded real hangs from seeds 1, 2, 33 (models mug_9, mug_4,
  mug_8 with offset handle) accepted, one also after 2 s settling; every model has a measured hole.
- **Baseline**: `rescore_replay.py` on the 10 exp12 worlds, bit-exact (no drift) with both the 7b997b2 rule and the
  threading rule: **0/10, max reward 0**. Recorded 8/10 "successes" ended at steps 250–9850; 4 single-mug worlds were
  cut at 250–400 steps by the original band-only rule. What the policy did (`mr_replay_contacts.py` world-07):
  leaned mug_1 against the tree with its base on the table (handle touching above CoM, rejected for the table
  contact) and dropped mug_3 upside-down onto the tree (rejected, peg not through the handle). No world ever had a
  threaded handle. The baseline gives no grasp or insertion source.
- **Other checkouts with an older mug-tree scorer** (not synced): `../simulation/abc` (differs from this branch),
  RoboEnv managed patch. The fix changes benchmark numbers only where a mug is touched only by the tree.

## Feasibility (template F — can a scripted IK policy thread a peg through the handle?)
`probes/mr_insert.py`, privileged best case: goal = a settled real hang; mug teleported 7 cm out along that peg's
axis (+1 cm up) in the hang orientation; a mug-flip library grasp (same mug assets) placed with kit `hold_object`;
then a physical Cartesian slide along the peg axis (IK waypoints, no teleport), open, back off, settle, stock score.

| scene (model) | goal hangs | grasp attempts | IK/hold failed | slipped in slide | slid to goal | **hanging** |
|---|---|---|---|---|---|---|
| seed 1 (mug_9) | 4 | 60 | 49 | 7 | 4 | 0 |
| seed 2 (mug_4) | 2 | 60 | 56 | 0 | 4 | **2** |
| seed 33 (mug_8) | 4 | 104 | 93 | 4 (+1 slide IK) | 6 | **3** |
| **total** | 10 | 224 | 198 | 12 | 14 | **5** (19 % of 26 held, 36 % of completed slides) |

Reading: threading is **physically possible for a scripted policy** (mug_4: 2 of 4 completed slides end hanging,
threaded, released, at rest). But (a) library grasps rarely suit the hang orientation (handle up, cup axis near
horizontal): 80–90 % fail IK; (b) during the slide the mug collides with the peg and is displaced 1–2.5 cm from the
plan (grasp compliance), which with 2–8 mm radial slack ends un-threaded; (c) slips with table contact for low-peg
goals. A real solver needs hang-oriented grasp synthesis (rim grasp opposite the handle, approach along the cup
axis), closed-loop correction of the hole position from sim state, and upper-peg goals. Expected outcome under the
gates: reach/pick should pass (template A-like); align plausible; **insert and release are STOP candidates** (G4
needs ≥ 40 % solver success over 12+ starts and ≥ 70 % of planned; current best-case pilot: 5/26 held attempts,
5/14 completed slides, 0/11 on mug_9). Not yet evidence for a hard stop: the pilot uses no closed-loop correction.
Three models (mug_1/7/10, 3 of 19, ~16 % of dealt mugs) can never be hung (hole < peg).

## Stages
| stage | status | why (if stopped) |
|---|---|---|
| reach | **STOPPED (by the letter of G5)** | G3 93/100, G4 20/20. The approved reach goal is "grasp site ≤ 10 cm **or a finger touches it**", so the approved push hack (expect 0 / −1) touches the mug first and gets +1 on 20/20 (mean return 1.35 vs solver 1.43, margin 0.08 < 0.5). Not a leak: touching is the approved goal. No repair possible without changing the goal (not allowed). Same situation as clutter reach. Decision for the user. |
| pick | **PASSED** | G3 87/100 (max 3.1 s), idle 100 %; G4 solver 18/20 (2 slipped: back to reach); G5 margin 0.97; close_only 0, lift 2 cm + open 0, handle grasp −1 on 18/20 (moving to the handle leaves the pick region), +1 on 1/20 (5 %, ≤ 10 %). |
| align | **STOPPED (G3)** | 66–73/100 valid after 3 rounds (needs 80). Failures: rim pinch slips/creeps after the lift chain (5–8), the mug resting on the gripper palm without a finger contact (8), pick-chain IK (3), 7 skipped seeds. Pilot solver 4/16 (5 planned): the pre-insert pose with a table-pick rim pinch is IK-unreachable for 11/16 (wrist limits when the cup axis turns horizontal). |
| insert | **STOPPED (G3)** | 36–50/100 valid after 3 rounds. Hang placement collides for the low −x peg on most seeds; hang-orientation rim pinches slip/creep (weight across the pinch); the pulled-out held pose sags in idle (\|return\| ≥ 0.05 on 8–11). Pilot solver (closed-loop slide) 8/16 (8/15 planned, < 70 %); carrying a held mug over the trunk top threaded a peg 2 cm deep while held on 3/16 (+1 — a real insertion by the goal's definition, but > 10 % by the letter). |
| release | **STOPPED (G3)** | 46–66/100 valid after 3 rounds; build time is the limit (settle a hang per peg option, IK grasp search, hold checks: 24–32 over the 10 s CPU budget in rounds 0/2). Pilot solver (open + back off along the approach) 6/16: 4 released but not hanging (the mug slid off / rested on the base), 4 still touching after the back-off, 2 pulled back off the peg (back to pick). |

The STOP is per stage: pick ships. The phase 0 conclusion stands — threading is physically possible for a scripted
policy, but hang-oriented holds are fragile in this sim (rim pinch with the weight across the jaws), so neither the
start sets nor the scripted solvers reach the gates.

## Decisions (made with or for the user)
User approval 2026-11-14 (G1): all default options of `stage_table.md` "Decisions" accepted as written:
1. keep the threading rule in the stock scorer (benchmark change on this branch);
2. mug_1/7/10 (hole < peg) are never a target; episodes where they are the only remaining mugs are skipped;
3. one mug per episode: the nearest not-yet-hanging (threadable) mug;
4. pick: the handle must stay free (no finger on the handle pieces);
5. release = placed, not dropped (hanging at rest, no finger);
6. any free peg counts; align progress uses the best free peg;
7. five stages reach, pick, align, insert, release;
8. build reach + pick fully; align/insert/release are a pilot with the STOP rule applied per stage.
- SEED_BASE = 1_200_000_000 (the table's 950M is taken by nuts_bolts); starts use SEED_BASE + i.

Judgement calls made during phases 2–5 (conservative option, for review):
- Prompts: fixed strings ("mug rack subtask: ... the target mug") instead of "{color} mug": the scene does not
  name mug colours; the target is the nearest threadable non-hanging mug (decision 3).
- Pick regression: the kit's generic rule (falling back to reach at any time is −1), stricter than the table's
  "back to reach after a lift". Knocked over (cup > 60° from vertical, no finger) is −1 in reach **and** pick.
- Align/insert "dropped": no finger for 8 actions while not threaded ≥ 1 cm is −1 at once (stricter than the
  table's "> 2 chunks resting on the tree").
- Release progress: 0.025 × (2 − fingers) − 0.5 × min(depth lost since the start, 0.1 m).
- A mug threaded ≥ 2 cm and still moving with no finger keeps / enters release (it is settling); it ends done only
  via the stock hanging test at rest.
- Insert/release starts use synthesized grasps on a settled real hang (source 3), not the align solver (stopped).
- Pick hack `lift_open` was re-scripted (G5 repair round 1): the first version lifted 10 cm, i.e. reached the goal
  (+1) before opening. The new one lifts 2 cm (below the goal) and opens; the mug lands on the table → 0. Its
  expectation was changed from −1 to 0|−1 (the table's "lift then open → −1" assumed a lift above the goal).
- Start validity counts the 7 decision-2 skipped seeds as invalid (G3 letter), and `build_s` is CPU time.
- The stock scorer's `_threaded` now pre-filters hole points by each tree piece's bounding sphere (same verdict;
  scorer tests 8/8; per-step cost on a hanging mug ~20 ms → < 1 ms).

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
| G0 | 1 | handle-touch rule accepted 9/59 settled rests where the handle only leans on the tree (incl. regression pose HANDLE_ON_BRANCH) | require a tree piece through the measured handle hole (`threaded`); tests updated; lazy hole measurement (eager cost 2 s/reset) |
| G3 align/insert/release | 1 | align 73, insert 50, release 46 (32 over the 10 s budget); slow slips in idle | align chain: 2.5 cm pinch + one-chunk creep check; hang starts: creep check after the hold, 8 hold tries, setup steps run without the evaluator (~20 ms/step saved), threaded test pre-filter |
| G3 | 2 | align 67, insert 49, release 66 | creep threshold 1 mm/chunk, a finger (not both) required in the creep check, 12 peg×side options |
| G3 | 3 | align 67, insert 22, release 55 (1 mm creep and 12 options cost time) | align back to the 1.5 cm pinch (+3 tries), creep 2 mm, 8×3 grasp candidates (IK was ~75 % of build time), 8 s CPU cut-off → final align 66, insert 36, release 51: **STOP** |
| G5 pick | 1 | `lift_open` reached +1 (it lifted past the goal before opening) | lift 2 cm then open → 0; margin 0.97, G5 passed |
