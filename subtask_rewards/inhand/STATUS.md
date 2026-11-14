# inhand_transfer_item_to_other_side — subtask rewards ledger

short name: `inhand` · template(s): D (handoff) with A-style reach/pick/carry/place · seeds: SEED_BASE = 1_100_000_000 (930–980M are taken by lego…drawer) · baseline: exp12 1/10 (re-scored bit-exactly with the fixed scorer: 1/10)
status: **PARTIAL** — reach, pick, present, handoff, place PASSED; carry STOPPED at G5 (giver-shoves-receiver +1 on 2/20; fix on hold by the user 2026-11-14). Reach passes under the planned-runs G5 rule (user, 2026-11-14). G1 approved by the user 2026-11-14.

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASSED with findings (scorer fixed; randomizer spawn overlap left to the user) | resets 300/300 score 0 at reset, 0 crashes; scorer FPs fixed: held 19/19 → 0/19, touched 13/13 → 0/13, single-arm-held 12/12 → 0/12, success-while-moving 121/121 → 0; tests 5/5; exp12 replays bit-exact 10/10 (drift 0), 1/10 → 1/10 | `probes/ih_{reset,scorer,handoff}.py`, `tests/test_inhand_scorer.py`, `../simulation/outputs/subtask_rewards/inhand/` | 2026-11-13 |
| 1 stage table | G1 | **APPROVED** by the user (all 7 default options as written) | 6 stages, 7 decisions | `stage_table.md` | 2026-11-14 |
| 2 code + tests | G2 | PASSED | `tests/test_inhand_subtasks.py` 14/14 (a–e + task-independent handoff tests + closed-fist touch), `tests/test_inhand_scorer.py` 5/5; kit unchanged (mug regression not needed) | `abc_sim/task_eval/{handoff_stage,inhand_stages,inhand_subtask,inhand_starts}.py` | 2026-11-14 |
| 3 starts | G3 | PASSED all six | valid of the first 100 eligible seeds: reach 100, pick 100, present 99, handoff 87, carry 97, place 91 (strict first-100 seeds: 81/81/80/56/64/62 — the rest are approved exclusions); mean build 0.2–3.8 CPU-s, max valid 9.7 CPU-s; idle 10 chunks still in stage: 100/100/99/99/99/98 % | `../simulation/outputs/subtask_rewards/inhand/starts/round0/{reach,pick,present,place}`, `round1/handoff`, `round3/carry` | 2026-11-14 |
| 4 scripted | G4/G5 | G4 PASSED all six; G5 PASSED pick/present/handoff/place, FAILED reach/carry after 3 rounds | see Stages | `../simulation/outputs/subtask_rewards/inhand/scripted/round2/summary_{pick,handoff}.json`, `round3/summary_{reach,present}.json`, `round0/summary_place.json`, `round6/summary_carry.json` | 2026-11-14 |
| 5 report | – | written | | `design.md` | 2026-11-14 |

## Phase 0 findings
Stock rule (`InhandTransferOtherSideEvaluator`): item origin y·s ≤ −0.10 (s = +1 for a left/+y spawn), z ≤ table +
0.10, xy on the table; reward = travel fraction, latched 1.0 once success. Nothing about the robot, motion or history.

**Scene** (`probes/ih_reset.py`, seeds 0–299, 300 idle actions each): compiles, 0 crashes, success at reset 0/300,
reward at reset 0/300. Categories: can 29, grater 29, cream cheese 29, brush 26, ladle 33, pizza cutter 40, rolling pin
25, shaker 19, bottle 38, whisk 32; sides 135 L / 165 R. Assets are in-tree (`models/assets/task_inhand_transfer`).
- **Randomizer bug: spawn overlap.** The item origin is always put at z = 0.82 whatever its geometry, so 58/300 items
  start > 5 mm inside the table (all 29 graters, 9/26 brushes, 19/38 bottles). 40/300 are launched 1–9 m up; 3/300
  (graters) land on the far side and score success without the robot (seeds 227, 261, 298), 32/300 have reward > 0.05
  after settling. exp12 world-03 (brush) has a 6 mm overlap → a randomizer fix would change that benchmark world.
  Not fixed (changes ~19 % of benchmark episodes) — decision 5; start sets skip those seeds.

**Scorer false positives** (`probes/ih_scorer.py`, seeds 0–29, 8 overlap seeds skipped; stock → fixed):
| case | stock success | fixed |
|---|---|---|
| held by B top-down 3 cm above rest on the far side | 19/19 | 0/19 |
| resting on the far side, B's half-open fingers touching it | 13/13 | 0/13 |
| held by A alone ≥ 10 cm past the middle (no handoff) | 12/12 | 0/12 (still reachable: decision 1) |
| dropped from rest in the middle (|y| ≤ 8 cm, 10–25 cm up) | 0/88 | 0/88 |
| thrown from the middle towards the far side (0.4–1.2 m/s) | 78/88 end, 77 first scored mid-bounce | 78/88 end, 0 mid-bounce |
| dropped from 15–25 cm above the far side | 44/44, all first scored mid-bounce | 44/44, 0 mid-bounce |
With terminate-on-success, every "first scored while moving/held" ended the episode as a success.
- **Fix** (`abc_sim/task_eval/inhand_transfer.py`): success additionally needs **no robot geom touching the item**
  (contacts of the `left_arm`/`right_arm` subtrees) and **linear speed ≤ 2 cm/s** (`evaluate_model_data`, the path
  `env.evaluate_task` uses). The qpos-only `evaluate_qpos_batch` recomputes contacts with `mj_forward` but cannot check
  rest (metric `rest_checked: False`). New metrics `object_released`, `object_at_rest`.
- Not fixable statelessly (decisions, enforced in the subtask chain instead): drops/throws that come to rest in the
  band (2, 3), push-across and single-arm carry (1).
- Tests `tests/test_inhand_scorer.py` 5/5: resets 0; held low → not success (both paths); moving → not success;
  resting → success; held then opened + arm home → success.

**Baseline re-score** (`../simulation/experiments/subtask_rewards/inhand/replay_exp12.py`): 10/10 worlds bit-exact
(max state drift 0 at every model call), old == new at every recorded check. Success 1/10 → 1/10: world-05 (bottle,
left spawn) — A picks, in-air handoff at x 0.53, y 0.03, z 1.09 (both grippers at 19700/19790, A re-grabbed once in
between), B carries to y −0.56, releases ~11 cm up, item rests (0.0003 m/s, no robot contact) at step 19950. Other 9
worlds: A grasps in all, B never touches the item; max far-side travel 0.06–0.28 m. Outputs:
`../simulation/outputs/subtask_rewards/inhand/replay_exp12{,_newscorer}/`.

**Handoff feasibility** (`probes/ih_handoff.py`, seeds 0–59): item moved weightless to the middle (x 0.50–0.60,
y 0, z 0.85–0.95); A holds it (kit `hold_object`: weightless close, gravity hold, checks); B teleported open onto a
second grasp (must touch nothing); both hold 15 actions; A opens (15), lifts 10 cm and backs off 6 cm (30); B alone
holds 30 actions. Pass = B 2 fingers, A 0, A's grasp site > 6 cm away, drift < 3 cm since A opened, < 5 cm/s,
item touches only B. **47/60** (42 in the first run + 5 of 11 failing compact items in a retry with side–side grasps;
47/51 without graters and cream-cheese sticks):
| item | pass | grasps that worked |
|---|---|---|
| can | 9/9 | A top-down on the upper part, B side-on lower part |
| pizza cutter | 8/8 | two top-down end grasps (2 with B side-on) |
| whisk | 6/6 | two top-down end grasps |
| rolling pin | 5/5 | two top-down end grasps |
| ladle | 6/7 | two top-down end grasps |
| dish brush | 5/7 | two top-down end grasps |
| water bottle | 7/8 | A top-down on top + B side-on (3), or both side-on from their own sides (4) |
| cream-cheese stick | 0/6 | A's pinch slips or B finds no free part of a 3–6 cm × ≤ 1.5 cm block (all 4 grasp families) |
| salt shaker | 1/1 | both side-on (A upper part, B lower part) |
| cheese grater | 0/3 | A's grasp collides / no IK (also launched at spawn) |
After the handoff A's grasp site is 11–25 cm from the item, drift ≤ 2.7 cm, item centre z 0.83–0.95. Top-down grasps
need tilts up to 65° in the middle (wrist limits; put_relative saw the same). Outputs
`../simulation/outputs/subtask_rewards/inhand/handoff_{0..5,compact_0..3}.{json,log}`. Gripper fact (measured): fingers close along link_6 **x**.

## Stages
Final: reach → pick → present → handoff → carry → place (design: `design.md`). 20 starts per stage in scripted runs.

| stage | status | starts valid/100 | solver +1 (planned) | solver mean return | best hack mean return | margin | G3 | G4 | G5 |
|---|---|---|---|---|---|---|---|---|---|
| reach | **PASSED (planned-runs G5 rule, user 2026-11-14)** | 100 | 20/20 (20) | 1.455 | idle 0.000 | 1.46 | pass | pass | **fail**: push_across −1 on 14/20, 6/20 unplannable (IK); other_arm_reach −1 17/20; move_away/idle 0 20/20; 0 hack +1 |
| pick | PASSED | 100 | 19/20 (20) | 1.172 | close_only 0.246 | 0.93 | pass | pass | pass: close_only 0 20/20, one_finger_drag 0 20/20, lift_open −1 17/20 |
| present | PASSED | 99 | 19/20 (19) | 1.150 | idle −0.001 | 1.15 | pass | pass | pass: shake 0 18/20, idle 0 20/20, release_early −1 20/20, carry_past_middle −1 16/20 (4 unplannable) |
| handoff | PASSED | 87 | 16/20 (18) | 1.211 | release_stay 0.365 | 0.85 | pass | pass | pass: drop_between −1 19/20, both_hold never +1 20/20 (return 0.18), release_stay no +1 19/20 (1 +1 = 5 %), set_down_regrasp −1 20/20, idle 0 20/20 |
| carry | **STOPPED (G5, letter)** | 97 | 20/20 (20) | 1.142 | idle 0.001 | 1.14 | pass | pass | **fail**: regrasp_A −1 13/20 (5 stuck short against B's gripper, 2 +1 = 10 %: A shoved B's gripper, not the item, into the far zone); release_early −1 20/20, wrong_way 0 20/20, idle 0 20/20 |
| place | PASSED | 91 | 19/20 (20) | 1.165 | hold_low 0.268 | 0.90 | pass | pass | pass: release_high −1 19/20 (1 +1 = 5 %), hold_low 0 20/20 |

**Reach — was STOPPED by the letter of G5; PASSED under the planned-runs rule adopted by the user on 2026-11-14** (push_across 14/14 planned as expected, other_arm_reach 17/17, ≥ 10 planned each, 0 hack +1). Rounds 1–3 removed the real leak (round 0: the closed-fist push scored +1 on
17/20 because touching / being near is the reach goal) by requiring an open gripper to enter "near" and failing a
closed-fist touch; after that no hack ever scores +1 and every *planned* hack behaves as expected, but the push hack
cannot be planned on 6/20 starts (IK unreachable beside the item), so as_expected/tried = 70 % < 80 %. Not a reward
leak; the stage is usable in practice (as clutter reach). Evidence `../simulation/outputs/subtask_rewards/inhand/scripted/round3/summary_reach.json`.

**Carry STOPPED — G5 by the letter.** regrasp_A (A grabs the item again) is obstructed by B's gripper: rounds 1–3
(direct touch, aim inside the item, try the free end) reached 12, 14, 13/20 as expected. In round 3, 2/20 got +1:
A's arm pushed B's gripper (A never touched the item, so no "giver touched" failure) and the item, still held by B,
reached the far zone — a possible exploit of a "giver bumps receiver" kind, exactly at the 10 % ceiling. Solver 20/20,
release/wrong-way/idle all as expected. Evidence `../simulation/outputs/subtask_rewards/inhand/scripted/round{3,4,5,6}/summary_carry.json`. A fix would need a
new failure ("giver touches the receiver arm") — not in the approved table; left for the user.

## Decisions (made with or for the user)
- Phase 0 (agent): fixed the stock scorer (released + at rest) — the subtask place goal reuses it.
- **G1 (user, 2026-11-14): all default options accepted as written**: (1) handoff required, A carrying it past the
  middle is −1 in present; (2) handoff in the air only, table touch during handoff −1; (3) "set down" = release with
  the lowest point ≤ 5 cm; (4) cream-cheese sticks and cheese graters dropped from handoff/carry/place start sets, kept
  in reach/pick/present; (5) spawn-overlap seeds skipped, randomizer unchanged; (6) spawn-side arm gives, B touching
  in reach/pick is −1; (7) stock scorer unchanged. Stages reach → pick → present → handoff → carry → place; `handoff`
  kept task-independent (`abc_sim/task_eval/handoff_stage.py`).

- Phase 2–4 (agent, conservative, **for review**):
  - Start sets = the first 100 *eligible* seeds from SEED_BASE (decisions 4/5 skip seeds); the strict first-100-seed
    counts are recorded too (81/81/80/56/64/62). Same rule as put_relative's same-side carry.
  - Pinch centre instead of grasp site: "near" (reach) and "clear" (handoff) are measured from the finger-pad centre
    link_6 (0, −0.031, 0.125), not `grasp_site` (3 cm off the pads, measured on both arms). "Clear" = 6 cm from the
    item's *surface* — stricter than the table's 6 cm from the CoM.
  - Reach: entering "near" needs A's fingers ≥ 50 % open, and touching the item with a closed gripper in reach is −1
    (both stricter than the table; they close the push-hack leak).
  - Handoff exit hysteresis = 10 cm outside M (the table's "leaves M by > 10 cm → −1", implemented as back to present).
  - Carry: any release > 4 cm up is "dropped", also over the far zone (carry never pays for a drop).
  - Place: "released too high" = robot contact ends while the item's lowest point is > 5 cm up and nothing else
    touches it. Place starts whose item slips in B's hand during the 10 idle chunks (8/100) are dropped from the set.
  - Carry starts: 52 chained from the scripted handoff, 45 from grasp source 3 (B holds it alone in the middle).
- Open for the user: (a) accept reach/carry despite the G5 letter (no leak in reach; carry's 2/20 shove), or add a
  "giver touches the receiver arm" failure to carry; (b) whether to publish renders/Space tab (needs the org token).

## Proposed stagekit changes (not made; kit is shared)
- `ObjectScene` / `holding`: the YAM `<arm>_grasp_site` is 3 cm beside the finger pads (pads at link_6 y −0.047…
  −0.017); a kit-level pinch-centre constant would fix grasp-site distances for every task (mug flip's 10 cm "near"
  and put_relative's 5 cm absorb it today).
- `holding.hold_action`/`hold_object`: let the other arm keep its current target instead of going home, and a
  two-arm variant (both grippers on one item) — needed for handoff starts and the handoff solver.
- `stagekit.robot`: document the closing axis (link_6 x) and grasp-site offset 0.1347; a top-down grasp helper with
  tilt search (probe `ih_grasp.py`: `topdown_T6`, `side_T6`, `tilt_T6`, `reachable`, dense mesh surface samples for
  principal axes — collision meshes of long items have vertices only at their ends).
- `ObjectScene.observe`: per-arm finger counts (`hits.reshape(m, 2, 2).sum(2)`) as a named entry — every handoff
  test uses them.

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
| G0 | 1 | stock scorer: held / touched / moving items scored success | added released (no robot contact) + at rest (≤ 2 cm/s); 5 regression tests |
| G3 handoff | 1 | round 0: 73/100 (26 timeouts: end grasps of long items creep 2–18 mm/chunk in A's hand) | `mid_topdown6/9` families: both grasps 6/9 cm either side of the CoM → 87/100 |
| G3 carry | 1 | (round 0 aborted mid-edit) round 1: 59/100 — receiver grasps that cannot hold alone | (round 1 = handoff fix) |
| G3 carry | 2 | 67/100 — fallback ran only when the solver failed | source-3 fallback: B holds it alone in the middle |
| G3 carry | 3 | 67 → chain results rejected later (creep, not clear) skipped the fallback | 5 s cap on the chain, full validation, fallback on any rejection → 97/100 |
| G5 reach | 1 | push_across +1 17/20 (closed fist touching = reach goal); other_arm_reach hovered open | closed-fist touch failure; B closes |
| G5 reach | 2 | push_across +1 14/20 (closed fist is "near" before touching) | entering near needs ≥ 50 % open → 0 hack +1 |
| G5 reach | 3 | push_across 6/20 unplannable | motion IK tolerance 0.4 rad → unchanged 14/20 → STOP |
| G5 pick | 1–2 | lift_open stayed within the 6 cm exit (8/20), then 12/20 planned | lift 15/12/10 cm → 17/20 PASS |
| G5 present | 1–3 | carry_past_middle passed through M (+1) / unplannable (1, 4, 14/20) | path under/around M, wrist tilts, 0.4 rad tolerance → 16/20 PASS |
| G5 carry | 1–3 | regrasp_A obstructed by B's gripper (10, 12, 14, 13/20) | direct touch, aim inside, free end → STOP |
| G2/G3 all | – | receivers closed on nothing; held items missed | finding: grasp site is 3 cm off the pads → pinch-centre grasps (not a gate repair) |
