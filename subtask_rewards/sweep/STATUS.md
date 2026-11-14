# sweep_away_paper_scraps_from_table — subtask rewards ledger

short name: `sweep` · template: A per scrap (chain H, approved); tool chain E deferred · seeds: SEED_BASE = 970_000_000 (starts on 970_000_000 + i, i < 100) · baseline: exp12 canon 14B 10/60 recorded → **7/60 re-scored** (mean final 0.381 → 0.304)
status: chain H DONE — reach/pick/carry/drop all PASSED (CPU single-world, progress reward, 32-action chunks); tool chain (brush + dustpan) deferred, not built

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | PASSED after fix (2 false positives found and fixed) | resets 0/40 (old scorer: seed 2 scored 1/4); settle probe 2,407 kept drops: inside-upright 246/246 accepted, table rests 0/2,063 (old 27), inside-tipped-bin 0/96 (old 96); tests 8/8 (old scorer fails 5/8); exp12 replay bit-exact 60/60 | `tests/test_sweep_scorer.py`; `probes/sw_{reset,geom,settle,replay,old_scorer_tests}.py`; `../simulation/outputs/subtask_rewards/sweep/{reset40.log,settle/,rescore_exp12/}` | 2026-11-13 |
| 1 stage table | G1 | APPROVED by the user (decision 1 = hand-pick chain H first) | chain H: reach → pick → carry → drop per target scrap, bottles semantics; tool chain deferred (not built) | `stage_table.md` (section "Alternative chain H"), user message 2026-11-14 | 2026-11-14 |
| 2 code + tests | G2 | PASSED | `tests/test_sweep_subtasks.py` 18/18 (lookup, active set, qpos-only refused, bin move/25° tilt/tip, push, target and other scrap off table/floor, knocked-out binned scrap, inactive never a target, pick retreat, floating not settled, scrap against bin base not done, 1 big = 8 small moves, release checks, drop-from-above reaches +1, starts classify/idle/restore exact); scorer tests 8/8 unchanged; stagekit unchanged | `../simulation/outputs/subtask_rewards/sweep/h/tests.log` | 2026-11-14 |
| 3 starts | G3 | PASSED all stages | valid/100: reach 100, pick 93, carry 89, drop 85; held idle kept: carry 89/93 (95.7 %), drop 88/93 (94.6 %); max valid build 3.1 / 4.0 / 3.3 / 6.2 s; drop sources: 33 carry chain, 52 sampled over-bin fallback | `../simulation/outputs/subtask_rewards/sweep/h/starts/*.{json,npz}`, `*_build.log` | 2026-11-14 |
| 4 scripted | G4/G5 | PASSED all stages (pick G5 after 1 repair round) | solver +1/20: reach 20, pick 20, carry 15 (15/15 planned), drop 20. Solver − best hack mean return: reach 1.532, pick 0.995 (close_only 0.259), carry 0.926, drop 1.225. No hack +1 anywhere. Hacks as expected: reach idle/away 20/20; pick idle 20, close_only 20, push 20/20 (was 13/20); carry idle/release/wrong_way/shake 20/20; drop idle 20/20, release_beside 16/20 (4 unplannable, at the 80 % bar) | `../simulation/outputs/subtask_rewards/sweep/h/scripted/summary_*.json`, `round0/` (pick before repair); `validation.json` | 2026-11-14 |
| 5 report | – | written | CPU single-world; no RL run; no renders | `design.md` | 2026-11-14 |

## Phase 0 findings (stock `SweepAwayEvaluator`, randomizer `SweepRandomizer`)
- Assets present (`abc_sim/models/assets/{brush_flat,dustpan,garbage_can,paper_ball}`); scene compiles; 40 resets
  no crash; active scraps 2:10, 3:11, 4:19.
- **Rule (stock)**: a scrap is in the bin when its centre, in the bin's own frame (+y up), lies within the *rim*
  radius + 5 mm of the axis and between floor − 5 mm and rim + 5 mm; success = every active scrap in. Nothing about
  bin tilt, scrap at rest, or the robot.
- **Gap 1 — cylinder vs tapered can**: the inner wall is 6.9 cm from the axis at the floor and 8.5–9 cm at the rim
  (`sw_geom.py`; wall ~1 mm thick). A scrap lying on the table against the outside of the base has its centre at
  8.4–9.2 cm, inside the rim-radius cylinder and inside the height band → counted. Reset seed 2 scored 1/4 this way;
  26 of 931 random table rests next to an upright bin were accepted.
- **Gap 2 — tilt**: the cylinder follows the bin frame, so (a) under the lifted edge of a pushed/tilted bin a scrap
  on the table counts (DreamZero exp7 seed 970910683: bin tilted 20°, trash_1 on the table at 8.9 cm → recorded
  success at step 600; replayed bit-exactly with interleave stride 5 in both this checkout and `../simulation/abc`);
  (b) scraps inside a bin lying on its side (rests at 84°) counted 96/96.
- **Fix** (commit 35c9cc9, `abc_sim/task_eval/sweep.py`): wall radius interpolated between the measured bottom and
  top radii of `gc_inner` (per scale) at the scrap's height, + the stock 5 mm margin; bin must stand within 45° of
  vertical. New metrics `bin_up_z`, `bin_upright`, `bin_radius_bottom/top`. Separation after the fix: inside scraps
  ≤ 4.5 cm from the axis, table scraps ≥ 8.4 cm, threshold ≈ 7.4 cm at the floor.

| class (settle probe, 3 seeds) | n | old accepts | new accepts |
|---|---|---|---|
| inside upright bin, touching only the bin | 246 | 246 | 246 |
| resting on the table near an upright bin | 931 | 26 | 0 |
| inside a tipped bin (lying on its side) | 96 | 96 | 0 |
| on the table near a tipped bin | 1,132 | 1 | 0 |
| on the tipped bin's outside | 2 | 0 | 0 |

- **Not checked by the scorer (left as is, handled by stage goals)**: a scrap held by a gripper inside the bin below
  the rim counts; a bin lifted off the table or standing on the floor with scraps inside counts.
- **Old scorer still in**: `../simulation/abc` (exp7–12 eval checkout) and any RoboEnv copy — not synced (not asked).

## Baseline re-score (exp12 canon 14B step-57000, 60 worlds, bit-exact, `rescore_replay.py --metrics`)
| | recorded (old) | fixed scorer |
|---|---|---|
| successes | 10/60 | **7/60** (06, 14, 20, 38, 44, 54, 57) |
| mean final reward | 0.381 | 0.304 |
| mean max reward | 0.501 | 0.326 |

- Lost: world-31 (3/4 in; trash_2 on the table at 9.1 cm, z 0.766), world-34 (1/3 in; trash_1/trash_3 on the
  table at 8.6/9.2 cm, never lifted above z 0.90 < rim 0.94), world-55 (1/2 in; trash_2 at 8.5 cm, z 0.764, bin
  tilted 25°). All rejected scraps sit at table height against the bin base — hidden behind the bin in a final
  frame, which is why the RESULT.md visual check passed them. These three episodes were **cut short by the old
  false positive** (stopped at task_success), so their 0 is not a policy verdict.
- 30/60 worlds change reward somewhere; 27 have a lower max, mostly from scraps nudged against the bin base.
- The seven real successes keep their recorded success step exactly.
- **How the baseline solves it** (`probes/sw_tooluse.py`, 10 recorded successes): every binned scrap was
  gripper-touched and lifted above the rim (max z 0.98–1.12), i.e. picked by hand. Brush grasped in 10/10 worlds,
  dustpan in 6/10, scrap–dustpan contacts rare; no scrap carried in the pan.

## Stages
| stage | status | why (if stopped) |
|---|---|---|
| reach | PASSED | |
| pick | PASSED | |
| carry | PASSED | |
| drop | PASSED | |
| tool chain (grasp_dustpan … dump) | DEFERRED by the user (G1) | not built |

## Decisions (made with or for the user)
- Scorer: tapered wall + 45° upright rule (to confirm with decision 7 in `stage_table.md`).
- **G1 (user, 2026-11-14)**: decision 1 = build the hand-pick chain H first: reach → pick → carry → drop per target
  scrap (target = nearest unbinned *active* scrap to a gripper), exactly the approved bottles table: drop from above
  the opening allowed, the scrap must settle inside the fixed tapered-bin scorer; bin moved (> 5 cm or > 20°) /
  tipped, scrap off the table or on the floor, knocking a binned scrap out are failures. Reuse bottles code paths.
  The tool chain (brush + dustpan) is **deferred — not built**.
- Other decisions, table defaults as they apply to chain H: 6 scraps on the floor = −1 (every stage); 7 stages fail
  at > 5 cm / > 20° bin disturbance, the scorer keeps its 45° upright rule. Decisions 2–5 and 8–10 concern only the
  tool chain and stay open.

## Judgement calls made for review (not settled by the approved table; conservative option taken)
- **Gripper distance from the pinch zone, not the grasp site.** On this YAM gripper the finger pads close 4.4 cm from
  `<arm>_grasp_site` along link_6 −y (pads y −0.050…−0.038, z 0.067…0.147 in the link_6 frame; site at z 0.1347).
  A 3.5 cm scrap held "at the site" is missed by both fingers (`probes/sw_hold.py`: 0/12 holds). The sweep scene
  measures `dist` (reach goal ≤ 5 cm, exit 7 cm, reach progress) from the pinch point (0, −0.044, 0.115). This is
  stricter than the site for a ball; the bottles numbers (5/7 cm) are kept.
- **Pinch depth 0.115** (not 0.125): at 0.125, 2/8 drop holds slipped out within 10 idle chunks and one fell into the
  bin (idle earning +1); at 0.115 8/8 held. On the table the pick start centres the scrap 1 cm from the fingertips
  (0.137) so the tips clear the tabletop.
- **Settled resets.** Reset scraps still roll 1–4 cm for ~100 actions. Every start (reach included) is taken after
  128 idle actions, so the 5 cm push rule is not tripped by settling. A plain `env.reset` episode of
  `__reach` (no saved start) does *not* settle — use `env.reset_to_stage`.
- **"Scrap off the table" covers every active scrap**, not only the target (table: "scrap off table or on floor" is a
  failure); non-target scraps may be nudged freely otherwise. Inactive (parked) scraps never count.
- Target = nearest unbinned **active** scrap to a pinch point; arm = the scrap's side (y sign).

## Residual notes
- Drop starts: 52/85 come from the sampled over-bin fallback (grasp source 3) because the carry chain's IK path is
  unreachable from some held poses (positive/zero leans never plan; negative leans do). Carry solver itself 15/20.
- 5 built drop starts slipped and fell into the bin during the 10-chunk idle check (+1 for idling); they were
  rejected (not in the start set). Physically this is a release over the opening, which the approved table allows,
  but an RL policy could find slip-prone holds; watch idle returns in training.
- Tooling (`../simulation/experiments/subtask_rewards/sweep/{build_starts,scripted_rollouts}.py`) is untracked in the
  shared `../simulation` checkout, like the other tasks' tooling.

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
| G0 | 1 | reset seed 2 scored; table rests against the base and tipped-bin interiors accepted | tapered wall radius from `gc_inner` + upright ≤ 45°; `tests/test_sweep_scorer.py` |
| G5 pick | 1 | push hack unplannable on 7/20 starts (18 cm push along the closing axis: IK rotation error 0.22–0.32) → 13/20 as expected | try the closing axis, then ±0.5/±1.0/π rad rotated directions, 18 then 12 cm (still > the 5 cm push rule); 20/20 −1, mean −0.975 |
