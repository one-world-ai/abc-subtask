# sweep_away_paper_scraps_from_table — single-stage subtasks (hand-pick chain H)

Status 2026-11-14: **reach, pick, carry, drop all PASSED** (CPU, single world, progress reward, 32-action chunks).
The tool chain (brush + dustpan) was **deferred by the user at G1** and is not built. Ledger: `STATUS.md`.

## Why a hand-pick chain
The bin is a tapered can whose rim (z ≈ 0.94–0.95) is 20 cm above the table, so scraps can't be swept into it
across the table. Every scrap the exp12 baseline (7/60 with the fixed scorer) put in the bin was picked by a gripper
and lifted over the rim (`probes/sw_tooluse.py`). The user chose to build this chain first (decision 1). Its semantics
match the approved bottles table exactly.

## Stages
One episode = one **target scrap**: the unbinned *active* scrap nearest a gripper's pinch point at the start. Only the
randomizer's active scraps (metadata `trash_joints`, 2–4) take part. The inactive ones are parked below the floor,
never become targets and never trigger a failure.

| stage | prompt (spec / live) | start (precondition) | goal (= next start) | progress (span) |
|---|---|---|---|---|
| reach | `sweep subtask: move a gripper to the target paper scrap` / `move a gripper to paper scrap n` | settled reset | pinch point ≤ 5 cm from the scrap CoM (exit 7 cm), or a finger on it | −dist (0.30 m) |
| pick | `… grasp the target paper scrap and lift it` / `grasp paper scrap n and lift it` | open gripper around the resting scrap | both fingers of one gripper, lifted ≥ 3 cm | 0.25·fingers + 0.5·lift frac·grasped (1.0) |
| carry | `… carry the target paper scrap over the bin opening` | held 6–12 cm above its spot | over the opening (radius ≤ r_top − 2 cm, above the rim) while touched | −3-D gap to the opening region (0.40 m) |
| drop | `… release the target paper scrap into the bin` | held over the opening | fixed stock in-bin, no finger contact, at rest, supported by the bin (directly or through scraps resting on it) | −radial gap + 0.05·released (0.10 m) |

**Failures (−1, episode ends):** bin moved > 5 cm or tilted > 20° (the scorer itself keeps its 45° upright rule);
a previously binned scrap knocked out; any active scrap off the table or on the floor (5 cm below rest, outside the
bin); in reach/pick, the target pushed > 5 cm while not grasped or entering the bin region unheld; from carry on,
released anywhere except over the opening or inside the bin, or landed at rest outside the bin; falling back to an
earlier stage.

**Reward per chunk** (`chunk_reward`, `reward_mode="progress"`, shaping_total 0.5): +1 at the goal, −1 at a failure,
otherwise 0.5·Δprogress/span. Worked example for carry: the scrap starts 0.20 m from the opening region, the policy
brings it to 0.05 m in 3 chunks, then reaches it. The shaping sums to 0.5·0.15/0.40 = 0.1875 whatever the path, plus
+1 at the goal, for 1.1875 in total. An idle episode earns about 0.

## Measurement choices (for review)
- **Pinch point, not grasp site.** On the YAM gripper the finger pads close 4.4 cm from `<arm>_grasp_site` (link_6
  −y). The scene measures `dist` from `PINCH = (0, −0.044, 0.115)` in the link_6 frame. Holding a 3.5 cm ball at the
  site misses it with both fingers (0/12 holds, `probes/sw_hold.py`).
- **Pinch depth 0.115.** At 0.125, 2/8 over-bin holds slipped within 10 idle chunks (one fell in and idle got +1).
  At 0.115, 8/8 held. Table picks centre the scrap 1 cm from the fingertips (0.137) so the tips clear the table.
- **Settled starts.** Reset scraps keep rolling 1–4 cm for about 100 actions, so starts are taken after 128 idle
  actions. Use `env.reset_to_stage`; a plain `env.reset` on `__reach` is not settled.

## Starts (`sweep_starts.py`, seeds 970_000_000 + i)
| stage | source | valid/100 | idle kept (held) | max build |
|---|---|---|---|---|
| reach | settled reset | 100 | – | 3.1 s |
| pick | IK teleport, open, around the resting scrap (wrist yaws 0, ±0.39, …) | 93 | – | 4.0 s |
| carry | lifted 6–12 cm, pinch grasp closed on the weightless scrap, gravity hold check | 89 | 89/93 | 3.3 s |
| drop | carry chain (33) or sampled leaning over-bin grasp fallback (52) | 85 | 88/93 | 6.2 s |

Rejected seeds: IK/collision at the build pose, the bin bumped by the teleported gripper (pick idle), and slow slips.
The carry lean that plans is negative (−30°/−45° first). Zero or positive leans never plan from the held start.

## Scripted checks (20 starts each, `scripted_rollouts.py`)
| stage | solver +1 | solver mean | hacks (as expected / mean return) | margin |
|---|---|---|---|---|
| reach | 20/20 | 1.533 | idle 20 / 0.000, away 20 / −0.011 | 1.532 |
| pick | 20/20 | 1.254 | idle 20 / 0, close_only 20 / 0.259, push 20 / −0.975 | 0.995 |
| carry | 15/20 (15/15 planned) | 0.926 | idle 20, release 20 / −1, wrong_way 20 / −0.060, shake 20 | 0.926 |
| drop | 20/20 | 1.225 | idle 20 / 0, release_beside 16/20 (4 unplannable) / −0.830 | 1.225 |

No hack earned +1 on any start. Pick push needed one repair round: the plan now tries several push directions.

## Codemap
- `abc_sim/task_eval/sweep_stages.py`: `SweepScene` (active mask, pinch distance, bin frame, support graph),
  `classify`, `progress`, `StageConfig`.
- `abc_sim/task_eval/sweep_subtask.py`: `SweepSubtaskEvaluator` (active set from `set_active_trash_joints`, target,
  failures, prompts). Registered as `sweep_subtask`, specs `sweep_away_paper_scraps_from_table__{reach,pick,carry,drop}`.
- `abc_sim/task_eval/sweep_starts.py`: `build_start`, `pinch_grasp`, `carry_to_opening`, `sampled_drop_hold`.
  Reuses bottles' `move_link`/`idle_action`.
- `tests/test_sweep_subtasks.py` (18 checks), `tests/test_sweep_scorer.py` (8, phase 0).
- Tooling: `../simulation/experiments/subtask_rewards/sweep/` (untracked). Outputs:
  `../simulation/outputs/subtask_rewards/sweep/h/`.

## Open questions
- The tool chain (brush + dustpan, decisions 2–5 and 8–10 of `stage_table.md`) remains to be decided and built.
- Slip-prone over-bin holds can turn idling into +1. They are excluded from the start set, but RL might find them.
- Drop starts lean on the sampled fallback (52/85). A better carry planner would make more of them chained.
- Should 45° stay the scorer's upright threshold (decision 7, second part)?
