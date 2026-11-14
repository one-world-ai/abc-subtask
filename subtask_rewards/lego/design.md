# LEGO sorting subtasks for single-stage RL: design + codemap

**Status (2026-11-14):** reach, pick, handoff, carry, drop PASSED. Cross-side blocks (target and its bin on
opposite sides of the table, ~half the scenes) now go through a **handoff** stage (amendment 3, user-approved
2026-11-14) instead of being excluded from drop. Gate numbers and the repair log are in `STATUS.md`; the approved
table and the amendment are `stage_table.md`.

**Code:** this worktree, branch `subtask-lego`.
- `abc_sim/task_eval/lego_blocks.py` — repaired stock scorer (measured collision cavity, upward opening)
- `abc_sim/task_eval/lego_stages.py` — `StageConfig`, `LegoScene.observe`, `classify`, `progress`
- `abc_sim/task_eval/lego_subtask.py` — `LegoSubtaskEvaluator` (target choice, latched failures, prompts)
- `abc_sim/task_eval/lego_starts.py` — `build_start` per stage, native-pad grasps, numerical-warning guard
- `abc_sim/task_eval/lego_handoff.py` — handoff grasp plans (giver top-down across the width, receiver on the end
  faces), the scripted handoff solver, cross-side carry/drop start setup
- `abc_sim/task_eval/handoff_stage.py` — the shared, task-independent handoff tests (used unchanged)
- `task_specs.py` / `registry.py` — `sort_lego_blocks__{reach,pick,handoff,carry,drop}`, evaluator `lego_subtask`
- tests: `tests/test_lego_subtasks.py` (15), `tests/test_lego_scorer.py`

**Tooling** (`../simulation/experiments/subtask_rewards/lego/`): `build_starts.py` (100 starts per stage +
10-chunk idle check), `scripted_rollouts.py` (solvers and hacks, `summary_<stage>.json`). Outputs and saved
starts: `../simulation/outputs/subtask_rewards/lego/`.

## Idea
Train one block at a time. Each episode fixes one **target**: the active, not-yet-sorted block nearest a gripper
at the start. Its matching-colour bin is the goal. All other blocks stay in the scene and only matter through
failures (knocked off the table, sorted ones knocked out, any block in a wrong bin). Stages come from simulator
state only (MuJoCo contacts, velocities, poses). Goal of stage k = start condition of stage k+1.

Two chains, fixed per episode from the rest spots at reset: **same-side** blocks (block and its bin on the same side
of the table, sign of y) `reach → pick → carry → drop`; **cross-side** blocks `reach → pick → handoff → carry → drop`.
Giver = the arm away from the bin (in practice always the picking arm), receiver = the arm on the bin's side.

## Quick start
```python
import abc_sim, numpy as np
env = abc_sim.make_env(task="sort_lego_blocks__pick", render_cameras=True)
z = np.load(".../outputs/subtask_rewards/lego/starts/round0/pick.npz", allow_pickle=True)
row = {key: z[key][i] for key in ("seed", "target", "arm", "state", "rest_height", "labels")}
obs, info = env.reset_to_stage(int(row["seed"]), row)  # exact restore; omit row to build online (~1.6 s)
prompt = info["prompt"]                                 # e.g. "grasp red block 3 and lift it"
ev = env._task_evaluator
done = False
while not done:
    for a in policy(obs, prompt)[:32]:
        obs, *_ = env.step(a)
    reward, done = ev.chunk_reward()
```
Saved starts: reach/pick `starts/round0`, handoff and carry `starts/round4`, drop `starts/round5` (100 each;
handoff only on cross-side seeds, the seed list is in the npz, exclusions in `handoff.json`). `info`/metrics
`cross_side`, `giver`, `receiver` tell the chains apart. The
spec prompt (lookup key) is the fixed `STAGE_PROMPTS` string; the per-episode prompt names the block.
The qpos-only batched evaluation path is refused: rest and contact need real `MjData`.

## Stages (`lego_stages.py`)
| stage | in it when (= start) | goal (= next start) | progress | failures (−1) |
|---|---|---|---|---|
| reach | none of the below | grasp site ≤ 5 cm from the block centre | −distance (span 0.30 m) | block moved > 3 cm (any contact); off table; bin moved > 5 cm or tilted > 20°; any block in a wrong bin; sorted block knocked out; another block off the table |
| pick | grasp site ≤ 5 cm (stays while ≤ 7 cm or a finger touches) | both fingers of one arm on it and lifted ≥ 3 cm | 0.25·fingers + 0.5·clip(lift/3 cm)·grasped (span 1.0) | moved > 3 cm without a grasp; back to reach; + reach's scene failures |
| carry | same-side: grasped and lifted. Cross-side: received (receiver alone, giver ≥ 6 cm clear), lifted | still grasped, centre inside the measured bin opening (inset by the block's projected half-extent + 5 mm) and above the rim | −gap to that opening box (span 0.40 m) | released while not over the opening; landed outside the bin; + scene failures |
| handoff (cross-side) | giver holds it (both fingers) lifted ≥ 3 cm, or shared `handing` (CoM in M = x 0.40–0.60, \|y\| ≤ 0.10, z 0.80–1.10, 10 cm exit hysteresis; lifted; giver both fingers or receiver ≥ 1) | shared `received_in_air`: receiver both fingers, giver nothing on it and its pad midpoint ≥ 6 cm from the block surface, lifted, touching nothing else | shared `handoff_stage.progress` − min(distance to M, 0.30) (span 1.55) | dropped (free > 4 cm up); touched anything but the arms; giver carried it > 10 cm past M on the receiver's side; + scene failures |
| drop | held/touched over the opening, or lowered inside the cavity | inside the matching cavity (repaired scorer), no finger contact, speed < 2 cm/s and < 0.2 rad/s, supported through the bin (directly or via resting blocks), debounced 5 actions | −opening gap + 0.05 when released (span 0.10) | release outside the opening; settled outside or in a wrong bin; + scene failures |

Hysteresis/debounce: labels must persist `hold = 5` actions; a block in free flight (released, still moving or
not touching anything) keeps its previous label, so a drop from above the opening is not counted as a regression
while it falls. Floating or still-moving blocks inside the cavity never count (the audit found 2 such moving
positives in 180 drops under the qpos-only scorer).

Two tightenings were made after approval (STATUS.md, "Decisions"): bare finger contact does not complete reach,
and any > 3 cm slide during reach fails. Both came from the side-push hack.

## Reward
`reward_mode="progress"`: +1 on the chunk the goal is confirmed, −1 on a failure (episode ends), otherwise
`0.5 × Δprogress / span`, so the in-between rewards sum to 0.5 × total progress / span and going back cancels
earlier shaping.

Worked examples (32-action chunks):
- reach, grasp site moves from 28 cm to 10 cm away over 4 chunks: shaping 0.5 × 0.18 / 0.30 = 0.30 in total
  (however it is split), then +1 when it gets within 5 cm → return ≈ 1.48 (the solver's mean is 1.483).
- reach, pushing the block with the outside of closed fingers: the grasp site stays 6–8 cm away, the block slides
  3 cm → −1 (side_push mean return −0.54: some shaping on the approach, then −1).
- pick, close the gripper without lifting: two fingers → 0.25 × 2 / 1.0 × 0.5 = 0.25, never +1 (close_only 0.250).
- drop, open the gripper over the opening: block falls, settles on the cavity floor → +1 (return 0.94 on
  average; on one start in 20 the released block landed outside the bin → −1).
- handoff, solver: the giver brings the block ~25 cm into M (progress −0.25 → 0: shaping 0.5 × 0.25 / 1.55 ≈ 0.08),
  receiver closes (+0.5), giver opens (+0.5) and backs off 6 cm (+0.25): shared progress 1.25 → shaping ≈ 0.40,
  then +1 → ≈ 1.45 (solver successes 1.45–1.48).
- handoff, both arms keep holding: shared progress stays 0.5 (receiver 2 fingers, giver still 2) → ≈ 0.19, never +1.

## Starts (`lego_starts.py`, SEED_BASE 930 000 000, 100 seeds per stage)
| stage | how | valid/100 | mean s |
|---|---|---|---|
| reach | plain reset; target = nearest unsorted active block | 100 | 1.3 |
| pick | open gripper teleported around the block (pads, not the grasp site, at the block centre; 8 yaw variants), 10 open-gripper steps must not touch or move it | 100 | 1.6 |
| carry, same-side | physical pick from the open pose, else sampled table-side hold (block lifted 6–12 cm, weightless close, gravity hold); then pad-axis centring ≤ 6 mm and ≤ 1 mm creep over 64 idle actions | 48/48 (round 4; round 1: 97/100 over both sides) | 5.2 (all carry) |
| handoff (cross-side seeds only) | the giver's sampled table-side hold (block lifted 6–12 cm, weightless close, gravity hold, ≤ 1 mm creep) = pick's goal | 100 (of the first 100 cross-side seeds; 216 scanned) | 2.3 CPU |
| carry, cross-side | handoff start + scripted handoff solver (33/52), else block moved into M and held by the receiver alone (19/52); ≤ 1 mm creep | 52/52 | ~7 CPU |
| drop, cross-side | receiver hold in M, then the carry solver with the receiver (`follow=True`, tilts 0/−30/30); ≤ 1.5 mm creep | 52/52 | ~4.5 CPU |
| (superseded) drop, same-side scope | sampled table-side hold (no physical pick: it was never pad-centred), then the carry solver at −30° / 0° approach tilts from one snapshot; same centring check, creep ≤ 1.5 mm; from the 2nd attempt a sampled over-bin hold with the bin-side arm as fallback | 100 (of the first 100 same-side seeds; 224 scanned, 106 cross-side excluded) | 3.9 CPU |

Every start is then idled 10 chunks; it must stay in its stage with |return| < 0.05 (all built starts did).
Numerical warnings, resets and non-finite states during building or scripted runs invalidate the rollout.
The finger pads sit 4.4 cm from `<arm>_grasp_site` (link_6 −y), so a held block is ~4.45 cm from the grasp site;
`native_grasp` places the measured pad midpoint, not the site.

## Checking the reward (scripted, first 20 starts per stage)
| stage | behaviour | expect | result |
|---|---|---|---|
| reach | solver (4 jaw yaws, approach from 10 cm above) | +1 | 20/20 |
| | idle / away (rotate shoulder away) | 0 / 0 | 20/20, 20/20 |
| | side_push (closed fingertips tilted 20–35°, grasp site kept ≥ 6 cm away, 12 cm sweep) | −1 | 20/20 |
| pick | solver (close, lift 8 cm) | +1 | 20/20 |
| | idle / close_only | 0 / 0 | 20/20, 20/20 |
| | push / one_finger_drag / lift_open | −1 / 0 or −1 / −1 | 18/20, 18/20 (2 unplanned each), 20/20 |
| carry | solver (lift above rim, move over the opening; 7 approach tilts, wrist follows the bearing for the receiver) | +1 | 19/20 (same 9/9, cross 10/11 after the handoff; was 3/11 with one arm) |
| | idle / shake / wrong_way | 0 | 20/20 each |
| | release / wrong_bin | −1 | 20/20, 19/20 (1 unplanned) |
| handoff | solver (giver to M, receiver end-face grasp, monitored release, giver backs off) | +1 | 14/20 (planned 15) |
| | idle | 0 | 20/20 |
| | drop_between / set_down_regrasp / giver_across (steps across as far as it reaches, lets go) | −1 | 19/20, 19/20 (1 unplanned each), 20/20 |
| | both_hold (giver never lets go) | 0, never +1 | 19/20 (return 0.19) |
| drop (both sides) | solver (open gripper) | +1 | 17/20 (same 8/9, cross 9/11; 2 cross-side blocks bounced out → −1; 1 jitters in the bin, 0) |
| | idle / hold_inside | 0 | 20/20 each |
| | release_beside / wrong_bin / tip_bin | −1 | 20/20, 19/20 (1 unplanned), 20/20 |

## Open questions
1. **Cross-table blocks — resolved by the handoff (2026-11-14).** About half the seeds put the target and its bin
   on opposite sides; they now go through handoff, and end-to-end start coverage is 100/100 (was 52.7 %).
   Open: "cross-side" is a y-sign rule, so centre-slot bins just past the middle also take the handoff although one
   arm sometimes reaches them; the handoff solver misses 6/20 (low receiver approaches close on nothing).
2. The reach goal is measured from the grasp site, which sits 4.4 cm off the pads; a 5 cm ball around the site
   allows hand poses whose pads are up to ~9 cm from the block. Measuring from the pad midpoint would be tighter
   and would need a `stagekit.scene` option (not changed here).
3. Start building uses a wall-clock retry budget (8.5 s), so marginal seeds can build or not depending on load.
   A fixed attempt count would make online `reset_to_stage(seed)` deterministic; the saved starts already are.
