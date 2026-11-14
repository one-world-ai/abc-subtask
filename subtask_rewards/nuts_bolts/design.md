# Nuts-and-bolts subtasks for single-stage RL

Task: `sort_nuts_and_bolts` (env task `nuts_bolts_sorting`). The user approved reach → pick → carry → drop on
2026-11-13 (G1) and a **handoff** stage for cross-side parts on 2026-11-14: reach → pick → handoff → carry → drop. Gate numbers, stops and evidence paths are in `STATUS.md`; this file says how the subtasks work.

## Use

```python
import abc_sim
env = abc_sim.make_env(task="sort_nuts_and_bolts__carry", render_cameras=False)
obs, info = env.reset_to_stage(seed)          # builds a start (~1-10 s), or pass a saved start row
prompt = info["prompt"]                       # e.g. "carry the nut over the teal bin"
ev = env._task_evaluator
done = False
while not done:
    for action in policy(obs, prompt)[:32]:   # one chunk = 32 executed actions
        obs, _, _, _, info = env.step(action)
    reward, done = ev.chunk_reward()
```

- Suffixes `__reach`, `__pick`, `__handoff` (cross-side scenes only), `__carry`, `__drop`. Fixed task prompts are lookup keys
  (`"nuts bolts subtask: …"`); `ev.stage_prompt()` gives the per-episode wording (nut/bolt, teal/orange).
- Saved starts: `../simulation/outputs/subtask_rewards/nuts_bolts/final/starts/<stage>.npz` (seed, target, arm,
  full MuJoCo state, rest heights, labels). `reset_to_stage(seed, row)` restores one exactly.
- CPU, single world. The goal needs real contacts and real linear/angular velocities, so the qpos-only path
  (`evaluate_qpos_batch`) raises instead of scoring a scratch state with zero velocities. Batched MJWarp training
  is not validated.
- Mid-task starts have no frame history.

## Stages (one target part per episode; `nuts_bolts_stages.py`)

Target: the nearest active part not already in its bin at the reset, by grasp-site distance to either gripper. It
is frozen for the episode. Bin assignment is fixed: **nuts → teal (`nuts_sorting_bin`), bolts → orange
(`bolts_sorting_bin`)**. The stock scorer also accepts the inverse partition; the subtasks do not, so a mixed
layout is never rewarded.

| stage | in it when (= start precondition) | goal (= next stage) | progress / span |
|---|---|---|---|
| reach | none of the below | grasp site ≤ 5 cm from the part CoM (exit 7 cm), or a finger touches it | −distance / 0.30 m |
| pick | near or touched, not lifted | two fingers of one gripper on it and lifted ≥ 3 cm | 0.25·fingers + 0.5·clip(lift/3 cm)·grasped / 1.0 |
| handoff (cross-side only) | held by the giver with both fingers, or the receiver has a finger on it; lifted ≥ 3 cm (`handoff_stage.handing`, region = the table volume) | `handoff_stage.received_in_air`: receiver both fingers, giver no contact and its pinch centre ≥ 6 cm from the surface, lifted, touching nothing else | `handoff_stage.progress` − min(distance to the middle box, 0.30) / 1.55 |
| carry | grasped and lifted, not over the opening (cross-side: held by the receiver alone after the handoff) | touched while its whole collision extent is above the rim and its CoM is ≥ 1 cm inside the inner walls | −distance to that region / 0.40 m |
| drop | touched over the opening; or in the scorer box below the rim; or (once in drop) lowered inside the walls | in the assigned scorer box, no finger contact, speed ≤ 2 cm/s, angular speed ≤ 0.2 rad/s, support rooted in the assigned bin | −height of CoM above bin floor (clip 0.30) + 0.05 once released / 0.15 m |

- Geometry is read per randomized model: scorer boxes from the stock evaluator (`*_goal_region` sites), inner wall
  faces, rim and floor from the bin's collision boxes, all in the live bin frame. The rim (8.5–10.9 cm) is above the
  scorer box top (7.9–10.2 cm).
- Over-rim test uses the AABB of every collision primitive of the part in the bin frame, so a tilted bolt clears the
  rim only if its lowest point does.
- Support is a contact graph: a released, stationary part in the box touching the assigned bin's geoms is supported;
  support then spreads through stationary released parts of the same bin. Two airborne parts touching each other are
  not a root.
- A released part in flight keeps its previous label until it is at rest or in contact; a zero-speed apex does not
  count as landing.
- Debounce 5 actions (`hold`). Hysteresis: near 5/7 cm; opening exit 1 cm.
- Round-5 fix: before it, a held part lowered more than 1 cm below the rim but not yet in the scorer box fell back
  to carry (round-4 drop solver: 9/20 "back to carry"). Now once in drop, a part whose CoM is inside the inner
  walls stays in drop and may be released there. Entering drop still needs the part over the opening.

### Handoff (amendment 2026-11-14)
- Cross-side = the target's rest y and its bin's rest y have opposite signs. 157/1000 screened seeds; every one is a
  bolt (bolts sit at +y, the orange bin at −y); no nut target is ever cross-side under the nearest-part rule, so the
  nut release blow-ups never enter the handoff. Receiver = the arm on the bin's side, giver = the other arm.
- `abc_sim/task_eval/handoff_stage.py` is used unchanged; this scene provides `f_give/f_recv/touch_give/touch_recv/
  gap_give/env_touch/com/lift` per part (`NutsBoltsScene.observe`). `gap_give` uses the pinch centre link_6
  (0, −0.031, 0.125) and the part's collision primitives; it is only computed while the receiver holds it with both
  fingers (the only case `handoff_stage` reads it).
- Same-side parts are labelled exactly as before (unit test against a frozen copy of the old `classify`).
- Physics of the bolt handoff (probes, `../simulation/outputs/subtask_rewards/nuts_bolts/handoff/`): bolts stand on
  their heads, a top-down grasp centred on the CoM encloses the whole bolt between 6.4 cm-wide finger plates, so the
  giver must hold it **high on the shank** (≥ 1.6 cm sticking out below its fingertips). The receiver takes the
  lower part from the side (link_6 horizontal, plates down, pad strip 3 mm under the giver's fingertips); the bolt
  hangs along its axis and stays held when the giver opens (no MuJoCo warning in any scripted handoff). A horizontal
  presentation pivots out of the receiver's fingers; a flipped giver wrist is out of reach. Before the low release
  the receiver rolls 180° about its approach axis so the shank hangs below its fingers.

## Reward (`NutsBoltsSubtaskEvaluator.chunk_reward`)

Terminal (checked every physics step, reported at the next `chunk_reward`): +1 goal; −1 failure. Otherwise

    reward = 0.5 × (progress now − progress at the previous chunk) / span[stage]

Worked examples:
- Reach: gripper 24 cm → 12 cm from the nut over three chunks: 0.5·0.12/0.30 = **+0.20** in total, however many
  chunks it takes; touching it then ends the episode with +1.
- Carry: 12 cm closer to the over-rim region: 0.5·0.12/0.40 = **+0.15**.
- Drop: lower from 6 cm to 1 cm above the floor (−0.06 → −0.01): **+0.17**; releasing there (+0.05) adds
  **+0.17**; settling supported in the box gives +1. Since the 2026-11-14 amendment a release with the part's lowest
  collision extent > 5 mm above the floor is −1 "released too high", so the 6 cm release of the old example now ends −1. Holding still earns ≈ 0 (idle |return| < 0.01 in the final run).

Failures (−1, episode ends):
- any stage: either bin moved > 3 cm in xy or tilted > 20° from the reset; a part that was supported in its bin at
  the start left the box ("knocked a sorted part out"); any active part > 5 cm below its rest height outside the bins
  ("fell off the table"); target in the other bin's box ("wrong bin"); fall back to an earlier stage.
- reach/pick: target moved > 3 cm in xy without a grasp, or reached the drop/done labels without one ("pushed").
- handoff: `handoff_stage.failure` — "dropped during the handoff" (no robot on it > 4 cm up), "touched the table
  during the handoff" (any non-arm contact: table, bins, other parts); fall back to pick.
- carry/drop, cross-side: "giver touched the part again".
- carry/drop: every touched → untouched transition must be over the opening, inside the walls below the rim, or in
  the box below the rim ("released outside the opening"); a released part at rest outside the box
  ("landed outside the bin"). A regrasp is rechecked at the next release.
- drop only (amendment 2026-11-14): at a release the part's lowest collision extent must be ≤ 5 mm above the bin
  floor (`StageConfig.release_max_clearance_m`), else "released too high". Measured in
  `../simulation/outputs/subtask_rewards/nuts_bolts/amend/release_sweep*.json`: releases in the air blow up
  numerically about half the time; on the floor they settle.
- **numerical instability**: any new MuJoCo warning count (e.g. `BADQACC`), a time reset, or non-finite qpos/qvel
  since the episode began latches a finite −1 with `numerical_failure=True`. Every exp12 replay world emits a QACC
  warning, and MuJoCo silently resets the state when it happens, so these trials must never count as a goal.

## Starts (`nuts_bolts_starts.py`)

SEED_BASE 950,000,000; candidate i uses seed base+i, i = 0…99. A start must build in ≤ 10 s and survive ten idle
chunks with no terminal and |shaping| < 0.05; any MuJoCo warning during setup rejects the candidate.

- reach: the randomized reset scene.
- pick: open gripper teleported around the resting part (native-pad grasp frame, three pad heights × eight yaws);
  rejected if the open fingers touch or move the part.
- carry: part lifted 6–12 cm in place, native-pad grasp, `hold_object` (close on the weightless part, then hold
  under gravity).
- handoff (first 100 cross-side seeds): the giver holds it lifted 6–12 cm where it lay, grasp high on the bolt
  (CoM 0.5–1.3 cm below the fingertips, link_6 z .152–.160, `HIGH_PAD_Z`; 2.4–3.2 cm of the bolt sticks out below them).
- carry, cross-side: the post-handoff state synthesized (`receiver_hold`: receiver side grasp in the middle, giver at
  home). Chaining the scripted handoff (~460 actions) costs 16–19 CPU-s on this node, over the 10 s budget.
- drop, cross-side: the state the cross-side carry solver ends in, synthesized (`receiver_hold_over_bin`): the bolt
  head-up over the opening, lowest point 1–4 cm above the rim, receiver side grasp with plates up on the head end.
  (The old sampled top-down lowered hold lets the upright bolt slide out: 57–65/100 valid in three tries.)
- drop: carry start, then a physical carry to over the opening (`carry_to_opening`); if that cannot plan, the
  sampled fallback: part held 1.5 cm above the scorer box centre by the arm on the bin's side (`source` recorded).

Target coverage: the nearest-part rule picks a nut on the bins' side in 87/100 seeds and a bolt across the table in
13/100 (bolts are on the other side from their bin in every case). A single-arm carry cannot plan for the
cross-table bolts; the drop fallback silently uses the other arm, which amounts to a handoff. See STATUS.md
("cross-table coverage") — the user decides whether to add a handoff stage or scope carry to same-side targets.

## Scripted checks (`../simulation/experiments/subtask_rewards/nuts_bolts/scripted_rollouts.py`)

Each behaviour runs from saved starts through `env.step` with `chunk_reward()` every 32 actions; IK planning
failures count as misses; a numerically invalid trial counts as a miss for solvers and as "not as expected" for
hacks.

| stage | solver | hacks (expect) |
|---|---|---|
| reach | top-down approach above, then down to pre-grasp, open | idle 0, rotate away 0, fast one-finger strike −1 |
| pick | 3.5 mm lower, close, lift 8 cm | idle 0, close only 0, push −1, open-gripper lift 0/−1, one-finger drag 0/−1 |
| carry | raise above rim + part extent, translate over the opening | idle 0, release −1, wrong way 0, shake 0, wrong bin −1, free arm shoves the bin −1 |
| handoff | giver presents it upright in the middle (x 0.40–0.45, y ≈ 0, z 0.88–0.92), receiver approaches horizontally to a side grasp under the giver's fingertips (approach path collision-checked), closes; giver opens (monitored), lifts and backs off | idle 0, drop_between −1, both_hold 0 (never +1), set_down_regrasp −1, giver_alone (carries it towards the bin in 4 cm steps, opens where its reach ends) 0/−1 |
| carry, cross-side | roll 180°, raise, carry over the opening turning about the vertical | as carry; wrong way / wrong bin moves may turn too; wrong bin carried in 4 cm steps |
| drop, cross-side (side grasp) | lower in 1 cm steps until the hand meets the rim, pitch the hand down about its closing axis in 15° steps (lowering after each) so the fingers enter the bin, then the same slow open | as drop |
| drop | lower until the fingertips stop the hand (target 2 mm), open at 1/960 per action until 16 actions after release, then open fully | idle 0, release high (open over the opening) −1, release beside −1, wrong bin −1, tip bin −1 |

## Files

| path | role |
|---|---|
| `abc_sim/task_eval/nuts_bolts_stages.py` | config, scene measurements, support graph, `classify`, `progress` |
| `abc_sim/task_eval/nuts_bolts_subtask.py` | evaluator: target, failures, release checks, numerical guard, prompts |
| `abc_sim/task_eval/nuts_bolts_starts.py` | native-pad grasps, carry chain, fallback drop starts, checked physics |
| `abc_sim/task_specs.py`, `abc_sim/task_eval/registry.py` | four `sort_nuts_and_bolts__<stage>` specs |
| `tests/test_nuts_bolts_subtasks.py` | 13 unit tests (G2); run as a script |
| `../simulation/experiments/subtask_rewards/nuts_bolts/` | `build_starts.py`, `scripted_rollouts.py`, release diagnostics |
| `../simulation/outputs/subtask_rewards/nuts_bolts/` | `final/` starts + scripted traces, `round*/` history, logs |

Shared `stagekit/` is unchanged.

## Open questions

1. Drop release physics: a nut released in free fall after a firm grasp (finger penetration ~3.6 mm) often gets a
   huge angular acceleration with no contacts and MuJoCo resets the state. Resolved for the stage by the user's
   low-release amendment (drop PASSED 2026-11-14); a scene-level numerics fix is still open for any other release.
   The scripted drop solver needs ~10–16 chunks (slow open), more than the spec's default max_chunks = 10.
2. Cross-table bolts: handoff stage added 2026-11-14 (see STATUS.md for gates). A pick that encloses the whole bolt
   (CoM-centred grasp) cannot be handed off by these grippers; the pick reward does not prefer high grasps.
3. Reach push hack: contact is the reach goal, so a push cannot be scored before the goal unless it happens within
   the 5-action debounce. The user accepted reach with this documented G5 exception (2026-11-14).
