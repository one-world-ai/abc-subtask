# Bottle subtasks for single-stage RL

Task: `put_plastic_bottles_in_bin`, using the randomized tall-tub **put** scene. The 41/60 throw-scene baseline
does not apply. User approved reach → pick → carry → drop, with release above the opening allowed. Lowering
before release is also allowed; a bottle must finish released, supported inside the bin, and at rest.
Gate evidence and any limitations are recorded in `STATUS.md`.

## Use

```python
import abc_sim
env = abc_sim.make_env(task="put_plastic_bottles_in_bin__carry", render_cameras=False)
obs, info = env.reset_to_stage(seed)  # builds a start, or pass a saved start row as the second argument
prompt = info["prompt"]
while True:
    for action in policy(obs, prompt)[:32]:
        obs, _, _, _, info = env.step(action)
    reward, done = env._task_evaluator.chunk_reward()
    if done:
        break
```

The four suffixes are `__reach`, `__pick`, `__carry`, `__drop`. Use the CPU, single-world environment: contacts
and real linear/angular velocities are required. Direct qpos-only scoring raises an explicit error rather than
inventing zero velocities. Batched MJWarp training is not validated by this work. The stock whole-task bottle
scorer remains unchanged.

## Stages and goals

One target per episode: the nearest not-yet-binned bottle at the original reset. The target stays fixed, including
when a fallback start uses the arm on the bin's side. Geometry is measured per randomized model in the live bin
frame; bin local y is its opening direction.

| stage | precondition | goal | progress / span |
|---|---|---|---|
| reach | original scene, target away from fingers | grasp site within 5 cm of target CoM, or finger contact | −distance / 0.30 m |
| pick | gripper near resting target | two fingers of one hand on target, lifted ≥3 cm | 0.25·finger count +0.5·lift fraction / 1.0 |
| carry | target held, lifted, outside opening | target held above opening; CoM within top radius −2 cm | −3-D distance to above-opening region / 0.40 m |
| drop | held/touched over opening, or lowered into bin | stock per-target in-bin condition, no finger contact, speed ≤2 cm/s, angular speed ≤0.2 rad/s, and support connected to bin | −radius past opening +0.05 once released / 0.10 m |

Labels debounce for five executed actions. Near enter/exit is 5/7 cm; opening exit hysteresis is 2 cm. A released
airborne bottle retains its previous label until it lands or another physical failure resolves it. A zero-speed
apex does not count as landing. A held bottle lowered below the rim remains in drop, and may be released there.

Support is a contact graph rooted at the bin fixture. Support can pass through other released, stationary,
stock-in-bin bottles. A pair of airborne bottles touching each other is not a support root. Neither is a bottle
held by the other hand. This avoids using the stock scorer's loose spatial test as proof of rest.

## Reward and failures

Call `chunk_reward()` after each 32-action chunk. Goal returns +1 and ends the episode; a latched failure returns
−1 and ends it. Otherwise reward is `0.5 * (progress_now - progress_previous_chunk) / span`.

For example, carrying 12 cm closer to the opening with span 40 cm earns 0.15 total shaping, independent of how many
chunks cover that movement. Releasing above the opening adds 0.25 shaping if the episode remains nonterminal;
settling inside then gives terminal +1. Holding still earns approximately zero. Terminal reward replaces the
last chunk's shaping.

Failures:

- Any stage: bin translation >5 cm XY from the saved anchor, opening tilt >20°, a previously binned bottle knocked
  out, target off table, or regression to an earlier stage.
- Reach/pick: target pushed >5 cm while ungrasped, or pushed straight to the drop/done region without a grasp.
- Carry/drop: every touched→untouched transition must be above the approved opening or already below its rim
  inside the bin. A second release after regrasp is checked again. Landing outside fails. High release above the
  opening is allowed.

The policy controls its own time budget; there is no extra time-limit reward. Use the default `progress` reward
mode. The inherited `sign`/`best` modes remain optional and are not the validated training default.

## Starts and physical solvers

Seed base is 920,000,000; candidate i uses base+i, i=0…99. Start snapshots preserve full MuJoCo integration state,
target, original bottle rest positions, original bin position, stage labels, and holding arm.

- Reach: unchanged randomized reset scene.
- Pick: open top-down gripper around the resting bottle, aligned perpendicular to its long inertia axis. Try
  symmetric yaw directions because equivalent parallel-jaw grasps have different wrist feasibility.
- Carry: sampled grasp, 6–12 cm lift, close with gravity compensation, then verify two-finger hold under gravity.
- Drop: first chain a physical carry with positive/negative leaning approaches. When that cannot plan, use the
  skill's sampled-grasp fallback over the bin, verify under gravity, and record the source and failed chain.

Only starts surviving ten idle chunks without terminal reward, regression, or |total shaping|≥0.05 are retained.
Raw built, retained, idle retention, construction time, and failures are all reported, rather than hiding failed
candidate seeds. Drop starts from the sampled fallback validate that individual stage; they do not establish that
every carry start can chain into drop. Carry solver coverage is reported separately.

Scripted policies run through the same env stepping and 32-action scoring interface. Traces contain actions,
chunk rewards, terminal causes, and planning failures. Planning failures count against solver success over all
starts. Safety policies include idle, moving away, close-only, ungrasped pushing, wrong-way carry, early release,
shaking, and release beside the bin. At least 20 starts per stage are used for solver/reward-order checks.

## Files

| path | role |
|---|---|
| `abc_sim/task_eval/bottles_stages.py` | measured geometry, contact support graph, stage labels and potentials |
| `abc_sim/task_eval/bottles_subtask.py` | per-target failures, release checking, CPU scoring contract |
| `abc_sim/task_eval/bottles_starts.py` | grasp generation, exact snapshots, leaning carry, fallback drop starts |
| `abc_sim/task_specs.py`, `abc_sim/task_eval/registry.py` | four task registrations |
| `tests/test_bottles_subtasks.py` | failures, unsupported goals, lowering, release rechecks, shaping, idle and restore |
| `../simulation/experiments/subtask_rewards/bottles/` | reproducible start-set and scripted-policy runners |
| `../simulation/outputs/subtask_rewards/bottles/` | final gate summaries, per-seed diagnostics, action/reward traces |

Shared `stagekit` code is unchanged. No RL training or updated policy success rate is claimed by these scripted
checks. Stage prompting has no frame history at a saved mid-task start, matching the existing pilot convention.
