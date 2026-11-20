# Held-object ABC pilot

These two candidate environments run on the CPU MuJoCo ABC simulator. The policy evaluation is pending. The new reward thresholds and the seed split are pilot choices. Use the split column in each NPZ before sampling.

| Environment | Prompt | Accepted start | Terminal success |
|---|---|---|---|
| `pouring__pour` | Tilt the held mug and pour the beads into the receiving cup. Keep the beads inside. | Source mug lifted ≥3 cm with a two-finger grasp, all beads inside it, zero beads in receiver. | Tilt the held mug past cosine 0.75. At least 9 of the 10 beads must be inside the upright receiver and moving under 0.1 m/s continuously as a count for 0.5 s, with the source mug held. |
| `throw_plastic_bottles_in_bin__throw` | Throw the held bottle from outside the bin opening so it lands and stays inside. | Target bottle lifted ≥3 cm with a two-finger grasp, center outside the opening. | At an observed step the bottle has no contacts outside the opening, then its center crosses the rim downwards inside the opening. It must rest in contact with the bin, with no finger contact, below 0.05 m/s and 0.5 rad/s for 0.5 s. |

Both pay 0 while running, +1 on success and −1 on failure or timeout. Read the terminal reward once and reset. Pouring allows 18 simulated seconds (530 control steps at 34 ms); throwing allows 12 seconds (353 steps). An observed free goal-object velocity over 4 m/s or 60 rad/s latches `sim_fault` at the physics step. Discard that episode from training. The throwing rule checks flight and rim crossing at control sample times. A brief unsupported flight completely between samples can be missed. Pouring uses a 0.12 s grace interval for finger contact flicker. Retention is measured over the stated 0.5 s window.

## Starts

The donor is the leading sample of ABC sim_224 train-split demonstrations in release archive order. The harvester reconstructs the recorded qpos in the seed-generated simulator, applies a stationary grip under gravity, validates the physical preconditions, and saves the resulting complete MuJoCo integration state. Each accepted row records the source episode ID and frame. These are newly settled states. The three largest distinct seed IDs per task are reserved for evaluation within this pilot. This split cannot establish independence from the training history of an external checkpoint.

| Environment | Train seeds | Eval seeds |
|---|---|---|
| Pour | 0, 1261875904, 1261908051, 1261932817, 1261967784, 1261991168, 1262011448, 1262036268, 1262055823, 1262100751 | 1262132038, 1262160506, 1262186638 |
| Throw | 3632411537, 3632798043 (two starts) | 3632835885, 3632931531, 3633048973 |

```python
import abc_sim
from abc_sim.task_eval.stagekit import load_starts

task, stage = "throw_plastic_bottles_in_bin", "throw"
rows = load_starts("subtask_rewards/held_starts", task, stage)
row = next(row for row in rows if row["split"] == "eval")
env = abc_sim.make_env(task=f"{task}__{stage}", render_cameras=False)
obs, info = env.reset_to_stage(row["seed"], row)
# Run the policy with info["prompt"] and env.step(action).
# For a chunked policy, stop at the first terminal step.
env.close()
```

## Reproduction and interpretation

Baseline `one-world-ai/abc-subtask@e39835d`, branch `subtask-rewards`, MuJoCo 3.8.1. This revision has automatic stage registration. Install assets with `python prepare.py --sim-task pour bottles mug_flip put_bottles`. Run the following from the repo root with extracted episode directories as arguments.

```sh
python -m scripts.replay_abc_goal EPISODE...
python -m scripts.harvest_held_start --output /tmp/held_starts EPISODE...
python -m scripts.check_held_stages /tmp/held_starts --episodes EPISODE... --report /tmp/checks.json
python -m pytest -q tests/test_held_stages.py tests/test_mug_flip_subtasks.py tests/test_subtask_api.py
```

[Episode release manifest](https://abc-data.timehorizons.org/dataset/sim_224/manifest.json) · [Per-episode verdicts and check counts](checks.json)

The original full-task evaluator accepted the final frame of 12/13 pouring and 6/6 throwing demonstrations. Each task had one episode that crossed the original success threshold and subsequently fell below it. The throw full task counts all bottles; the new subtask scores only its held target.

For each new task, `trace` scores donor qpos with velocities estimated by finite differences and contacts reconstructed on the seed model. `action_replay` executes recorded donor commands from the settled start in physics. Their verdicts can differ. The idle, open and sweep scripts cover the saved starts. The sweep opens the holding gripper and sweeps an arm joint; a drop may end the episode before a push occurs. It does not establish coverage of arbitrary pushes. Separate tests exercise transient bead membership and releasing a bottle inside the opening without prior outside flight.

In a further 30-control-step idle check, all 13/13 source mugs and 6/6 bottles remained grasped. Across the free objects the maximum translation was 6.5 mm for pouring and 8.3 mm for throwing. Maximum observed contact penetration was 0.40 mm and 0.43 mm respectively.

The [geometry audit](geometry.json) samples 2,048 points per separate visible mesh assembly on each start. Source and receiving cup coverage was 100% with a 2 mm hull-plane tolerance on all 13 starts. For beads, bottles and bin, the visible primitive collision geoms are shared. No goal geom was configured to collide exclusively with the fingers by bit mask. A motion-and-grip scan flagged two pouring clips (4 frames each) and one throwing clip (5 frames). The gripper views show objects adjacent to fingers but cannot resolve pad forces or prove a pinch. Treat the flagged donor segments as unresolved; see their episode IDs and frames in the audit.

Both candidates remain excluded from the training-ready list until the current policy is evaluated on the reserved worlds. Policy trials n=0 for each, so a Wilson interval is undefined. The current checkpoint path, inference entry point and rollout measurements are pending. Exact pad contact and release instants in 30 Hz recorded traces can remain ambiguous. Existing mug-flip tests passed 9/9, including exact restore, stationary holds, drops, rotation reward and 3/3 scripted place episodes.


## Final measurements

| Check | Pour | Throw |
|---|---:|---:|
| Start states accepted / attempted | 13/13 | 6/6 |
| Exact restore attempts | 52/52 | 24/24 |
| Idle / open / sweep successes | 0/13 each | 0/6 each |
| Success in first five control steps of those tests | 0/39 | 0/18 |
| Injected velocity faults latched | 13/13 | 6/6 |
| Recorded poses accepted under the new subtask rule | 12/13 | 5/6 |
| Recorded actions successfully executed from the new start | 13/13 | 6/6 |
| Current-policy held-out trials | 0 | 0 |

The three targeted test files passed 18/18. The scripted sweep has limited coverage as described above. The reserved seeds are held out from pilot training, and were used for environment QA.

## Failure review and changes made during verification

- Pour seed 1261932817 briefly lost one pad at donor-action frame 480 and regained it one frame later. Seed 1261908051 regained it after two frames. Immediate release failure caused false negatives. Physical time debounce resolved this check fault.
- With the initial 12 s budget, seed 1261932817 timed out at donor frame 584. Original scoring first reaches the bead threshold at frame 696, 15.7 s after the selected start. The wide and wrist views show an ongoing transfer. This exposed a budget mismatch. The pouring budget was increased to 18 s.
- Pour seed 1261991168 triggered the recorded-trace speed guard at frame 371. The estimated angular speed of bead 6 was 61.98 rad/s. The donor ended with only 7 beads inside. Exclude the flagged reconstructed episode; the physical action replay is a separate run.
- Throw seed 3632411537 first lost finger contact without an observed unsupported outside frame (donor frame 255). The wrist views show the gripper by the rim. The physical replay from the settled state passed. Keep the trace rejection and flag the exact release path as unresolved at 30 Hz.

## Files in this coding round

| File | Change and reason |
|---|---|
| `abc_sim/env.py` | Invoke the held-stage velocity guard at physics frequency and end Gym episodes on its terminal result. |
| `abc_sim/task_eval/stagekit/held_stage.py` | Share held-start validation, seed lookup, time budget, fault latching and sparse reward. |
| `abc_sim/task_eval/pour_subtask.py` | Register the pour stage and check tilt, grasp and retained beads. |
| `abc_sim/task_eval/throw_subtask.py` | Register the throw stage and check outside release, rim crossing and rest in the bin. |
| `scripts/replay_abc_goal.py` | Replay donor poses against original ABC completion rules. |
| `scripts/harvest_held_start.py` | Reconstruct, settle, validate, split and save held starts. |
| `scripts/check_held_stages.py` | Exercise counterexamples and fault latches, then replay donor poses and commands separately. |
| `scripts/audit_held_geometry.py` | Measure visible mesh coverage and flag possible unsupported carries. |
| `tests/test_held_stages.py` | Test transient bead membership, release without outside flight and deterministic seed lookup. |
| `tests/test_subtask_api.py` | Add both environments to the expected registration inventory. |
| `subtask_rewards/held_starts/pouring/pour.npz` | Store all 13 accepted pouring starts and provenance. |
| `subtask_rewards/held_starts/throw_plastic_bottles_in_bin/throw.npz` | Store all 6 accepted throwing starts and provenance. |
| `subtask_rewards/held_starts/checks.json` | Record check counts and the two replay verdicts per episode. |
| `subtask_rewards/held_starts/geometry.json` | Record asset measurements and candidate sticky-carry frame ranges per start. |
| `subtask_rewards/held_starts/README.md` | Document the reward, budget, split, results, exclusions, and reproduction commands. |
