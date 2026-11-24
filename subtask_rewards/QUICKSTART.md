# Subtask rewards — quickstart for RL

Each long manipulation task is split into short **stages** (reach → pick → carry → place, …). Every stage is its own
RL env with:
- **its own start states**: the scene is already in that stage's starting condition, e.g. the object held for carry;
- **its own per-chunk reward**: +1 when the stage goal is reached, −1 on a failure, and a dense progress signal in
  between.

Two families of tasks run in this one repo, on the same simulator (MuJoCo, bimanual YAM) and with the same API:

| family | what it is | env id | example |
|---|---|---|---|
| **ABC** | the ABC benchmark tasks | `<task>__<stage>` | `put_plastic_bottles_in_bin__carry` |
| **Real2Gym** | Yuyang's real-to-sim scenes (`OneWorld-AI/ABC_simulators`), ported onto the ABC table | `r2g_<task>__<stage>` | `r2g_put_the_screwdriver_in_the_bin__carry` |

Every stage marked usable below was checked with a scripted solver, which succeeds, and with scripted reward hacks,
which don't earn +1. It has **not** been checked with real RL training yet, so watch for new exploits.

Videos of the scripted runs, per task and stage: https://huggingface.co/spaces/OneWorld-AI/subtask-reward-visualization

To add a new task or a new simulator's tasks: [ADDING_A_TASK.md](ADDING_A_TASK.md).

## Setup

```bash
git clone git@github.com:one-world-ai/abc-subtask.git && cd abc-subtask && git switch subtask-r
uv sync                                                    # or any venv with this repo's pyproject (MuJoCo 3.8)
python prepare.py --sim-task <abc task>                    # ABC 3D models for that task (once per task)

# Real2Gym 3D models (private: OneWorld-AI org access, then `hf auth login`)
hf download OneWorld-AI/abc-rl-envs --repo-type dataset --include "abc_assets_v4/*" --local-dir ../abc-rl-envs
ln -s "$(realpath ../abc-rl-envs/abc_assets_v4)" abc_sim/models/assets_real2gym

# Optional: reproduce a historical release that used saved rows (private)
hf download OneWorld-AI/abc-subtask-starts --repo-type dataset --local-dir final_starts
```

Run with `PYTHONPATH=<repo>`. Headless: `MUJOCO_GL=disable` without cameras, EGL with them. Set
`OPENBLAS_NUM_THREADS=1` when running many processes.

## Fresh resets for newly authored stages

```python
import abc_sim

task, stage = "put_plastic_bottles_in_bin", "carry"        # or "r2g_put_the_screwdriver_in_the_bin", "carry"
env = abc_sim.make_env(task=f"{task}__{stage}", render_cameras=False)   # True for images in obs
for seed in training_seeds:
    obs, info = env.reset_to_stage(seed)               # generate the stage from this seed
    prompt = info["prompt"]                            # e.g. "carry over the bin opening bottle 5"
    for chunk in range(MAX_CHUNKS):
        for _ in range(32):
            obs, _, _, _, _ = env.step(policy(obs))    # 14-D action, see below
        reward, done = env.subtask.chunk_reward()
        if done:                                       # +1 env.subtask.success, or -1 env.subtask.failure
            break
```

Use disjoint training and evaluation seeds. A builder can exhaust its bounded attempts. Record the rejected seed
and reason and count it in start-quality reports. Each new stage must also be tested through its training worker's
fresh-reset path. Resuming a previous training trajectory uses the worker's explicit restore path.

The stage API:

| call | what it does |
|---|---|
| `abc_sim.make_env(task="<task>__<stage>")` | the stage env |
| `abc_sim.load_starts(dir, task, stage)` | the saved starts of that stage; `.sample(rng)` gives one row |
| `env.reset_to_stage(seed, row)` | start an episode from that row (without `row`: build a start now, 1–10 s) |
| `env.subtask.chunk_reward()` | `(reward, done)` for everything since the last call |
| `env.advance_stage()` | after a +1, continue in the next stage without a reset |

`env.subtask` is the current stage's evaluator: `chunk_reward()`, `success`, `failure` (reason string), `last`
(metrics of the latest step: label, progress, task extras), `next_stage()`. Read it through `env.subtask` each time:
a reset can rebuild it.

### Chaining stages in one episode

A stage's goal is the next stage's starting condition. After a +1, `env.advance_stage()` switches to the stage the
target is now in, without a reset (same sim state, same target, labels carried over):

```python
obs, info = env.reset_to_stage(seed)                  # e.g. a reach start
while True:
    ...                                                # run the policy for this stage until done
    if not env.subtask.success or env.subtask.next_stage() is None:
        break                                          # failed, or the target is done
    info = env.advance_stage()                         # reach -> pick -> (handoff) -> carry -> drop
```

The next stage is the one the scene actually needs (a same-side LEGO block skips handoff). `advance_stage("x")`
refuses a stage the target is not in. A policy trained only on saved starts may see slightly different states when
chained (e.g. reach can end anywhere within 5 cm of the object), so mix in chained starts if needed.

## Usable stages

### ABC (11 tasks, 43 stages)

| task (`<task>`) | short | stages |
|---|---|---|
| `turn_mug_right_side_up` | mug_flip | reach, pick, rotate, place |
| `put_plastic_bottles_in_bin` | bottles | reach, pick, carry, drop |
| `sort_lego_blocks` | lego | reach, pick, handoff, carry, drop |
| `grab_specific_object_from_clutter` | clutter | reach, pick, handoff, carry, place |
| `put_relative` | put_relative | reach, pick, handoff, carry, place |
| `sort_nuts_and_bolts` | nuts_bolts | reach¹, pick, handoff, carry, drop |
| `sweep_away_paper_scraps_from_table` | sweep | reach, pick, carry, drop |
| `inhand_transfer_item_to_other_side` | inhand | reach, pick, present, handoff, place |
| `open_drawer` | drawer | reach, grasp, release |
| `close_drawer` | drawer | reach, release |
| `hang_mug_on_mug_rack` | mug_rack | pick |

¹ Accepted as an exception: the push-hack test couldn't be carried out properly, but no hack earned +1.

**Handoff stages** only cover scenes where the object has to change arms, e.g. the bin is on the other side.

Not usable (registered, but failed a check; no saved starts; details in each task's `STATUS.md`):

| task | stage | why |
|---|---|---|
| dish rack | reach, pick | **Not built.** No scripted policy can pick a flat plate up off the table: a rim pinch leaves the plate tilted ~35° with its far edge on the table, and it slips out by 16 cm. |
| dish rack | carry, insert | The plate is placed directly in the gripper, since it can't be picked up. Too few scenes have an upright held plate the arm can also bring to a rack slot (65 and 72 of 100 valid starts, 80 needed). |
| dish rack | release | After the gripper opens, the plate stays in the rack on only 8 of 16 runs. |
| mug rack | reach | The approved goal counts a finger touch as reaching, so a push earns +1. |
| mug rack | align, insert, release | The gripper can't hold the mug sideways in the hanging pose without it slipping (66, 36 and 51 of 100 valid starts). |
| inhand | carry | **Real exploit:** the giving arm shoving the receiving arm's gripper pushes the item across for +1 (2 of 20 starts). |
| open / close drawer | pull / push | The reward looks fine (solver 19/20 and 20/20, no hack earns +1), but the hack test, using both arms to open another drawer, could not be carried out (the arms collide). Use these at your own risk. |

### Real2Gym (10 tasks, 33 of 40 stages usable)

All ten are "put the objects in / on a container" tasks and share one stage design: reach → pick → carry → place.
Place is passed when the task's own check says the object is in place. Env id: `r2g_<task>__<stage>`. Details:
`subtask_rewards/r2g/STATUS.md`, numbers: `subtask_rewards/r2g/validation.json`.

| task (`r2g_<task>`) | reach | pick | carry | place |
|---|---|---|---|---|
| `put_the_screwdriver_in_the_bin` | ✓ | ✓ | ✓ | ✓ |
| `place_and_organize_the_fake_fruits_in_the_fruit_bowl` | ✓ | ✓ | ✓ | ✓ |
| `organize_the_makeup` | ✓ | ✓² | ✓ | ✓ |
| `organize_the_condiment_bottles` | ✓ | ✓² | ✓ | ✓ |
| `sort_the_hair_cutting_tools` | ✓ | ✓² | ✓ | ✓ |
| `sort_the_stationery_into_containers` | ✓ | ✓ | ✓ | ✓³ |
| `sort_the_tools_into_containers` | ✓² | ✗ solver 9/20 | ✓³ | ✗ 78 starts |
| `place_and_organize_the_beverage_onto_the_shelf` | ✓ | ✓² | ✓ | ✗ 61 starts, solver 10/20 |
| `place_and_organize_the_canned_foods_onto_the_counter` | ✓ | ✓² | ✓ | ✗ 52 starts, solver 13/20 |
| `put_the_wine_bottle_on_the_wine_rack` | ✗ solver 5/20 | ✗ 76 starts | ✓ | ✗ solver 1/20 |

² Accepted (2026-11-17): every hack behaved as expected, but the other-arm hack could be planned on fewer than 10 of
20 starts (the other arm can't reach the object). ³ Accepted: fails only the 10 s CPU limit per built start.

## Saved starts of historical releases

The saved starts are on Hugging Face, not in git: the dataset
[`OneWorld-AI/abc-subtask-starts`](https://huggingface.co/datasets/OneWorld-AI/abc-subtask-starts) (private: ask for
access to the OneWorld-AI org, then `hf auth login`). Layout: `<task>/<stage>.npz` (Real2Gym tasks as
`r2g_<task>/<stage>.npz`), plus `manifest.json` with row counts and the repo commit the starts were built at.

**What a start is.** A saved snapshot of the whole simulation (every object and joint position and velocity) at the
moment the stage begins, e.g. bottle 5 already held 7 cm above where it rested, plus the scene seed and the target
object.

**A start only works with the scene code that built it.** The snapshot is a raw list of numbers whose order and
length come from the scene. If someone changes a task's scene (adds an object, swaps a model, changes the
randomizer), the old numbers no longer fit: restoring fails with "state has shape X, this scene expects Y", or worse,
restores into the wrong poses. After any scene change, rebuild that task's starts and upload them again.

**Saved vs built starts.** `env.reset_to_stage(seed, row)` restores a saved start in milliseconds.
`env.reset_to_stage(seed)` without a row builds one on the spot: it randomizes the scene for that seed, moves the arm
to the object, closes the gripper and checks that the start is valid and holds still. That takes about 1–10 s of CPU,
which motivated these historical releases to prebuild rows with `env.subtask.build_start(env, seed)`
(returns `(start, log)`, `start` is `None` when that seed has no valid start) and save them with
`abc_sim.task_eval.stagekit.save_starts(path, rows, stage=...)`.

### What's in a file

Each `<stage>.npz` holds N starts (N = 52–217), one per row. Example: row 0 of
`put_plastic_bottles_in_bin/carry.npz`:

| key | value | meaning |
|---|---|---|
| `seed` | `920000000` | scene seed: `reset(seed)` rebuilds this exact randomized scene first |
| `stage` | `'carry'` | stage name (one scalar per file; mug flip files don't have it) |
| `target` | `4` | the object this episode is about (0-based: the prompt says "bottle 5"); mug flip calls it `mug` |
| `arm` | `0` | the arm holding it (0 left, 1 right) |
| `state` | float64 × 531 | `mj_getState(mjSTATE_INTEGRATION)`: time, qpos (58), qvel (52), act, …. The whole sim at the stage start |
| `rest_height` | 6 × 3 | reference positions from the scene's start: each bottle's resting centre of mass, plus the bin. Lift and "fell / bin moved" checks compare against these |
| `labels` | `[0 0 0 0 2]` | each object's stage label at the start: the target (index 4) is in `carry` (`LABELS = reach, pick, carry, drop, done, fallen`), the others still at `reach` |

- The shapes of `state`, `rest_height` and `labels` vary by task.
- Handoff carry/drop files add `handoff_scene` and `source`; mug flip files add `grasp`, `upz`, `lift`, `variant`, `mugs`;
  Real2Gym files add `object` (the target's name).
- `reset_to_stage(seed, row)` uses only `seed`, `state`, `target`, `rest_height` and `labels`.
- Load files with `abc_sim.load_starts`: it gives every file the same row format.

## Critical points

- **Action:** 14-D, 7 per arm, left then right: 6 absolute joint targets, then the gripper in [0, 1] (1 = open).
  `obs["state"]` holds the robot state. Images are in `obs["images"]` if `render_cameras=True`.
- **The reward ignores env.step's reward.** Use `env.subtask.chunk_reward()`.
- **32 is not fixed.** The evaluator checks the goal and the failures after every physics step. `chunk_reward()`
  can be called at any interval: every 32 actions (our runs, matching the policy's action chunk), every step, etc.
- **Reward modes:** set `env.subtask.reward_mode` after each reset.
  - `"progress"` (default) is potential-based. The in-between rewards sum to 0.5 × progress gained, so going
    slowly or wobbling earns nothing extra.
  - `"sign"` gives ±1 per chunk of progress. `"best"` gives +1 per new best. Both need a discount or a per-chunk
    cost, because slow progress can out-earn finishing.
- **Failures are latched:** dropping the object, knocking the scene over, or numerical blow-ups give −1 and end the
  episode.
- **No time limit is built in.** Set your own horizon. We used 10 chunks of 32 (drawer: 40, Real2Gym: 12).
- **Real2Gym `env.reset(seed)`** always starts the arms at the home pose (`home_arms_on_reset`), so a reset depends
  on the seed alone. `reset_to_stage` restores the arms from the start row for both families.
- **Seeds don't overlap across tasks.**
- **Speed:** about 5–15 ms per physics step on one CPU core, with no cameras. Run many envs in parallel processes.

## Paths

| what | where |
|---|---|
| **saved start pools** (not in git) | HF dataset `OneWorld-AI/abc-subtask-starts` |
| **Real2Gym 3D models** (not in git) | HF dataset `OneWorld-AI/abc-rl-envs`, folder `abc_assets_v4` → `abc_sim/models/assets_real2gym` |
| reset / restore / chaining | `abc_sim/env.py` → `reset_to_stage`, `subtask`, `advance_stage` |
| start files | `abc_sim/task_eval/stagekit/starts.py` (`load_starts`, `read_starts`, `save_starts`) |
| reward kit (`chunk_reward`, reward modes) | `abc_sim/task_eval/stagekit/subtask.py` (+ `scene.py`, `tracker.py`, `holding.py`) |
| shared handoff stage | `abc_sim/task_eval/handoff_stage.py` |
| per-task stage logic (ABC) | `abc_sim/task_eval/<short>_subtask.py`, `<short>_stages.py`, `<short>_starts.py` |
| stage logic (Real2Gym, all tasks) | `abc_sim/task_eval/r2g_subtask.py` (task list `R2G_STAGE_TASKS`), `r2g_stages.py`, `r2g_starts.py` |
| Real2Gym scenes and checks | `abc_sim/real2gym_catalog.py`, `abc_sim/randomization/tasks/real2gym_tasks.py`, `abc_sim/task_eval/real2gym_tasks.py` |
| env ids (`<task>__<stage>`) | generated from each evaluator class (`TASK`, `ENV_TASK`, `NAME`, `STAGE_PROMPTS`) by `abc_sim/task_eval/subtasks.py` |
| stage definitions (goals, failures) | `subtask_rewards/<short>/stage_table.md` |
| gate results and known issues | `subtask_rewards/<short>/STATUS.md`, `design.md`; overview in `subtask_rewards/README.md` |
| tests | `tests/test_<short>_subtasks.py`, `tests/test_r2g_subtask.py`, `tests/test_subtask_api.py` (plain scripts) |
| scripted solvers and hacks | Real2Gym: `subtask_rewards/r2g/tools/`; ABC: `subtask_rewards/<short>/tools/` where present, else the `simulation` repo (`experiments/subtask_rewards/<short>/`) |
| how to add a task, the pipeline and the agent skills | [ADDING_A_TASK.md](ADDING_A_TASK.md), `.claude/skills/` |

Branch: `subtask-r` on GitHub (`one-world-ai/abc-subtask`).
