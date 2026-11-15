# Subtask rewards — quickstart for RL

Each long ABC task is split into short **stages** (reach → pick → carry → drop, …). Every stage is its own RL task
with:
- **its own start states**: the scene is already in that stage's starting condition, e.g. the object held for carry;
- **its own per-chunk reward**: +1 when the stage goal is reached, −1 on a failure, and a dense progress signal in
  between.

Every stage marked usable below was checked with a scripted solver, which succeeds, and with scripted reward hacks,
which don't earn +1. It has **not** been checked with real RL training yet, so watch for new exploits.

Videos of the scripted runs, per task and stage: https://huggingface.co/spaces/OneWorld-AI/subtask-reward-visualization

## Usable stages

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

**Not usable.** These stages either don't exist or are registered in `task_specs.py` but failed a check. They have
no saved starts. Details are in each task's `STATUS.md`.

| task | stage | why |
|---|---|---|
| dish rack | reach, pick | **Not built.** No scripted policy can pick a flat plate up off the table: a rim pinch leaves the plate tilted ~35° with its far edge on the table, and it slips out by 16 cm. |
| dish rack | carry, insert | The plate is placed directly in the gripper, since it can't be picked up. Too few scenes have an upright held plate the arm can also bring to a rack slot (65 and 72 of 100 valid starts, 80 needed). |
| dish rack | release | After the gripper opens, the plate stays in the rack on only 8 of 16 runs. |
| mug rack | reach | The approved goal counts a finger touch as reaching, so a push earns +1. |
| mug rack | align, insert, release | The gripper can't hold the mug sideways in the hanging pose without it slipping (66, 36 and 51 of 100 valid starts). |
| inhand | carry | **Real exploit:** the giving arm shoving the receiving arm's gripper pushes the item across for +1 (2 of 20 starts). |
| open / close drawer | pull / push | The reward looks fine (solver 19/20 and 20/20, no hack earns +1), but the hack test, using both arms to open another drawer, could not be carried out (the arms collide). Use these at your own risk. |

**Handoff stages** only cover scenes where the object has to change arms, e.g. the bin is on the other side.

## Get the start states

The saved starts are on Hugging Face, not in git: the dataset
[`OneWorld-AI/abc-subtask-starts`](https://huggingface.co/datasets/OneWorld-AI/abc-subtask-starts).
- It is private: ask for access to the OneWorld-AI org, then `hf auth login`.
- It holds 43 stages and 4,410 starts, about 11 MB.

```bash
hf download OneWorld-AI/abc-subtask-starts --repo-type dataset --local-dir final_starts
# -> final_starts/<task>/<stage>.npz, manifest.json (row counts, abc commit), README.md
```

**A start only works with the scene code that built it.** `state` is a raw MuJoCo state vector, and `seed` rebuilds
the randomized scene it belongs to. `manifest.json` records the abc commit the starts were built at. If a task's
scene XML, randomizer or object set changes, rebuild that task's starts (see Starts below) and upload them again.

### What's in a file

Each `<stage>.npz` holds N starts (N = 80–217), one per row. Example: row 0 of
`put_plastic_bottles_in_bin/carry.npz`:

| key | value | meaning |
|---|---|---|
| `seed` | `920000000` | scene seed: `reset(seed)` rebuilds this exact randomized scene first |
| `stage` | `'carry'` | stage name (one scalar per file; mug flip files don't have it) |
| `target` | `4` | the object this episode is about (0-based: the prompt says "bottle 5"); mug flip calls it `mug` |
| `arm` | `0` | the arm holding it (0 left, 1 right) |
| `state` | float64 × 531 | `mj_getState(mjSTATE_INTEGRATION)`: time, qpos (58), qvel (52), act, …. The whole sim at the stage start, here with bottle 4 held 7 cm above where it rested |
| `rest_height` | 6 × 3 | reference positions from the scene's start: each bottle's resting centre of mass, plus the bin. Lift and "fell / bin moved" checks compare against these |
| `labels` | `[0 0 0 0 2]` | each object's stage label at the start: the target (index 4) is in `carry` (`LABELS = reach, pick, carry, drop, done, fallen`), the others still at `reach` |

- The shapes of `state`, `rest_height` and `labels` vary by task.
- Handoff carry/drop files add `handoff_scene` and `source`, saying which chain built the row.
- Mug flip files add `grasp`, `upz`, `lift`, `variant` and `mugs`.
- `reset_to_stage(seed, row)` uses only `seed`, `state`, `target`, `rest_height` and `labels`.
- Load files with `abc_sim.load_starts`: it gives every file the same row format (fills `stage`, maps mug flip's
  `mug` to `target`, makes `seed`/`target`/`arm` plain ints). Write new ones with `stagekit.save_starts`.

## Minimal loop

```python
import numpy as np, abc_sim

task, stage = "put_plastic_bottles_in_bin", "carry"
env = abc_sim.make_env(task=f"{task}__{stage}", render_cameras=False)   # True for images in obs
starts = abc_sim.load_starts("final_starts", task, stage)                # or set $ABC_SUBTASK_STARTS, pass None
rng = np.random.default_rng(0)

for ep in range(N):
    row = starts.sample(rng)
    obs, info = env.reset_to_stage(row["seed"], row)   # exact restore, ms
    prompt = info["prompt"]                            # stage prompt, e.g. "carry over the bin opening bottle 5"
    for chunk in range(MAX_CHUNKS):
        for _ in range(32):
            obs, _, _, _, _ = env.step(policy(obs))    # 14-D action, see below
        reward, done = env.subtask.chunk_reward()
        if done:                                       # +1 env.subtask.success, or -1 env.subtask.failure
            break
```

`env.subtask` is the current stage's evaluator: `chunk_reward()`, `success`, `failure` (reason string), `last`
(metrics of the latest step: label, progress, task extras), `next_stage()`. Read it through `env.subtask` each time:
a reset can rebuild it.

### Chaining stages in one episode

Stages connect: a stage's goal is the next stage's starting condition. After a +1, `env.advance_stage()` switches
to the stage the target is now in, without a reset (same sim state, same target, labels carried over):

```python
obs, info = env.reset_to_stage(row["seed"], row)       # e.g. a reach start
while True:
    ...                                                # run the policy for this stage until done
    if not env.subtask.success or env.subtask.next_stage() is None:
        break                                          # failed, or the target is done
    info = env.advance_stage()                         # reach -> pick -> (handoff) -> carry -> drop
```

The next stage is the one the scene actually needs (a same-side LEGO block skips handoff). `advance_stage("x")`
refuses a stage the target is not in. With the scripted solvers this chains reach → drop in 9 of 20 bottles
episodes; the misses are solver planning failures. A policy trained only on saved starts may see slightly different
states when chained (e.g. reach can end anywhere within 5 cm of the object), so mix in chained starts if needed.

## Critical points

- **Action:** 14-D, 7 per arm, left then right: 6 absolute joint targets, then the gripper in [0, 1] (1 = open).
  `obs["state"]` holds the robot state. Images are in `obs["images"]` if `render_cameras=True`.
- **The reward ignores env.step's reward.** Use `env.subtask.chunk_reward()`. It returns `(reward, done)` for everything since
  the last call.
- **32 is not fixed.** The evaluator checks the goal and the failures after every physics step. `chunk_reward()`
  can be called at any interval: every 32 actions (our runs, matching the policy's action chunk), every step, etc.
- **Reward modes:** set `env.subtask.reward_mode` after each reset.
  - `"progress"` (default) is potential-based. The in-between rewards sum to 0.5 × progress gained, so going
    slowly or wobbling earns nothing extra.
  - `"sign"` gives ±1 per chunk of progress. `"best"` gives +1 per new best. Both need a discount or a per-chunk
    cost, because slow progress can out-earn finishing.
- **Failures are latched:** dropping the object, knocking the scene over, or numerical blow-ups give −1 and end the
  episode.
- **No time limit is built in.** Set your own horizon. We used 10 chunks of 32 (drawer: 40).
- **Starts:**
  - `reset_to_stage(seed)` without `row` builds a fresh start: it randomizes the scene, scripts the earlier stages,
    checks the result is valid and that it holds still. That takes about 1–10 s of CPU, too slow per episode.
  - Use the saved pools (80–217 starts per stage, all checked), or prebuild your own with
    `env.subtask.build_start(env, seed)`. It returns `(start, log)`, and `start` is `None` when that seed has no
    valid start. Save a pool with `abc_sim.task_eval.stagekit.save_starts(path, rows, stage=...)`.
  - Seeds don't overlap across tasks.
- **Speed:** about 5–15 ms per physics step on one CPU core, with no cameras. Run many envs in parallel processes.
- **Setup:** the `simulation` venv (MuJoCo 3.8.1) with `PYTHONPATH=<abc checkout>`.
  - Headless: `MUJOCO_GL=disable` without cameras, or EGL with them.
  - Set `OPENBLAS_NUM_THREADS=1` when running many processes.

## Paths

| what | where |
|---|---|
| **Saved start pools** (not in git) | HF dataset `OneWorld-AI/abc-subtask-starts` (see "Get the start states"); our local copy: `simulation/outputs/subtask_rewards/final_starts/` |
| reset / restore / chaining | `abc_sim/env.py` → `reset_to_stage`, `subtask`, `advance_stage` |
| start files | `abc_sim/task_eval/stagekit/starts.py` (`load_starts`, `read_starts`, `save_starts`) |
| reward kit (`chunk_reward`, reward modes) | `abc_sim/task_eval/stagekit/subtask.py` (+ `scene.py`, `tracker.py`, `holding.py`) |
| shared handoff stage | `abc_sim/task_eval/handoff_stage.py` |
| per-task stage logic | `abc_sim/task_eval/<short>_subtask.py`, `<short>_stages.py`, `<short>_starts.py` |
| spec names (`<task>__<stage>`) | generated from each evaluator class (`TASK`, `ENV_TASK`, `NAME`, `STAGE_PROMPTS`) by `abc_sim/task_eval/subtasks.py` |
| stage definitions (goals, failures) | `subtask_rewards/<short>/stage_table.md` |
| gate results and known issues | `subtask_rewards/<short>/STATUS.md`, `design.md`; overview in `subtask_rewards/README.md` |
| tests | `tests/test_<short>_subtasks.py` (plain scripts: `python tests/test_bottles_subtasks.py`) |
| scripted solvers and hacks (examples of driving the env) | `simulation/experiments/subtask_rewards/<short>/scripted_rollouts.py`, `build_starts.py`; mug flip: `simulation/experiments/mug_stages/` |
| how the pipeline / gates work | `.claude/skills/abc-subtask-reward/` (`SKILL.md`, `references/gates.md`, `references/contract.md`) |

Branch: `subtask-rewards` on GitHub (`one-world-ai/abc-subtask`).
