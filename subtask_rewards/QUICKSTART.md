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

**Do not use** any other stage registered in `task_specs.py`:
- dish rack, all stages;
- mug rack reach / align / insert / release;
- inhand carry (the giving arm shoving the receiving arm earns +1);
- drawer pull / push.

Each task's `STATUS.md` says why.

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
- `reset_to_stage(seed, row)` uses only `seed`, `state`, `target`/`mug`, `rest_height` and `labels`.

## Minimal loop

```python
import numpy as np, abc_sim

task = "put_plastic_bottles_in_bin__carry"           # "<task>__<stage>"
env = abc_sim.make_env(task=task, render_cameras=False)   # True for images in obs
z = np.load("final_starts/put_plastic_bottles_in_bin/carry.npz", allow_pickle=True)

def start_row(i):                                     # some keys are scalars (e.g. stage)
    return {k: (z[k][i] if z[k].ndim else z[k].item()) for k in z.files}

for ep in range(N):
    row = start_row(np.random.randint(len(z["seed"])))
    obs, info = env.reset_to_stage(int(row["seed"]), row)  # exact restore, ms
    ev = env._task_evaluator                          # fetch AFTER every reset (reset may rebuild it)
    prompt = info["prompt"]                           # stage prompt, e.g. "carry over the bin opening bottle 5"
    for chunk in range(MAX_CHUNKS):
        for _ in range(32):
            obs, _, _, _, _ = env.step(policy(obs))   # 14-D action, see below
        reward, done = ev.chunk_reward()
        if done:                                      # +1 success or -1 failure: episode over
            break
```

## Critical points

- **Action:** 14-D, 7 per arm, left then right: 6 absolute joint targets, then the gripper in [0, 1] (1 = open).
  `obs["state"]` holds the robot state. Images are in `obs["images"]` if `render_cameras=True`.
- **The reward ignores env.step's reward.** Use `ev.chunk_reward()`. It returns `(reward, done)` for everything since
  the last call.
- **32 is not fixed.** The evaluator checks the goal and the failures after every physics step. `chunk_reward()`
  can be called at any interval: every 32 actions (our runs, matching the policy's action chunk), every step, etc.
- **Reward modes:** set `ev.reward_mode` after each reset.
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
    `env._task_evaluator.build_start(env, seed)`. It returns `(start, log)`, and `start` is `None` when that seed
    has no valid start. To save a pool, stack the returned dicts key by key into an npz in the same format.
  - Seeds don't overlap across tasks.
- **Speed:** about 5–15 ms per physics step on one CPU core, with no cameras. Run many envs in parallel processes.
- **Setup:** the `simulation` venv (MuJoCo 3.8.1) with `PYTHONPATH=<abc checkout>`.
  - Headless: `MUJOCO_GL=disable` without cameras, or EGL with them.
  - Set `OPENBLAS_NUM_THREADS=1` when running many processes.

## Paths

| what | where |
|---|---|
| **Saved start pools** (not in git) | HF dataset `OneWorld-AI/abc-subtask-starts` (see "Get the start states"); our local copy: `simulation/outputs/subtask_rewards/final_starts/` |
| reset / restore | `abc_sim/env.py` → `reset_to_stage` |
| reward kit (`chunk_reward`, reward modes) | `abc_sim/task_eval/stagekit/subtask.py` (+ `scene.py`, `tracker.py`, `holding.py`) |
| shared handoff stage | `abc_sim/task_eval/handoff_stage.py` |
| per-task stage logic | `abc_sim/task_eval/<short>_subtask.py`, `<short>_stages.py`, `<short>_starts.py` |
| spec names (`<task>__<stage>`) | `abc_sim/task_specs.py` |
| stage definitions (goals, failures) | `subtask_rewards/<short>/stage_table.md` |
| gate results and known issues | `subtask_rewards/<short>/STATUS.md`, `design.md`; overview in `subtask_rewards/README.md` |
| tests | `tests/test_<short>_subtasks.py` (plain scripts: `python tests/test_bottles_subtasks.py`) |
| scripted solvers and hacks (examples of driving the env) | `simulation/experiments/subtask_rewards/<short>/scripted_rollouts.py`, `build_starts.py`; mug flip: `simulation/experiments/mug_stages/` |
| how the pipeline / gates work | `.claude/skills/abc-subtask-reward/` (`SKILL.md`, `references/gates.md`, `references/contract.md`) |

Branch: `subtask-rewards` on GitHub (`one-world-ai/abc-subtask`).
