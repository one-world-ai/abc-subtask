# Simulation evaluation for abc_dit_xl_200k_model.pt

To run all scored tasks, use `bash slurm/submit_foundation_all_tasks.bash`.
It reads `slurm/foundation_tasks.txt` and submits **29 separate Slurm jobs**,
one per canonical task, including put-bottles, all five counting specs and all
eight spelling specs. 

The generic worker is `eval_foundation_task.bash`. It keeps official RTC, reset and prompt behavior. LEGO receives 400 chunks per world; other tasks receive 236. Four hours is the requested limit, not a guarantee that every task will finish.

Batch outputs are under
`outputs/foundation_200k/all_tasks/run-<UTC timestamp>/<task>/job-<id>/`.
The batch's `jobs.tsv` maps each task to its submitted job ID and output path;
`tasks.txt` records the submitted task list. Logs remain under
`../policy-finetuning-logs/abc-foundation/abc200k_<task>-<id>.out/.err`.

To submit just one other task:

```bash
TASK=turn_mug_right_side_up sbatch slurm/eval_foundation_task.bash
```

Before a fresh all-task setup, install each listed task's required assets:

```bash
mapfile -t TASKS < slurm/foundation_tasks.txt
.venv-sim-eval/bin/python prepare.py --sim-task "${TASKS[@]}"
```

This includes the additional RoboCasa object packs used by in-hand transfer.

`test_eval_foundation_put_bottles.bash` evaluates the downloaded ABC-DiT 200k
foundation model on the current `put_plastic_bottles_in_bin` catalogue task.
Defaults: 100 worlds, one L40S, sequential CPU MuJoCo physics with MJWarp
rendering, RTC with a four-action prefix, 10 diffusion steps, 15 executed
actions/chunk, and 236 chunks/world. 

The new environment is `.venv-sim-eval`; existing `.venv` and OpenPI environments
are preserved. It has MuJoCo 3.8.1, MuJoCo-Warp 3.8.0.3, Warp 1.13.0 and Torch
2.11.0+cu128. `sim-eval-requirements.freeze.txt` records all installed versions.
CLIP weights and vocabulary were already present under `cache/clip/`.

Run from the ABC checkout:

```bash
cd /projects/work/yang-lab/projects/policy-finetuning/abc

# Full evaluation, after checking the smoke result.
sbatch slurm/eval_foundation_put_bottles.bash

# Optional: disable videos.
SAVE_VIDEO=0 sbatch slurm/eval_foundation_put_bottles.bash
```

Default logs:
`../policy-finetuning-logs/abc-foundation/<job-name>-<job-id>.out/.err`.
That directory has been created. Each job writes under
`outputs/foundation_200k/put_plastic_bottles_in_bin/job-<job-id>/`:

- `launch.json`: exact evaluation arguments
- `evaluator.patch`: local evaluator/configuration changes relative to git
- `summary.json`: final aggregate and per-world results (written when all
  worlds finish; interrupted runs are not resumable)
- `world_*.mp4`: when `SAVE_VIDEO=1`, frames sampled every 15 actions at 2 FPS

`SAVE_VIDEO=1` is the default and saves every world's video. Videos at
2 FPS approximate simulation time; use them to inspect gross behavior.


The launcher uses the official evaluator unchanged: robot arm pose carries
between worlds and policy noise uses one continuous seeded random stream.
No independent-reset or per-world-noise modifications are enabled. This is
an ordered 100-world evaluation matching the upstream execution protocol.

Other overrides: `NUM_WORLDS`, `NUM_CHUNKS`, `SCENE_SEED`, `POLICY_SEED`,
`RTC`, `FAST_INFERENCE`, `CHECKPOINT`, `EVAL_PYTHON`, `PYTHON_INCLUDE_DIR`, `OUTPUT_DIR`.
`RTC=0` selects synchronous inference with prefix length zero. Numeric seeds
must be nonnegative; switches are 0/1. An existing completed/launched output
directory is rejected to prevent overwriting results.

