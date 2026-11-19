#!/bin/bash
#SBATCH --job-name=r2g_render
#SBATCH --account=torch_pr_56_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Videos of one task's scripted rollouts for the Space (CPU rendering):
#   sbatch --array=0-N slurm/torch/r2g_render.bash STARTS ROLLOUTS OUT task1 task2 ...
set -euo pipefail
P=/projects/work/yang-lab/projects/policy-finetuning
STARTS=$1; ROLL=$2; OUT=$3; shift 3
TASKS=("$@"); T=${TASKS[$SLURM_ARRAY_TASK_ID]}
cd $P/simulation
source experiments/subtask_reward_space/render_env.sh
export ABC_REPO=$P/abc-r2g PYTHONPATH=$P/abc-r2g ABC_STAGE_STARTS=$P/abc_rl_data/abc-subtask-starts LP_NUM_THREADS=2
.venv/bin/python experiments/subtask_reward_space/render_scripted.py --task "r2g_$T" --env-sep __ \
    --starts "$STARTS/r2g_$T" --scripted "$ROLL/$T" --per-behaviour 2 --workers 8 \
    --out "$OUT/r2g_$T"
