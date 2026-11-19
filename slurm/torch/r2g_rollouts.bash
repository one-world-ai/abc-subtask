#!/bin/bash
#SBATCH --job-name=r2g_rollouts
#SBATCH --account=torch_pr_56_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Scripted solver / hack rollouts of one (task, stage): sbatch --array=0-N slurm/torch/r2g_rollouts.bash STARTS_ROOT OUT N task1 task2 ...
set -euo pipefail
P=/projects/work/yang-lab/projects/policy-finetuning
STARTS=$1; OUT=$2; N=$3; shift 3
TASKS=("$@"); STAGES=(reach pick carry place)
T=${TASKS[$((SLURM_ARRAY_TASK_ID / 4))]}; S=${STAGES[$((SLURM_ARRAY_TASK_ID % 4))]}
cd $P/abc-r2g
export ABC_REPO=$P/abc-r2g PYTHONPATH=$P/abc-r2g NO_RENDER=1 MUJOCO_GL=disable ABC_STAGE_STARTS=$P/abc_rl_data/abc-subtask-starts
$P/simulation/.venv/bin/python subtask_rewards/r2g/tools/scripted_rollouts.py "$T" "$S" "$STARTS" "$OUT" --n "$N"
