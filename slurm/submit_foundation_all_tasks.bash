#!/usr/bin/env bash
# Submit one independent four-hour Slurm job per scored canonical task.
# Defaults: all 29 tasks, 100 worlds/task, all videos, official RTC/reset behavior.
set -euo pipefail
umask 027
ROOT=/projects/work/yang-lab/projects/policy-finetuning/abc
cd "$ROOT"
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/.venv-sim-eval/bin/python}"
TASK_FILE="${TASK_FILE:-$ROOT/slurm/foundation_tasks.txt}"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/foundation_200k/all_tasks/run-$(date -u +%Y%m%dT%H%M%SZ)}"
export NUM_WORLDS="${NUM_WORLDS:-100}"
export SAVE_VIDEO="${SAVE_VIDEO:-1}"
export EVAL_PYTHON

# Resolve and validate the whole list before submitting its first job.
"$EVAL_PYTHON" - "$TASK_FILE" <<'PY'
from pathlib import Path
import sys
import abc_sim
names = Path(sys.argv[1]).read_text().splitlines()
if not names or len(names) != len(set(names)):
    raise ValueError("Task list must be nonempty and contain no duplicate names")
for name in names:
    spec = abc_sim.get_task_spec(name)
    if name != spec.name or spec.evaluator_name is None:
        raise ValueError(f"Expected a canonical scored task, got {name!r}")
print(f"Submitting {len(names)} separate jobs; 04:00:00 per task")
PY
[[ ! -e "$RUN_ROOT/jobs.tsv" ]] || { echo "ERROR: existing submission manifest: $RUN_ROOT/jobs.tsv" >&2; exit 2; }
mkdir -p "$RUN_ROOT" "$ROOT/../policy-finetuning-logs/abc-foundation"
cp "$TASK_FILE" "$RUN_ROOT/tasks.txt"
printf 'job_id\ttask\toutput_dir\n' > "$RUN_ROOT/jobs.tsv"
while IFS= read -r task; do
    job=$(/usr/bin/env -u OUTPUT_DIR TASK="$task" OUTPUT_ROOT="$RUN_ROOT" \
        sbatch --parsable --export=ALL --time=04:00:00 \
        --job-name="abc200k_${task}" slurm/eval_foundation_task.bash)
    job_id="${job%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "ERROR: unexpected sbatch response: $job" >&2; exit 2; }
    output="$RUN_ROOT/$task/job-$job_id"
    printf '%s\t%s\t%s\n' "$job_id" "$task" "$output" >> "$RUN_ROOT/jobs.tsv"
    echo "$job_id $task -> $output"
done < "$RUN_ROOT/tasks.txt"
echo "Submission manifest: $RUN_ROOT/jobs.tsv"
