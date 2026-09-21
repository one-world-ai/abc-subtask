#!/usr/bin/env bash
#SBATCH --job-name=abc200k_put100
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning/abc
#SBATCH --account=torch_pr_147_courant
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/abc-foundation/%x-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/abc-foundation/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail
umask 027

ROOT=/projects/work/yang-lab/projects/policy-finetuning/abc
cd "$ROOT"
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/.venv-sim-eval/bin/python}"
CHECKPOINT="${CHECKPOINT:-$ROOT/cache/abc_dit_xl_200k_model.pt}"
NUM_WORLDS="${NUM_WORLDS:-100}"
NUM_CHUNKS="${NUM_CHUNKS:-236}"
SCENE_SEED="${SCENE_SEED:-20260511}"
POLICY_SEED="${POLICY_SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
RTC="${RTC:-1}"
FAST_INFERENCE="${FAST_INFERENCE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/foundation_200k/put_plastic_bottles_in_bin/job-${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}}"

for name in NUM_WORLDS NUM_CHUNKS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: $name must be positive" >&2; exit 2; }
done
for name in SCENE_SEED POLICY_SEED; do
    [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "ERROR: $name must be nonnegative" >&2; exit 2; }
done
for name in SAVE_VIDEO RTC FAST_INFERENCE; do
    [[ "${!name}" == 0 || "${!name}" == 1 ]] || { echo "ERROR: $name must be 0 or 1" >&2; exit 2; }
done
[[ -x "$EVAL_PYTHON" ]] || { echo "ERROR: missing $EVAL_PYTHON" >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR/summary.json" && ! -e "$OUTPUT_DIR/launch.json" ]] || {
    echo "ERROR: output directory already contains a run: $OUTPUT_DIR" >&2; exit 2;
}
mkdir -p "$OUTPUT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WARP_CACHE_PATH="$ROOT/cache/sim-eval/warp"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/sim-eval/torchinductor"
export TRITON_CACHE_DIR="$ROOT/cache/sim-eval/triton"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# The system Python on compute nodes lacks development headers. Supply the
# matching cached headers to both Triton's C compiler and Inductor's C++ compiler.
if [[ "$FAST_INFERENCE" == 1 ]]; then
    PYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR:-$ROOT/cache/sim-eval/python-headers/include/python3.12}"
    [[ -f "$PYTHON_INCLUDE_DIR/Python.h" ]] || {
        echo "ERROR: missing $PYTHON_INCLUDE_DIR/Python.h; see slurm/README.md" >&2; exit 2;
    }
    export CPATH="$PYTHON_INCLUDE_DIR${CPATH:+:$CPATH}"
    echo "Python development headers: $PYTHON_INCLUDE_DIR"
fi

echo "checkpoint=$CHECKPOINT output=$OUTPUT_DIR"
echo "worlds=$NUM_WORLDS chunks=$NUM_CHUNKS rtc=$RTC reset=upstream-carried-pose"

ARGS=(
    --checkpoint "$CHECKPOINT"
    --policy dit
    --task put_plastic_bottles_in_bin
    --num-worlds "$NUM_WORLDS" --parallel-worlds 0
    --num-chunks "$NUM_CHUNKS" --execute-chunk-dim 15
    --seed "$SCENE_SEED" --policy-seed "$POLICY_SEED"
    --diffusion-steps 10 --rtc-prefix-length 4 --rtc-inference-lead-steps 4
    --camera-backend mjwarp --camera-height 168 --camera-width 224
    --device cuda --gpu-id 0
    --clip.cache-dir "$ROOT/cache/clip"
    --output-dir "$OUTPUT_DIR"
    --log-every-chunk
)
if [[ "$RTC" == 1 ]]; then ARGS+=(--rtc); else ARGS+=(--no-rtc --prefix-length 0); fi
if [[ "$FAST_INFERENCE" == 1 ]]; then ARGS+=(--fast-inference); else ARGS+=(--no-fast-inference); fi
if [[ "$SAVE_VIDEO" == 1 ]]; then
    # 15 actions per frame: approximately 2 FPS in simulated time.
    ARGS+=(--save-video --video-every-n-actions 15 --video-fps 2)
else
    ARGS+=(--no-save-video)
fi

# Save exact arguments plus the local changes that define this evaluation.
"$EVAL_PYTHON" - "$OUTPUT_DIR/launch.json" "${ARGS[@]}" <<'PY'
import datetime, json, os, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "python": sys.executable,
    "argv": ["eval_policy.py", *sys.argv[2:]],
}, indent=2) + "\n")
PY
git diff -- abc_minimal/config.py abc_minimal/eval_policy.py > "$OUTPUT_DIR/evaluator.patch"
"$EVAL_PYTHON" eval_policy.py "${ARGS[@]}"
"$EVAL_PYTHON" - "$OUTPUT_DIR/summary.json" "$NUM_WORLDS" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["num_worlds"] == int(sys.argv[2]), "Incomplete world count"
print({key: result.get(key) for key in (
    "task", "prompt", "num_worlds", "num_success", "success_rate", "mean_reward", "mean_max_progress"
)})
PY
echo "Completed: $OUTPUT_DIR/summary.json"
