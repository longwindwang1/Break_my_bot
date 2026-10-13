#!/usr/bin/env bash
# Headless RSL-RL training wrapper around Isaac Lab's stock train.py.
#
# Usage:
#   isaac/train_policy.sh <task> <num_envs> <seed> <max_iter> <exp_name> [extra train.py args...]
# Example:
#   isaac/train_policy.sh Isaac-Velocity-Rough-Unitree-Go2-v0 4096 0 1500 w1_go2_rough_baseline
#
# Writes experiments/<exp_name>/{run_info.txt,train.log,vram.csv}.
# Isaac Lab itself writes checkpoints + params/{env,agent}.yaml to logs/rsl_rl/<experiment>/<timestamp>/
# (relative to the repo root, since we cd there).
set -euo pipefail

if [[ $# -lt 5 ]]; then
    sed -n '4,7p' "$0"
    exit 1
fi
TASK=$1 NUM_ENVS=$2 SEED=$3 MAX_ITER=$4 EXP_NAME=$5
shift 5

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
ISAACLAB_DIR=${ISAACLAB_DIR:-$HOME/IsaacLab}
CONDA_ENV=${CONDA_ENV:-$HOME/miniforge3/envs/env_isaaclab}
PYTHON=$CONDA_ENV/bin/python
# rsl_rl imports GitPython, which needs a git binary on PATH (git lives in the conda env).
export PATH=$CONDA_ENV/bin:$PATH
GIT=$(command -v git)

EXP_DIR=$REPO_ROOT/experiments/$EXP_NAME
RUN_NAME=${EXP_NAME//\//_}  # Isaac Lab log dir suffix; no slashes
mkdir -p "$EXP_DIR"
cd "$REPO_ROOT"

CMD=("$PYTHON" "$ISAACLAB_DIR/scripts/reinforcement_learning/rsl_rl/train.py"
     --task "$TASK" --num_envs "$NUM_ENVS" --seed "$SEED" --max_iterations "$MAX_ITER"
     --run_name "$RUN_NAME" --headless "$@")

{
    echo "date:        $(date -Iseconds)"
    echo "task:        $TASK"
    echo "num_envs:    $NUM_ENVS"
    echo "seed:        $SEED"
    echo "max_iter:    $MAX_ITER"
    echo "repo_commit: $("$GIT" -C "$REPO_ROOT" rev-parse -q --verify HEAD || echo none)"
    echo "repo_dirty:  $([[ -n $("$GIT" -C "$REPO_ROOT" status --porcelain 2>/dev/null) ]] && echo yes || echo no)"
    echo "isaaclab:    $("$GIT" -C "$ISAACLAB_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "gpu:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "command:     ${CMD[*]}"
} > "$EXP_DIR/run_info.txt"

# Sample GPU memory once per second for the whole run.
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 1000 > "$EXP_DIR/vram.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

START=$(date +%s)
set +e
"${CMD[@]}" 2>&1 | tee "$EXP_DIR/train.log"
STATUS=${PIPESTATUS[0]}
# Kit can exit 0 even when train.py raised, so also treat a Python traceback as failure.
if [[ $STATUS -eq 0 ]] && grep -q '^Traceback' "$EXP_DIR/train.log"; then STATUS=1; fi
set -e
END=$(date +%s)
kill $SMI_PID 2>/dev/null || true

LOG_DIR=$(ls -td logs/rsl_rl/*/*_"$RUN_NAME" 2>/dev/null | head -1 || true)
{
    echo "exit_status: $STATUS"
    echo "wall_time_s: $((END - START))"
    echo "peak_vram_mib: $(awk -F', ' 'max<$2{max=$2} END{print max}' "$EXP_DIR/vram.csv")"
    echo "log_dir:     ${LOG_DIR:-not found, see train.log}"
} >> "$EXP_DIR/run_info.txt"

cat "$EXP_DIR/run_info.txt"
exit "$STATUS"
