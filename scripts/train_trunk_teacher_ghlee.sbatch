#!/bin/bash
# Candidate runner owned by e06_trunk_teacher_ghlee. Submit from ProtoMotions,
# or set REPO_ROOT to a frozen checkout. MOTION_FILE and PYTHON_BIN are explicit
# machine-local inputs; the motion must have the human_model_v2 59-DOF schema.
# Example: MOTION_FILE=/data/v3.pt PYTHON_BIN=/venv/bin/python \
#   RUN_ROOT=/data/experiment sbatch scripts/train_trunk_teacher_ghlee.sbatch
# Override ENVS, BATCH, SEED, MAX_STEPS and RUN_NAME for comparison runs.
# Use argument "smoke" for six epochs at the same environment/batch allocation.
#SBATCH --partition=train
#SBATCH --gres=gpu:48g:1
#SBATCH --job-name=trunk_teacher_ghlee
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=trunk_teacher_ghlee_%j.log
set -euo pipefail

: "${MOTION_FILE:?Set MOTION_FILE to the retargeted v3 motion pack}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the Isaac Lab Python environment}"
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$REPO_ROOT"
ENVS=${ENVS:-6144}
BATCH=${BATCH:-49152}
SEED=${SEED:-2933}
MAX_STEPS=${MAX_STEPS:-500000000}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/results/trunk_teacher_ghlee}
RUN_NAME=${RUN_NAME:-hm_v2_trunk_v3_stage_a_seed${SEED}_ghlee}
ROBOT_NAME=${ROBOT_NAME:-human_model_v2}
EXPERIMENT_PATH=${EXPERIMENT_PATH:-examples/experiments/steering/mlp_human_model_v2_stage_a_trunk_ghlee.py}
MODE=${1:-train}
if [[ "$MODE" == smoke ]]; then
    MAX_STEPS=$((ENVS * 32 * 6))
    RUN_NAME="${RUN_NAME}_smoke"
elif [[ "$MODE" != train ]]; then
    echo "Expected train or smoke" >&2
    exit 2
fi
test -f "$MOTION_FILE"
test -f "$EXPERIMENT_PATH"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OMNI_KIT_ACCEPT_EULA=YES
resources() {
    free -h
    swapon --show
    nvidia-smi || true
    ps -u "$(id -un)" -o pid,ppid,etimes,rss,pcpu,args --sort=-rss | head -20 || true
    pstree -ap $$ || true
}
trap resources EXIT
resources
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import subprocess
m = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
output = subprocess.check_output(['vmstat', '1', '3'], text=True)
print(output, flush=True)
header = output.splitlines()[1].split()
rows = [dict(zip(header, line.split())) for line in output.splitlines()[-2:]]
swap_fraction = (m['SwapTotal'] - m['SwapFree']) / max(1, m['SwapTotal'])
swap_io = all(int(r['si']) > 0 or int(r['so']) > 0 for r in rows)
if m['MemAvailable'] < 4 * 1024**2 or (swap_fraction >= .75 and swap_io):
    raise SystemExit('Insufficient system memory or sustained swap I/O; launch stopped')
PY
"$PYTHON_BIN" -u protomotions/train_agent.py \
    --robot-name "$ROBOT_NAME" --simulator isaaclab \
    --experiment-path "$EXPERIMENT_PATH" \
    --experiment-name "$RUN_ROOT/models/$RUN_NAME" \
    --motion-file "$MOTION_FILE" --num-envs "$ENVS" --batch-size "$BATCH" \
    --seed "$SEED" --training-max-steps "$MAX_STEPS"
