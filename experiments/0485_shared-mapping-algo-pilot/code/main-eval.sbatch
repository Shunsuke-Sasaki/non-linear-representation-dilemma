#!/bin/bash
# Main evaluation of Issue 485. Usage (from the experiment directory):
#   sbatch code/main-eval.sbatch smoke                       # tests + tiny end-to-end run
#   sbatch code/main-eval.sbatch prepare
#   sbatch --array=0-2 code/main-eval.sbatch models
#   sbatch --array=0-7 code/main-eval.sbatch maps --num-tasks 8
#   sbatch --array=0-4 code/main-eval.sbatch adapt
#   sbatch code/main-eval.sbatch summarize
# Every stage is resumable: resubmitting skips finished outputs.
#SBATCH --job-name=0485_main-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%A_%a.out
#SBATCH --error=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%A_%a.err

set -euo pipefail
EXP=/model/experiments/0485_shared-mapping-algo-pilot
STAGE=$1; shift

module load cuda/12.8/12.8.2
module load python/3.12/3.12.13
source $EXP/environment/.venv/bin/activate

export WANDB_MODE=disabled
export WANDB_DIR=$EXP/logs/wandb
export TMPDIR=$EXP/environment/tmp
export PIP_CACHE_DIR=$EXP/environment/pip-cache
export XDG_CACHE_HOME=$EXP/environment/cache
export TORCH_HOME=$EXP/environment/cache/torch
export MPLCONFIGDIR=$EXP/environment/cache/matplotlib
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

cd $EXP
echo "start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-none} stage=$STAGE"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
if [ "$STAGE" = smoke ]; then
  python code/test_main_eval.py
fi
python code/main_eval.py "$STAGE" --task "${SLURM_ARRAY_TASK_ID:-0}" "$@"
echo "end $(date -Is)"
