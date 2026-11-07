#!/bin/bash
# IOI part of Issue 485 (Pythia-410m seeds). Usage (from the experiment directory):
#   sbatch code/ioi-eval.sbatch smoke                     # tiny end-to-end run with timings, then tests
#   sbatch code/ioi-eval.sbatch bench                     # full-size per-update timings, high vs highest precision (after smoke)
#   sbatch code/ioi-eval.sbatch prepare                   # download models and data, build pairs
#   sbatch --array=0-N code/ioi-eval.sbatch maps --num-tasks M [--kind individual] [--cand alpha]
#   sbatch --array=0-9 code/ioi-eval.sbatch adapt         # (fold, state) tasks
#   sbatch code/ioi-eval.sbatch summarize
# Every stage is resumable (maps also resume mid-training from *.resume.pt).
#SBATCH --job-name=0485_ioi-eval
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
export HF_HOME=$EXP/environment/cache/huggingface
export TORCH_HOME=$EXP/environment/cache/torch
export MPLCONFIGDIR=$EXP/environment/cache/matplotlib
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$WANDB_DIR"

cd $EXP
echo "start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID:-none} stage=$STAGE"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python code/ioi_eval.py "$STAGE" --task "${SLURM_ARRAY_TASK_ID:-0}" "$@"
if [ "$STAGE" = smoke ]; then
  python code/test_ioi_eval.py
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "end $(date -Is)"
du -sh $EXP/results $HF_HOME
