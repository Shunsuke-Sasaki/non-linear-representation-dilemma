#!/bin/bash
# Issue 485, IOI: why some seeds fit namesplit C_name (code/ioi_namewhy.py; no map training). Usage (from the experiment directory):
#   sbatch code/ioi-namewhy.sbatch   (then: python3 code/ioi_namewhy_summary.py $EXP on the login node)
#SBATCH --job-name=0485_ioi-namewhy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=llm-jp-noble-gpu03,llm-jp-noble-gpu17
#SBATCH --output=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%A_%a.out
#SBATCH --error=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%A_%a.err

set -euo pipefail
EXP=/model/experiments/0485_shared-mapping-algo-pilot
STAGE=namewhy

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
python code/ioi_namewhy.py
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "end $(date -Is)"
du -sh $EXP/results
