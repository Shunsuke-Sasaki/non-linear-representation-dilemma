#!/bin/bash
# Full pilot run for Issue 485 (8 conditions EEE..FFF, production config).
# Estimate (jobs 94780, 94835): worst case ~8h if every training hits max_epochs=50.
# pilot.py cannot resume (new results/<run_id>/ each run), so --time has ~2x margin.
# Submit only after the supervisor's start confirmation.
#SBATCH --job-name=0485_pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%j.out
#SBATCH --error=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%j.err

set -euo pipefail
EXP=/model/experiments/0485_shared-mapping-algo-pilot

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
echo "start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python code/pilot.py --config config/pilot_config.json
echo "end $(date -Is)"
du -sh $EXP/results
