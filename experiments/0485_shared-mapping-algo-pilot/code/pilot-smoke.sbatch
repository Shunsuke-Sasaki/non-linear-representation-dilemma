#!/bin/bash
#SBATCH --job-name=0485_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%j.out
#SBATCH --error=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%j.err

set -euo pipefail

module load cuda/12.8/12.8.2
module load python/3.12/3.12.13
source /model/experiments/0485_shared-mapping-algo-pilot/environment/.venv/bin/activate

export WANDB_MODE=disabled
export WANDB_DIR=/model/experiments/0485_shared-mapping-algo-pilot/logs/wandb

cd /model/experiments/0485_shared-mapping-algo-pilot
hostname
pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python code/test_pilot.py
python code/pilot.py --smoke
