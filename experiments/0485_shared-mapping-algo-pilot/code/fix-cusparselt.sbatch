#!/bin/bash
# One-off repair (2026-11-09): libcusparseLt.so.0 in the venv became unreadable (reads hang in cl_sync_io_wait;
# single stripe on large-OST0026). Keep the bad file aside for the admins, reinstall the wheel, verify torch on GPU.
#SBATCH --job-name=0485_fix-cusparselt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%j.out
#SBATCH --error=/model/experiments/0485_shared-mapping-algo-pilot/logs/%x-%j.err
set -euxo pipefail
EXP=/model/experiments/0485_shared-mapping-algo-pilot
module load cuda/12.8/12.8.2
module load python/3.12/3.12.13
source $EXP/environment/.venv/bin/activate
export PIP_CACHE_DIR=$EXP/environment/pip-cache
export TMPDIR=$EXP/environment/tmp
export XDG_CACHE_HOME=$EXP/environment/cache
mkdir -p $TMPDIR
hostname
F=$EXP/environment/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0
if [ -e "$F" ] && [ ! -e "$F.unreadable-ost38" ]; then
  mv "$F" "$F.unreadable-ost38"   # rename only (metadata); the contents are never read
fi
pip install --force-reinstall --no-deps nvidia-cusparselt-cu12==0.7.1
ls -la "$(dirname "$F")"
lfs getstripe -i "$F"
timeout 120 cat "$F" > /dev/null
echo "read OK"
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x = torch.randn(64, 64, device="cuda"); print((x @ x).sum().item())'
