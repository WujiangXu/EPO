#!/bin/bash
set -x
set -o pipefail
export MODULEPATH=/public/modulefiles:$MODULEPATH
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.4.1
export CUDA_HOME=/public/apps/cuda/12.4.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAMBA_ROOT_PREFIX=$HOME/micromamba

cd /storage/home/impwxu/code/EPO
ENVNAME=verl-agent-sciworld
RUN="micromamba run -n $ENVNAME"

echo "===== [$(date)] create micromamba env (python 3.10) ====="
micromamba create -y -n $ENVNAME -c conda-forge python=3.10 pip setuptools wheel || exit 10
$RUN python -m pip install --upgrade pip setuptools wheel || exit 12

echo "===== [$(date)] scienceworld + gym + selenium ====="
$RUN pip install scienceworld gym==0.23.1 selenium || exit 20

echo "===== [$(date)] torch 2.6.0 cu124 ====="
$RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 || exit 30

echo "===== [$(date)] flash-attn 2.7.4.post1 ====="
$RUN pip install flash-attn==2.7.4.post1 --no-build-isolation || exit 40

echo "===== [$(date)] verl (pip install -e .) ====="
$RUN pip install -e . || exit 50

echo "===== [$(date)] vllm 0.8.5 ====="
$RUN pip install vllm==0.8.5 || exit 60

echo "===== [$(date)] analysis + hf deps ====="
$RUN pip install wandb pandas numpy matplotlib scipy "huggingface_hub[cli]" || exit 70

echo "===== [$(date)] import check ====="
$RUN python -c "import torch, vllm, verl, scienceworld; print('torch', torch.__version__, '| vllm', vllm.__version__, '| CUDA build', torch.version.cuda)" || exit 80

echo "===== [$(date)] BUILD COMPLETE ====="
