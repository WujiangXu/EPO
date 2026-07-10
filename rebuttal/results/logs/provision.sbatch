#!/bin/bash
#SBATCH --job-name=epo-provision
#SBATCH --account=usr-sr
#SBATCH --qos=h200_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --output=/storage/home/impwxu/code/EPO/rebuttal/results/logs/provision.out
#SBATCH --error=/storage/home/impwxu/code/EPO/rebuttal/results/logs/provision.out

set -x
set -o pipefail
source /storage/home/impwxu/code/EPO/rebuttal/results/logs/env_common.sh
cd "$EPO_REPO" || exit 1

echo "===== [$(date)] HOST=$(hostname) ; EPO_BASE=$EPO_BASE ====="
df -h "$EPO_BASE" | tail -1
echo "===== [$(date)] create micromamba env ($EPO_ENV, python 3.10) ====="
if micromamba env list 2>/dev/null | grep -qE "/envs/$EPO_ENV\b|\b$EPO_ENV\s"; then
  echo "env $EPO_ENV already exists; skipping create"
else
  micromamba create -y -n "$EPO_ENV" -c conda-forge python=3.10 pip setuptools wheel || exit 10
fi

echo "===== [$(date)] scienceworld + gym + selenium ====="
epo_run pip install scienceworld gym==0.23.1 selenium || exit 20

echo "===== [$(date)] torch 2.6.0 cu124 ====="
epo_run pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 || exit 30

echo "===== [$(date)] flash-attn build deps (psutil/ninja/packaging) ====="
epo_run pip install psutil ninja packaging || exit 35

echo "===== [$(date)] flash-attn 2.7.4.post1 ====="
MAX_JOBS=32 epo_run pip install flash-attn==2.7.4.post1 --no-build-isolation || exit 40

echo "===== [$(date)] verl (pip install -e .) ====="
epo_run pip install -e . || exit 50
echo "===== [$(date)] pin setuptools<81 (verl imports pkg_resources) ====="
epo_run pip install "setuptools<81" || exit 52

echo "===== [$(date)] vllm 0.8.5 ====="
epo_run pip install vllm==0.8.5 || exit 60

echo "===== [$(date)] analysis + hf + wandb deps ====="
epo_run pip install wandb pandas numpy matplotlib scipy "huggingface_hub[hf_transfer]" || exit 70

echo "===== [$(date)] openjdk 17 (ScienceWorld JVM via py4j) ====="
micromamba install -y -n "$EPO_ENV" -c conda-forge openjdk=17 || exit 72
epo_run java -version 2>&1 | head -1 || exit 73

echo "===== [$(date)] download Qwen2.5-7B-Instruct -> $SCI_MODEL ====="
HF_HUB_ENABLE_HF_TRANSFER=1 epo_run hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$SCI_MODEL" || exit 80

echo "===== [$(date)] import check ====="
epo_run python -c "import torch, vllm, verl, scienceworld; print('torch', torch.__version__, '| vllm', vllm.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.is_available())" || exit 90

echo "===== [$(date)] model files ====="
ls -la "$SCI_MODEL" | head

echo "===== [$(date)] PROVISION COMPLETE ====="
