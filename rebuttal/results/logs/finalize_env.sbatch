#!/bin/bash
#SBATCH --job-name=epo-finalize
#SBATCH --account=usr-sr
#SBATCH --qos=h200_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=/storage/home/impwxu/code/EPO/rebuttal/results/logs/finalize.out
#SBATCH --error=/storage/home/impwxu/code/EPO/rebuttal/results/logs/finalize.out
set -x
set -o pipefail
source /storage/home/impwxu/code/EPO/rebuttal/results/logs/env_common.sh
cd "$EPO_REPO" || exit 1

echo "===== [$(date)] HOST=$(hostname) finalize env ====="

echo "===== [$(date)] pin setuptools<81 (verl needs pkg_resources) ====="
epo_run pip install "setuptools<81" 2>&1 | tail -2 || exit 5

echo "===== [$(date)] install openjdk 17 (ScienceWorld JVM) if missing ====="
if ! epo_run java -version 2>/dev/null; then
  micromamba install -y -n "$EPO_ENV" -c conda-forge openjdk=17 || exit 10
fi
epo_run java -version 2>&1 | head -1 || exit 11

echo "===== [$(date)] import check (torch/vllm/verl/scienceworld + JVM launch) ====="
epo_run python -c "import torch, vllm, verl, scienceworld; print('torch', torch.__version__, '| vllm', vllm.__version__, '| GPU', torch.cuda.is_available())" || exit 20
# scienceworld first-use spawns the JVM — the real prereq test:
epo_run python -c "from scienceworld import ScienceWorldEnv; e=ScienceWorldEnv('', None, envStepLimit=30); print('SciWorld JVM OK; tasks:', len(e.get_task_names()))" || exit 21

echo "===== [$(date)] smoke test (GATE) ====="
epo_run python tests/test_epo_smoke.py || exit 30

echo "===== [$(date)] data preprocessing ====="
epo_run python3 -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size 16 || exit 40
ls -la $HOME/data/verl-agent/text/ | head

echo "===== [$(date)] W&B access check ====="
epo_run python -c "import os,wandb; api=wandb.Api(); r=list(api.runs(os.environ['WANDB_ENTITY']+'/verl_agent_sciworld_ppo')); print('W&B sciworld_ppo runs:', len(r))" || echo "WARN: W&B check failed (will retry in Exp2)"

echo "===== [$(date)] FINALIZE COMPLETE ====="
