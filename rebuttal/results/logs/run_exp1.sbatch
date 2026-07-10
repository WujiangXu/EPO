#!/bin/bash
#SBATCH --job-name=epo-exp1
#SBATCH --account=usr-sr
#SBATCH --qos=h200_rl_shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --time=24:00:00
# --output/--error set at submit time via sbatch -o/-e
set -x
set -o pipefail

# Required env from submit: ALGO (ppo|grpo), KL (0|0.5|0.8), SEED (0|1)
: "${ALGO:?set ALGO}"; : "${KL:?set KL}"; : "${SEED:?set SEED}"

source /storage/home/impwxu/code/EPO/rebuttal/results/logs/env_common.sh
cd "$EPO_REPO" || exit 1

# node-local scratch for ray/vllm temp (compute /tmp is a 1G tmpfs). Ray honors TMPDIR.
for cand in "/a100/epo_$SLURM_JOB_ID" "/dev/shm/epo_$SLURM_JOB_ID"; do
  if mkdir -p "$cand" 2>/dev/null; then export TMPDIR="$cand"; break; fi
done
echo "TMPDIR=$TMPDIR  HOST=$(hostname)  ALGO=$ALGO KL=$KL SEED=$SEED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

COMMON=(--environment sciworld --seed "$SEED"
        --lr_warmup_steps_ratio 0.1 --min_lr_ratio 0.2 --warmup_style cosine
        --total_epochs 125
        --entropy_smooth True --enable_smooth_weights True --entropy_smooth_mask_mode token
        --entropy_smooth_min_ratio "$KL" --entropy_smooth_max_ratio 2.0
        --entropy_smooth_out_range_penalty 0.05 --entropy_coeff 0.001
        --model_path "$SCI_MODEL" --model_load_method local --n_gpus 8)

if [ "$ALGO" = "ppo" ]; then
  ARGS=(--rl_algorithm ppo --lr 3e-6
        --ppo_mini_batch_size 64 --ppo_micro_batch_size_per_gpu 8
        --log_prob_micro_batch_size_per_gpu 8 "${COMMON[@]}")
elif [ "$ALGO" = "grpo" ]; then
  ARGS=(--rl_algorithm grpo --lr 5e-6 --group_size 8
        --ppo_mini_batch_size 128 --ppo_micro_batch_size_per_gpu 16
        --log_prob_micro_batch_size_per_gpu 16 "${COMMON[@]}")
else
  echo "bad ALGO=$ALGO"; exit 2
fi

echo "===== [$(date)] launch: $ALGO kl=$KL seed=$SEED ====="
micromamba run -n "$EPO_ENV" bash examples/general_running_server.sh "${ARGS[@]}"
rc=$?
echo "===== [$(date)] EXIT rc=$rc ====="
rm -rf "$TMPDIR" 2>/dev/null
exit $rc
