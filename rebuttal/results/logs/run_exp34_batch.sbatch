#!/bin/bash
#SBATCH --job-name=epo-exp34
#SBATCH --account=usr-sr
#SBATCH --qos=h200_usr-sr_high
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/storage/home/impwxu/code/EPO/rebuttal/results/logs/exp34_batch.out
#SBATCH --error=/storage/home/impwxu/code/EPO/rebuttal/results/logs/exp34_batch.out
set -x
source /storage/home/impwxu/code/EPO/rebuttal/results/logs/env_common.sh
cd "$EPO_REPO" || exit 1
E3=$EPO_REPO/rebuttal/results/exp3
E4=$EPO_REPO/rebuttal/results/exp4
mkdir -p "$E3" "$E4"

# base flags per algorithm (sciworld, seed 0); EXTRA (appended) overrides via launcher arg-parse
common_args() {  # $1=algo
  local a="--environment sciworld --seed 0 --lr_warmup_steps_ratio 0.1 --min_lr_ratio 0.2 \
--warmup_style cosine --total_epochs 125 --entropy_smooth True --enable_smooth_weights True \
--entropy_smooth_mask_mode token --entropy_smooth_min_ratio 0 --entropy_smooth_max_ratio 2.0 \
--entropy_smooth_out_range_penalty 0.05 --entropy_coeff 0.001 \
--model_path $SCI_MODEL --model_load_method local --n_gpus 8"
  if [ "$1" = ppo ]; then
    echo "--rl_algorithm ppo --lr 3e-6 --ppo_mini_batch_size 64 --ppo_micro_batch_size_per_gpu 8 --log_prob_micro_batch_size_per_gpu 8 $a"
  else
    echo "--rl_algorithm grpo --lr 5e-6 --group_size 8 --ppo_mini_batch_size 128 --ppo_micro_batch_size_per_gpu 16 --log_prob_micro_batch_size_per_gpu 16 $a"
  fi
}

# spec: NAME | ALGO | LOGDIR | EXTRA
SPECS=(
  "exp3_toggle_se40|ppo|$E3|--entropy_smooth_start_epoch 40"
  "exp3_never_se9999|ppo|$E3|--entropy_smooth_start_epoch 9999"
  "exp3_plainppo_lr3e6|ppo|$E3|--entropy_smooth False --enable_smooth_weights False --entropy_coeff 0 --lr 3e-6"
  "exp3_plainppo_lr5e6|ppo|$E3|--entropy_smooth False --enable_smooth_weights False --entropy_coeff 0 --lr 5e-6"
  "exp3_plainppo_lr1e5|ppo|$E3|--entropy_smooth False --enable_smooth_weights False --entropy_coeff 0 --lr 1e-5"
  "exp4_ec0p0005|grpo|$E4|--entropy_coeff 0.0005"
  "exp4_ec0p002|grpo|$E4|--entropy_coeff 0.002"
  "exp4_pen0p1|grpo|$E4|--entropy_smooth_out_range_penalty 0.1"
  "exp4_pen0p2|grpo|$E4|--entropy_smooth_out_range_penalty 0.2"
  "exp4_kr1p5|grpo|$E4|--entropy_smooth_max_ratio 1.5"
  "exp4_kr2p5|grpo|$E4|--entropy_smooth_max_ratio 2.5"
  "exp4_esc0p5|grpo|$E4|--entropy_smooth_coeff 0.5"
  "exp4_esc2p0|grpo|$E4|--entropy_smooth_coeff 2.0"
)

readarray -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Allocated ${#NODES[@]} nodes: ${NODES[*]}"

i=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r NAME ALGO LOGDIR EXTRA <<< "$spec"
  node=${NODES[$i]}
  ARGS="$(common_args "$ALGO") $EXTRA"
  echo "[$i] launch $NAME on $node :: $EXTRA"
  srun --nodes=1 --ntasks=1 --nodelist="$node" --gres=gpu:h200:8 --cpus-per-task=96 --mem=0 --exclusive \
       --output="$LOGDIR/$NAME.out" --error="$LOGDIR/$NAME.out" \
       bash -c "
         source $EPO_REPO/rebuttal/results/logs/env_common.sh
         cd \$EPO_REPO
         export EPO_DATA_DIR=\$EPO_BASE/data/exp34_${SLURM_JOB_ID}_$i
         for c in /a100/epo_${SLURM_JOB_ID}_$i /dev/shm/epo_${SLURM_JOB_ID}_$i; do mkdir -p \$c 2>/dev/null && export TMPDIR=\$c && break; done
         mkdir -p \$EPO_DATA_DIR
         echo RUN=$NAME HOST=\$(hostname) TMPDIR=\$TMPDIR
         micromamba run -n \$EPO_ENV bash examples/general_running_server.sh $ARGS
         echo \"===== EXIT $NAME rc=\$? =====\"
       " &
  i=$((i+1))
done

echo "All ${#SPECS[@]} steps launched; waiting..."
wait
echo "===== ALL EXP34 STEPS DONE ====="
