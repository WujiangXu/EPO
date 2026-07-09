#!/usr/bin/env bash
# End-to-end EPO smoke test — tiny PPO+EPO run to confirm the launcher fix works on real GPUs.
#
# PURPOSE: before the launcher fix, `PPO+EPO` launched via general_running_server.sh silently ran as
# plain PPO (the entropy_smooth* flags were never forwarded on the PPO branch). This runs a *tiny*
# PPO+EPO job and greps the log to confirm the corridor-smoothing code path is now actually entered.
#
# Run in the project env on a machine with >=2 free GPUs (vllm tensor_model_parallel_size=2 is
# hardcoded in the launcher) and the ALFWorld/ScienceWorld env installed + data preprocessable.
#
#   cd epo_code
#   MODEL=/path/to/Qwen2.5-0.5B-Instruct GPUS=2 bash tests/run_epo_smoke.sh
#   # or a HF id:
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct LOAD=huggingface GPUS=2 bash tests/run_epo_smoke.sh
#
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
LOAD=${LOAD:-huggingface}          # huggingface | local
ENVN=${ENVN:-alfworld}             # alfworld has max_steps=50 so episodes reach the phase gate (>=25)
GPUS=${GPUS:-2}
LOG=${LOG:-/tmp/epo_smoke_e2e.log}

echo ">>> EPO e2e smoke test | model=$MODEL env=$ENVN gpus=$GPUS -> log=$LOG"

# Tiny run: 2 RL steps, batch of 2, min corridor floor ON (kappa_l=0.8) to exercise both branches.
bash examples/general_running_server.sh \
  --environment "$ENVN" --rl_algorithm ppo --seed 0 \
  --total_epochs 2 \
  --train_data_size 2 --val_data_size 2 --group_size 2 --val_one_time_size 2 \
  --entropy_smooth True --enable_smooth_weights True --entropy_smooth_mask_mode token \
  --entropy_smooth_min_ratio 0.8 --entropy_smooth_max_ratio 2.0 \
  --entropy_smooth_out_range_penalty 0.1 --entropy_coeff 0.001 \
  --entropy_smooth_start_epoch 0 --entropy_smooth_gamma 3.0 --entropy_smooth_max_epochs 2 \
  --use_sliding_window False --window_size 5 \
  --ppo_mini_batch_size 4 --ppo_micro_batch_size_per_gpu 2 --log_prob_micro_batch_size_per_gpu 2 \
  --model_path "$MODEL" --model_load_method "$LOAD" --n_gpus "$GPUS" 2>&1 | tee "$LOG"

echo ""
echo ">>> ===== SMOKE ASSERTIONS ====="
rc=0

# (1) The PPO+EPO path is actually entered (this print only happens inside `if self.config.entropy_smooth`).
if grep -q "Should apply entropy smooth:" "$LOG"; then
  echo "PASS  PPO branch now forwards entropy_smooth -> smoothing code path entered."
else
  echo "FAIL  smoothing path NOT entered (launcher fix not effective, or crashed early). Check $LOG."
  rc=1
fi

# (2) The corridor penalty was actually computed (requires an episode to reach turn >= 25).
if grep -q "Token mode:" "$LOG"; then
  echo "PASS  corridor penalty computed (generate_entropy_penalty ran)."
else
  echo "WARN  penalty not computed — likely no episode reached the phase gate (turn>=25) in this tiny run."
  echo "      The unit test (tests/test_epo_smoke.py) covers the penalty logic deterministically."
fi

# (3) No crash / run reached at least one validation or step.
if grep -qiE "step:1|global_step|val/|validation" "$LOG"; then
  echo "PASS  training loop advanced without crashing."
else
  echo "WARN  could not confirm the loop advanced — inspect $LOG."
fi

echo ">>> ============================="
[ $rc -eq 0 ] && echo "E2E SMOKE: OK" || echo "E2E SMOKE: FAILED"
exit $rc
