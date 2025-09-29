set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WG_BACKEND='ray'

# Default parameter settings
train_data_size=16
val_data_size=16
group_size=8
adaptive_start_epoch=0
val_one_time_size=128
rl_algorithm="grpo"
enable_entropy_reward_logging=False
entropy_threshold=1.2
seed=0
n_gpus=8
ENGINE="vllm"  # Assuming ENGINE has a default value, adjust according to actual situation
entropy_penalty_enable=False
entropy_penalty_weight=0.8
window_size=5
model_path="Qwen/Qwen2.5-7B-Instruct"
# Learning rate related parameters
lr=1e-6
lr_warmup_steps=-1
lr_warmup_steps_ratio=0.0
min_lr_ratio=0.0
num_cycles=0.5
warmup_style="constant"
weight_decay=0.01
entropy_regularization=False
start_multiplier=2.0
end_multiplier=1.0
lambda_decay=3.0
total_epochs=150

# Parameter parsing function
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --train_data_size <number>           Train data size (default: $train_data_size)"
    echo "  --val_data_size <number>             Validation data size (default: $val_data_size)"
    echo "  --group_size <number>                Group size (default: $group_size)"
    echo "  --adaptive_start_epoch <number>      Adaptive start epoch (default: $adaptive_start_epoch)"
    echo "  --val_one_time_size <number>         Validation one time size (default: $val_one_time_size)"
    echo "  --rl_algorithm <string>              RL algorithm (default: $rl_algorithm)"
    echo "  --enable_entropy_reward_logging <bool>   Enable entropy reward logging (default: $enable_entropy_reward_logging)"
    echo "  --entropy_threshold <number>         Entropy threshold for detection (default: $entropy_threshold)"
    echo "  --seed <number>                      Random seed (default: $seed)"
    echo "  --n_gpus <number>                    Number of GPUs per node (default: $n_gpus)"
    echo "  --engine <string>                    Engine name (default: $ENGINE)"
    echo "  --entropy_penalty_enable <bool>      Enable entropy penalty (default: $entropy_penalty_enable)"
    echo "  --entropy_penalty_weight <number>    Entropy penalty weight (default: $entropy_penalty_weight)"
    echo "  --window_size <number>               Window size for entropy detection (default: $window_size)"
    echo "  --model_path <string>                Model path (default: $model_path)"
    echo "  --lr <number>                        Learning rate (default: $lr)"
    echo "  --lr_warmup_steps <number>           Learning rate warmup steps (default: $lr_warmup_steps)"
    echo "  --lr_warmup_steps_ratio <number>     Learning rate warmup steps ratio (default: $lr_warmup_steps_ratio)"
    echo "  --min_lr_ratio <number>              Minimum learning rate ratio (default: $min_lr_ratio)"
    echo "  --num_cycles <number>                Number of cycles (default: $num_cycles)"
    echo "  --warmup_style <string>              Warmup style (default: $warmup_style)"
    echo "  --weight_decay <number>              Weight decay (default: $weight_decay)"
    echo "  --entropy_regularization <bool>      Entropy regularization (default: $entropy_regularization)"
    echo "  --start_multiplier <number>          Start multiplier (default: $start_multiplier)"
    echo "  --end_multiplier <number>            End multiplier (default: $end_multiplier)"
    echo "  --lambda_decay <number>              Lambda decay (default: $lambda_decay)"
    echo "  --total_epochs <number>              Total epochs (default: $total_epochs)"
    echo "  -h, --help                           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --train_data_size 8 --val_data_size 32 --group_size 16"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train_data_size)
            train_data_size="$2"
            shift 2
            ;;
        --val_data_size)
            val_data_size="$2"
            shift 2
            ;;
        --group_size)
            group_size="$2"
            shift 2
            ;;
        --adaptive_start_epoch)
            adaptive_start_epoch="$2"
            shift 2
            ;;
        --val_one_time_size)
            val_one_time_size="$2"
            shift 2
            ;;
        --rl_algorithm)
            rl_algorithm="$2"
            shift 2
            ;;
        --enable_entropy_reward_logging)
            enable_entropy_reward_logging="$2"
            shift 2
            ;;
        --entropy_threshold)
            entropy_threshold="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --n_gpus)
            n_gpus="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --entropy_penalty_enable)
            entropy_penalty_enable="$2"
            shift 2
            ;;
        --entropy_penalty_weight)
            entropy_penalty_weight="$2"
            shift 2
            ;;
        --window_size)
            window_size="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --lr_warmup_steps)
            lr_warmup_steps="$2"
            shift 2
            ;;
        --lr_warmup_steps_ratio)
            lr_warmup_steps_ratio="$2"
            shift 2
            ;;
        --min_lr_ratio)
            min_lr_ratio="$2"
            shift 2
            ;;
        --num_cycles)
            num_cycles="$2"
            shift 2
            ;;
        --warmup_style)
            warmup_style="$2"
            shift 2
            ;;
        --weight_decay)
            weight_decay="$2"
            shift 2
            ;;
        --entropy_regularization)
            entropy_regularization="$2"
            shift 2
            ;;
        --start_multiplier)
            start_multiplier="$2"
            shift 2
            ;;
        --end_multiplier)
            end_multiplier="$2"
            shift 2
            ;;
        --lambda_decay)
            lambda_decay="$2"
            shift 2
            ;;
        --total_epochs)
            total_epochs="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Print current configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "Train data size: $train_data_size"
echo "Validation data size: $val_data_size"
echo "Group size: $group_size"
echo "Adaptive start epoch: $adaptive_start_epoch"
echo "Validation one time size: $val_one_time_size"
echo "RL algorithm: $rl_algorithm"
echo "Enable entropy reward logging: $enable_entropy_reward_logging"
echo "Entropy threshold: $entropy_threshold"
echo "Random seed: $seed"
echo "Number of GPUs per node: $n_gpus"
echo "Engine: $ENGINE"
echo "Entropy penalty enable: $entropy_penalty_enable"
echo "Entropy penalty weight: $entropy_penalty_weight"
echo "Window size: $window_size"
echo "Model path: $model_path"
echo "Learning rate: $lr"
echo "Learning rate warmup steps: $lr_warmup_steps"
echo "Learning rate warmup steps ratio: $lr_warmup_steps_ratio"
echo "Minimum learning rate ratio: $min_lr_ratio"
echo "Number of cycles: $num_cycles"
echo "Warmup style: $warmup_style"
echo "Weight decay: $weight_decay"
echo "Entropy regularization: $entropy_regularization"
echo "Start multiplier: $start_multiplier"
echo "End multiplier: $end_multiplier"
echo "Lambda decay: $lambda_decay"
echo "Total epochs: $total_epochs"
train_val_ratio=$(echo "scale=2; $train_data_size / $val_data_size" | bc -l)
echo "Train/Val ratio: $train_val_ratio"
echo "=========================================="

# Generate experiment name based on algorithm
# Convert lr to a filename-friendly format (replace 'e-' with 'e' and '.' with 'p')
lr_str=$(echo "$lr" | sed 's/e-/e/g' | sed 's/\./p/g')

# Extract model size from model path for experiment name
model_identifier=$(echo "$model_path" | sed 's/.*\///g' | sed 's/-Instruct//g' | sed 's/Qwen2\.5-//' | tr '[:upper:]' '[:lower:]')

if [ "$rl_algorithm" = "grpo" ]; then
    # For GRPO: only include relevant parameters
    if [ "$entropy_regularization" = "True" ]; then
        experiment_name="grpo_${model_identifier}_s${seed}_tbs${train_data_size}_vbs${val_data_size}_gs${group_size}_lr${lr_str}_er_sm${start_multiplier}_em${end_multiplier}_ld${lambda_decay}"
    else
        experiment_name="grpo_${model_identifier}_s${seed}_tbs${train_data_size}_vbs${val_data_size}_gs${group_size}_lr${lr_str}"
    fi
else
    # For OURS: include adaptive parameters
    if [ "$entropy_regularization" = "True" ]; then
        experiment_name="ours_${model_identifier}_s${seed}_tbs${train_data_size}_vbs${val_data_size}_gs${group_size}_ae${adaptive_start_epoch}_ws${window_size}_lr${lr_str}_er_sm${start_multiplier}_em${end_multiplier}_ld${lambda_decay}"
    else
        experiment_name="ours_${model_identifier}_s${seed}_tbs${train_data_size}_vbs${val_data_size}_gs${group_size}_ae${adaptive_start_epoch}_ws${window_size}_lr${lr_str}"
    fi
fi

echo "Generated experiment name: $experiment_name"

# Data preprocessing
echo "Starting data preprocessing..."
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$rl_algorithm \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$lr_warmup_steps_ratio \
    actor_rollout_ref.actor.optim.min_lr_ratio=$min_lr_ratio \
    actor_rollout_ref.actor.optim.num_cycles=$num_cycles \
    actor_rollout_ref.actor.optim.warmup_style=$warmup_style \
    actor_rollout_ref.actor.optim.weight_decay=$weight_decay \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.dtype=half \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.entropy_regularization=$entropy_regularization \
    actor_rollout_ref.actor.start_multiplier=$start_multiplier \
    actor_rollout_ref.actor.end_multiplier=$end_multiplier \
    actor_rollout_ref.actor.lambda_decay=$lambda_decay \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=$seed \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.enable_entropy_reward_logging=$enable_entropy_reward_logging \
    env.entropy_threshold=$entropy_threshold \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop_'$rl_algorithm \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=True \
    trainer.val_one_time_size=$val_one_time_size \
    trainer.rl_algorithm=$rl_algorithm \
    env.adaptive_start_epoch=$adaptive_start_epoch \
    trainer.entropy_penalty_enable=$entropy_penalty_enable \
    trainer.entropy_penalty_weight=$entropy_penalty_weight \
    trainer.window_size=$window_size 