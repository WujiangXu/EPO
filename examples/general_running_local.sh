set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WG_BACKEND='ray'
export WANDB_USER_NAME="ruwujiang"
export WANDB_API_KEY="9d760fc60646f681ea153bbccf0ed46bd117963d"

# Default parameter settings
environment="alfworld"  # Environment name: alfworld, webshop, sciworld
train_data_size=16
val_data_size=16
group_size=8
adaptive_start_epoch=0
val_one_time_size=128
rl_algorithm="grpo"  # RL algorithm: grpo, ppo
enable_entropy_reward_logging=False
entropy_threshold=1.2
seed=0
n_gpus=8
ENGINE="vllm"  # Assuming ENGINE has a default value, adjust according to actual situation
window_size=5
# Learning rate related parameters
lr=1e-6
lr_warmup_steps=-1
lr_warmup_steps_ratio=0.0
min_lr_ratio=0.0
num_cycles=0.5
warmup_style="constant"
weight_decay=0.01
total_epochs=150
entropy_smooth=False
entropy_smooth_mask_mode="token"
entropy_smooth_min_ratio=0.5
entropy_smooth_max_ratio=1.5
entropy_distribution_output_file="entropy_distribution_${environment}.json"
entropy_coeff=0.001
entropy_smooth_coeff=1.0
entropy_smooth_out_range_penalty=0.1
enable_smooth_weights=False
# Batch size parameters
log_prob_micro_batch_size_per_gpu=32
ppo_micro_batch_size_per_gpu=32
ppo_mini_batch_size=256
# Model loading parameters
# model_path="Qwen/Qwen2.5-3B-Instruct"
# model_load_method="huggingface"  # "huggingface" or "local"
model_path="/research/cbim/vast/cj574/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
model_load_method="local"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment)
            environment="$2"
            shift 2
            ;;
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
        --window_size)
            window_size="$2"
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
        --total_epochs)
            total_epochs="$2"
            shift 2
            ;;
        --entropy_smooth)
            entropy_smooth="$2"
            shift 2
            ;;
        --entropy_smooth_mask_mode)
            entropy_smooth_mask_mode="$2"
            shift 2
            ;;
        --entropy_smooth_min_ratio)
            entropy_smooth_min_ratio="$2"
            shift 2
            ;;
        --entropy_smooth_max_ratio)
            entropy_smooth_max_ratio="$2"
            shift 2
            ;;
        --entropy_distribution_output_file)
            entropy_distribution_output_file="$2"
            shift 2
            ;;
        --entropy_coeff)
            entropy_coeff="$2"
            shift 2
            ;;
        --entropy_smooth_coeff)
            entropy_smooth_coeff="$2"
            shift 2
            ;;
        --entropy_smooth_out_range_penalty)
            entropy_smooth_out_range_penalty="$2"
            shift 2
            ;;
        --enable_smooth_weights)
            enable_smooth_weights="$2"
            shift 2
            ;;
        --smooth_weights)
            enable_smooth_weights="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --model_load_method)
            model_load_method="$2"
            shift 2
            ;;
        --log_prob_micro_batch_size_per_gpu)
            log_prob_micro_batch_size_per_gpu="$2"
            shift 2
            ;;
        --ppo_micro_batch_size_per_gpu)
            ppo_micro_batch_size_per_gpu="$2"
            shift 2
            ;;
        --ppo_mini_batch_size)
            ppo_mini_batch_size="$2"
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

# Set environment-specific variables
case $environment in
    "alfworld")
        env_name="alfworld/AlfredTWEnv"
        project_name_suffix="alfworld"
        max_steps=50
        max_prompt_length=2048
        ;;
    "webshop")
        env_name="Webshop"
        project_name_suffix="webshop"
        max_steps=15
        max_prompt_length=2048
        ;;
    "sciworld")
        env_name="sciworld/SciWorldEnv"
        project_name_suffix="sciworld"
        max_steps=30
        max_prompt_length=2048
        # Set conservative GPU memory utilization for SciWorld to prevent OOM
        gpu_memory_util_override=0.4
        ;;
    *)
        echo "Error: Unknown environment '$environment'. Supported environments: alfworld, webshop, sciworld"
        exit 1
        ;;
esac

# Set algorithm-specific variables and adjust parameters
case $rl_algorithm in
    "ppo")
        adv_estimator="gae"
        # PPO uses critic configurations (same lr and model path as actor)
        use_critic=true
        critic_lr=$lr
        critic_model_path=$model_path
        critic_micro_batch_size_per_gpu=16
        # For PPO: multiply train_data_size by group_size
        train_data_size=$((train_data_size * group_size))
        ;;
    "grpo")
        adv_estimator="grpo"
        use_critic=false
        ;;
    *)
        echo "Error: Unknown RL algorithm '$rl_algorithm'. Supported algorithms: grpo, ppo"
        exit 1
        ;;
esac

# Generate experiment name based on algorithm
# Convert lr to a filename-friendly format (replace 'e-' with 'e' and '.' with 'p')
lr_str=$(echo "$lr" | sed 's/e-/e/g' | sed 's/\./p/g')

if [ "$rl_algorithm" = "grpo" ]; then
    # For GRPO: short experiment name
    experiment_name="grpo_s${seed}_tb${train_data_size}_vb${val_data_size}_gs${group_size}_lr${lr_str}"
elif [ "$rl_algorithm" = "ppo" ]; then
    # For PPO: short experiment name
    experiment_name="ppo_s${seed}_tb${train_data_size}_vb${val_data_size}_gs${group_size}_lr${lr_str}"

fi
# Add entropy coefficient
if [ "$(echo "$entropy_coeff != 0" | bc -l)" -eq 1 ]; then
    experiment_name="${experiment_name}_ec${entropy_coeff}"
else
    experiment_name="${experiment_name}_ec0"
fi

# Add other key parameters only if enabled
if [ "$entropy_smooth" = "True" ]; then
    experiment_name="${experiment_name}_es${entropy_smooth_coeff}"
fi

if [ "$enable_smooth_weights" = "True" ]; then
    experiment_name="${experiment_name}_sw"
fi
echo "Generated experiment name: $experiment_name"

# Data preprocessing
echo "Starting data preprocessing..."
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# Set GPU memory utilization based on environment
if [ "$environment" = "sciworld" ]; then
    gpu_memory_utilization=${gpu_memory_util_override:-0.6}
else
    gpu_memory_utilization=0.6
fi

# Run training with algorithm-specific configurations
if [ "$rl_algorithm" = "ppo" ]; then
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=$adv_estimator \
        data.train_files=$HOME/data/verl-agent/text/train.parquet \
        data.val_files=$HOME/data/verl-agent/text/test.parquet \
        data.train_batch_size=$train_data_size \
        data.val_batch_size=$val_data_size \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$model_path \
        actor_rollout_ref.model.load_method=$model_load_method \
        actor_rollout_ref.actor.optim.lr=$lr \
        actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$lr_warmup_steps_ratio \
        actor_rollout_ref.actor.optim.min_lr_ratio=$min_lr_ratio \
        actor_rollout_ref.actor.optim.num_cycles=$num_cycles \
        actor_rollout_ref.actor.optim.warmup_style=$warmup_style \
        actor_rollout_ref.actor.optim.weight_decay=$weight_decay \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.rollout.max_model_len=4096 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.dtype=half \
        actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        actor_rollout_ref.actor.max_steps=$max_steps \
        actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
        actor_rollout_ref.actor.entropy_smooth_coeff=$entropy_smooth_coeff \
        critic.optim.lr=$critic_lr \
        critic.model.use_remove_padding=True \
        critic.model.path=$critic_model_path \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_mini_batch_size=$train_data_size \
        critic.ppo_micro_batch_size_per_gpu=$(($log_prob_micro_batch_size_per_gpu/2)) \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.use_kl_in_reward=False \
        env.env_name=$env_name \
        env.seed=$seed \
        env.max_steps=$max_steps \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='verl_agent_'$project_name_suffix'_'$rl_algorithm \
        trainer.experiment_name=$experiment_name \
        trainer.n_gpus_per_node=$n_gpus \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=5 \
        trainer.total_epochs=$total_epochs \
        trainer.val_before_train=True
else
    # GRPO
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=$adv_estimator \
        data.train_files=$HOME/data/verl-agent/text/train.parquet \
        data.val_files=$HOME/data/verl-agent/text/test.parquet \
        data.train_batch_size=$train_data_size \
        data.val_batch_size=$val_data_size \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path=$model_path \
        actor_rollout_ref.model.load_method=$model_load_method \
        actor_rollout_ref.actor.optim.lr=$lr \
        actor_rollout_ref.actor.optim.lr_warmup_steps=$lr_warmup_steps \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$lr_warmup_steps_ratio \
        actor_rollout_ref.actor.optim.min_lr_ratio=$min_lr_ratio \
        actor_rollout_ref.actor.optim.num_cycles=$num_cycles \
        actor_rollout_ref.actor.optim.warmup_style=$warmup_style \
        actor_rollout_ref.actor.optim.weight_decay=$weight_decay \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.dtype=half \
        actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
        actor_rollout_ref.rollout.max_model_len=4096 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        actor_rollout_ref.actor.entropy_smooth=$entropy_smooth \
        actor_rollout_ref.actor.entropy_smooth_mask_mode=$entropy_smooth_mask_mode \
        actor_rollout_ref.actor.entropy_smooth_min_ratio=$entropy_smooth_min_ratio \
        actor_rollout_ref.actor.entropy_smooth_max_ratio=$entropy_smooth_max_ratio \
        algorithm.use_kl_in_reward=False \
        env.env_name=$env_name \
        env.seed=$seed \
        env.max_steps=$max_steps \
        env.rollout.n=$group_size \
        env.entropy_threshold=$entropy_threshold \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='verl_agent_'$project_name_suffix'_'$rl_algorithm \
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
        trainer.window_size=$window_size \
        actor_rollout_ref.actor.entropy_distribution_output_file=$entropy_distribution_output_file \
        actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
        actor_rollout_ref.actor.entropy_smooth_coeff=$entropy_smooth_coeff \
        actor_rollout_ref.actor.entropy_smooth_out_range_penalty=$entropy_smooth_out_range_penalty \
        actor_rollout_ref.actor.enable_smooth_weights=$enable_smooth_weights \
        actor_rollout_ref.actor.max_steps=$max_steps
fi