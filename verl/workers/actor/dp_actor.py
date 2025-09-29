# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple
import json
from collections import defaultdict

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty, compute_policy_loss_with_entropy_mask
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, ulysses_pad
from verl.workers.actor import BasePPOActor
import numpy as np

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()


    def _calculate_epoch_based_entropy_weight(self, current_epoch: int, max_epochs: int) -> float:
        """
        Calculate epoch-based entropy weight that decreases exponentially with training progress.
        - From epoch 0 to max_epochs//2: weight decreases slowly from 1.0 to 0.8 (exponential)
        - From max_epochs//2 to max_epochs: weight decreases quickly from 0.8 to 0.0 (exponential)
        
        Args:
            current_epoch: Current training epoch
            max_epochs: Maximum number of training epochs
            
        Returns:
            weight: Float weight value between 0.0 and 1.0
        """
        import math
        
        if max_epochs <= 0:
            return 1.0
            
        # Clamp current_epoch to valid range
        current_epoch = max(0, min(current_epoch, max_epochs))
        
        half_epochs = max_epochs // 2
        
        if current_epoch <= half_epochs:
            # First half: slow exponential decrease from 1.0 to 0.8
            if half_epochs == 0:
                return 1.0
            progress = current_epoch / half_epochs  # 0.0 to 1.0
            # Use exponential decay: weight = 1.0 - 0.2 * (1 - exp(-lambda * progress))
            # This gives smooth exponential transition from 1.0 to 0.8
            lambda_slow = 2.0  # Controls how slow the decay is in first half
            weight = 1.0 - 0.2 * (1 - math.exp(-lambda_slow * progress))
        else:
            # Second half: quick exponential decrease from 0.8 to 0.0
            remaining_epochs = max_epochs - half_epochs
            if remaining_epochs == 0:
                return 0.8
            progress = (current_epoch - half_epochs) / remaining_epochs  # 0.0 to 1.0
            # Use exponential decay: weight = 0.8 * exp(-lambda * progress)
            lambda_fast = 3.0  # Controls how fast the decay is in second half
            weight = 0.8 * math.exp(-lambda_fast * progress)
            
        return weight


    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    def generate_entropy_penalty(self, entropy_history: np.array, entropy: torch.Tensor, rollout_step: np.array, mask_mode: str, min_ratio: float = 0.8, max_ratio: float = 1.2, out_range_penalty: float = 0.1):
        """
        Generate entropy mask based on entropy history and entropy.
        First, find the entropy history of the current rollout_step.
        Then, generate the mask based on the entropy history and entropy judging the flucation of entropy.

        Args:
            entropy_history: np.array, shape (Step_length,) - average entropy of each step
            entropy: torch.Tensor, shape (batch_size, response_length) - current entropy values
            rollout_step: np.array, shape (batch_size,) - which step each sample comes from
            mask_mode: str, "token" or "seq"
            min_ratio: float, minimum ratio for entropy range (default: 0.8)
            max_ratio: float, maximum ratio for entropy range (default: 1.2)
        Returns:
            entropy_mask: torch.Tensor, shape depends on mask_mode
                - if "token": shape (batch_size, response_length)
                - if "seq": shape (batch_size,)
        """
        if len(entropy_history) == 0:
            # If no history, return all ones (no masking)
            if mask_mode == "token":
                return torch.ones_like(entropy)
            else:  # "seq"
                return torch.ones(entropy.size(0), device=entropy.device)
        
        # Get average entropy for each sample's rollout step
        # entropy_history[rollout_step] gives the historical average entropy for each sample's step
        rollout_step_int = rollout_step.astype(np.int32) if hasattr(rollout_step, 'astype') else np.array(rollout_step, dtype=np.int32)
        avg_entropy_per_sample = entropy_history[rollout_step_int]  # shape: (batch_size,)
        avg_entropy_per_sample = torch.tensor(avg_entropy_per_sample, device=entropy.device, dtype=entropy.dtype)
        
        if mask_mode == "token":
            # Calculate token-level average entropy for each sample
            token_avg_entropy = entropy
            avg_entropy_per_sample = avg_entropy_per_sample.unsqueeze(-1).expand_as(token_avg_entropy)
            # Check if current entropy is within acceptable range
            # avg * min_ratio < entropy < avg * max_ratio -> mask = 1.0, else -> mask = 0.1
            lower_bound = avg_entropy_per_sample * min_ratio
            upper_bound = avg_entropy_per_sample * max_ratio
            
            # Create boolean mask for samples within range
            within_range = (token_avg_entropy > lower_bound) & (token_avg_entropy < upper_bound)
            
            # Generate sample-level mask: 1.0 if within range, out_range_penalty otherwise
            sample_mask = torch.where(within_range, 0, out_range_penalty)  # shape: (batch_size,)
            
            # Debug info
            num_within_range = within_range.sum().item()
            total_tokens = within_range.numel()
            print(f"Token mode: {num_within_range}/{total_tokens} tokens, portion:{num_within_range/total_tokens:.2f} get mask=1.0 (within entropy range)")
            
            # Expand to token level
            entropy_mask = sample_mask  # shape: (batch_size, response_length)
            entropy_mask_ratio = num_within_range / total_tokens
        elif mask_mode == "seq":
            # Calculate sequence-level average entropy for each sample
            seq_avg_entropy = entropy.mean(dim=-1)  # shape: (batch_size,)
            
            # Check if current entropy is within acceptable range
            lower_bound = avg_entropy_per_sample * min_ratio
            upper_bound = avg_entropy_per_sample * max_ratio
            print(f"lower_bound: {lower_bound}, upper_bound: {upper_bound}")
            print(f"seq_avg_entropy: {seq_avg_entropy}")
            # Create boolean mask for samples within range
            within_range = (seq_avg_entropy > lower_bound) & (seq_avg_entropy < upper_bound)
            
            # Generate mask: 1.0 if within range, out_range_penalty otherwise
            entropy_mask = torch.where(within_range, 0, out_range_penalty)  # shape: (batch_size,)
            
            # Expand to token level for compatibility with policy loss computation
            entropy_mask = entropy_mask.unsqueeze(-1).expand_as(entropy)  # shape: (batch_size, response_length)
            
            # Debug info - count only samples that get mask value 1.0
            num_within_range = within_range.sum().item()
            total_samples = within_range.numel()
            print(f"Seq mode: {num_within_range}/{total_samples} samples, portion:{num_within_range/total_samples:.2f} get mask=1.0 (within entropy range)")
            entropy_mask_ratio = num_within_range / total_samples
            
        else:
            raise ValueError(f"Invalid mask_mode: {mask_mode}. Must be 'token' or 'seq'")
        
        return entropy_mask, entropy_mask_ratio
    
    def generate_entropy_mask(self, entropy_history: np.array, entropy: torch.Tensor, rollout_step: np.array, mask_mode: str, min_ratio: float = 0.8, max_ratio: float = 1.2, out_range_penalty: float = 0.1):
        """
        Generate entropy mask based on entropy history and entropy.
        First, find the entropy history of the current rollout_step.
        Then, generate the mask based on the entropy history and entropy judging the flucation of entropy.

        Args:
            entropy_history: np.array, shape (Step_length,) - average entropy of each step
            entropy: torch.Tensor, shape (batch_size, response_length) - current entropy values
            rollout_step: np.array, shape (batch_size,) - which step each sample comes from
            mask_mode: str, "token" or "seq"
            min_ratio: float, minimum ratio for entropy range (default: 0.8)
            max_ratio: float, maximum ratio for entropy range (default: 1.2)
        Returns:
            entropy_mask: torch.Tensor, shape depends on mask_mode
                - if "token": shape (batch_size, response_length)
                - if "seq": shape (batch_size,)
        """
        if len(entropy_history) == 0:
            # If no history, return all ones (no masking)
            if mask_mode == "token":
                return torch.ones_like(entropy)
            else:  # "seq"
                return torch.ones(entropy.size(0), device=entropy.device)
        
        # Get average entropy for each sample's rollout step
        # entropy_history[rollout_step] gives the historical average entropy for each sample's step
        rollout_step_int = rollout_step.astype(np.int32) if hasattr(rollout_step, 'astype') else np.array(rollout_step, dtype=np.int32)
        avg_entropy_per_sample = entropy_history[rollout_step_int]  # shape: (batch_size,)
        avg_entropy_per_sample = torch.tensor(avg_entropy_per_sample, device=entropy.device, dtype=entropy.dtype)
        
        if mask_mode == "token":
            # Calculate token-level average entropy for each sample
            token_avg_entropy = entropy
            avg_entropy_per_sample = avg_entropy_per_sample.unsqueeze(-1).expand_as(token_avg_entropy)
            # Check if current entropy is within acceptable range
            # avg * min_ratio < entropy < avg * max_ratio -> mask = 1.0, else -> mask = 0.1
            lower_bound = avg_entropy_per_sample * min_ratio
            upper_bound = avg_entropy_per_sample * max_ratio
            
            # Create boolean mask for samples within range
            within_range = (token_avg_entropy > lower_bound) & (token_avg_entropy < upper_bound)
            
            # Generate sample-level mask: 1.0 if within range, out_range_penalty otherwise
            sample_mask = torch.where(within_range, 1.0, out_range_penalty)  # shape: (batch_size,)
            
            # Debug info
            num_within_range = within_range.sum().item()
            total_tokens = within_range.numel()
            print(f"Token mode: {num_within_range}/{total_tokens} tokens, portion:{num_within_range/total_tokens:.2f} get mask=1.0 (within entropy range)")
            
            # Expand to token level
            entropy_mask = sample_mask  # shape: (batch_size, response_length)
            entropy_mask_ratio = num_within_range / total_tokens
        elif mask_mode == "seq":
            # Calculate sequence-level average entropy for each sample
            seq_avg_entropy = entropy.mean(dim=-1)  # shape: (batch_size,)
            
            # Check if current entropy is within acceptable range
            lower_bound = avg_entropy_per_sample * min_ratio
            upper_bound = avg_entropy_per_sample * max_ratio
            print(f"lower_bound: {lower_bound}, upper_bound: {upper_bound}")
            print(f"seq_avg_entropy: {seq_avg_entropy}")
            # Create boolean mask for samples within range
            within_range = (seq_avg_entropy > lower_bound) & (seq_avg_entropy < upper_bound)
            
            # Generate mask: 1.0 if within range, out_range_penalty otherwise
            entropy_mask = torch.where(within_range, 1.0, out_range_penalty)  # shape: (batch_size,)
            
            # Expand to token level for compatibility with policy loss computation
            entropy_mask = entropy_mask.unsqueeze(-1).expand_as(entropy)  # shape: (batch_size, response_length)
            
            # Debug info - count only samples that get mask value 1.0
            num_within_range = within_range.sum().item()
            total_samples = within_range.numel()
            print(f"Seq mode: {num_within_range}/{total_samples} samples, portion:{num_within_range/total_samples:.2f} get mask=1.0 (within entropy range)")
            entropy_mask_ratio = num_within_range / total_samples
            
        else:
            raise ValueError(f"Invalid mask_mode: {mask_mode}. Must be 'token' or 'seq'")
        
        return entropy_mask, entropy_mask_ratio

    def collect_token_entropy_distribution(self, entropy: torch.Tensor, responses: torch.Tensor, output_file="entropy_distribution.json"):
        """
        Collect token-level entropy distribution with token IDs.
        
        Args:
            entropy: torch.Tensor, shape (batch_size, response_length) - entropy values per token
            responses: torch.Tensor, shape (batch_size, response_length) - token IDs
            output_file: str, path to save the distribution data
        """
        print(f"Collecting entropy distribution from tensors:")
        print(f"  Entropy shape: {entropy.shape}, dtype: {entropy.dtype}")
        print(f"  Responses shape: {responses.shape}, dtype: {responses.dtype}")
        
        # Initialize collection dictionary
        token_entropy_data = {
            'entropy_values': [],
            'token_ids': [],
            'metadata': {
                'entropy_tensor_shape': list(entropy.shape),
                'responses_tensor_shape': list(responses.shape),
                'entropy_dtype': str(entropy.dtype),
                'responses_dtype': str(responses.dtype)
            },
            'statistics': {}
        }
        
        # Convert to CPU and numpy for processing
        entropy_np = entropy.detach().cpu().numpy()
        responses_np = responses.detach().cpu().numpy()
        
        # Basic validation
        assert entropy_np.shape == responses_np.shape, f"Shape mismatch: entropy {entropy_np.shape} vs responses {responses_np.shape}"
        
        print(f"  Token ID range in responses: [{responses_np.min()}, {responses_np.max()}]")
        print(f"  Entropy value range: [{entropy_np.min():.4f}, {entropy_np.max():.4f}]")
        
        # Count different token types for validation
        unique_tokens = set()
        padding_count = 0
        
        # Collect all entropy values and corresponding tokens
        for batch_idx in range(entropy_np.shape[0]):
            for token_idx in range(entropy_np.shape[1]):
                entropy_val = entropy_np[batch_idx, token_idx]
                token_id = responses_np[batch_idx, token_idx]
                
                # Track unique tokens
                unique_tokens.add(int(token_id))
                
                # Skip padding tokens (assuming 0 is padding, adjust if needed)
                if token_id == 0:
                    padding_count += 1
                    continue
                    
                token_entropy_data['entropy_values'].append(float(entropy_val))
                token_entropy_data['token_ids'].append(int(token_id))
        
        print(f"  Found {len(unique_tokens)} unique token IDs")
        print(f"  Skipped {padding_count} padding tokens (ID=0)")
        print(f"  Collected {len(token_entropy_data['token_ids'])} valid token-entropy pairs")
        
        # Add validation info to metadata
        token_entropy_data['metadata']['unique_token_count'] = len(unique_tokens)
        token_entropy_data['metadata']['padding_token_count'] = padding_count
        token_entropy_data['metadata']['collected_pairs'] = len(token_entropy_data['token_ids'])
        token_entropy_data['metadata']['sample_token_ids'] = list(sorted(unique_tokens))[:20]  # Sample of token IDs
        
        # Compute statistics
        entropy_values = np.array(token_entropy_data['entropy_values'])
        if len(entropy_values) > 0:
            token_entropy_data['statistics'] = {
                'total_tokens': len(entropy_values),
                'mean_entropy': float(np.mean(entropy_values)),
                'std_entropy': float(np.std(entropy_values)),
                'min_entropy': float(np.min(entropy_values)),
                'max_entropy': float(np.max(entropy_values)),
                'percentiles': {
                    '25th': float(np.percentile(entropy_values, 25)),
                    '50th': float(np.percentile(entropy_values, 50)),
                    '75th': float(np.percentile(entropy_values, 75)),
                    '90th': float(np.percentile(entropy_values, 90)),
                    '95th': float(np.percentile(entropy_values, 95))
                }
            }
            
            # Print summary
            print(f"\n=== Token-level Entropy Distribution ===")
            print(f"Total tokens analyzed: {token_entropy_data['statistics']['total_tokens']}")
            print(f"Mean entropy: {token_entropy_data['statistics']['mean_entropy']:.4f}")
            print(f"Std entropy: {token_entropy_data['statistics']['std_entropy']:.4f}")
            print(f"Min entropy: {token_entropy_data['statistics']['min_entropy']:.4f}")
            print(f"Max entropy: {token_entropy_data['statistics']['max_entropy']:.4f}")
            print(f"Median entropy: {token_entropy_data['statistics']['percentiles']['50th']:.4f}")
            print(f"95th percentile: {token_entropy_data['statistics']['percentiles']['95th']:.4f}")
            
            # Show some examples of high/low entropy tokens with token IDs
            sorted_indices = np.argsort(entropy_values)
            
            print(f"\n--- Lowest Entropy Tokens (most certain) ---")
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[i]
                print(f"Entropy: {entropy_values[idx]:.4f}, Token ID: {token_entropy_data['token_ids'][idx]}")
            
            print(f"\n--- Highest Entropy Tokens (most uncertain) ---")
            for i in range(min(10, len(sorted_indices))):
                idx = sorted_indices[-(i+1)]
                print(f"Entropy: {entropy_values[idx]:.4f}, Token ID: {token_entropy_data['token_ids'][idx]}")
        else:
            print("No valid tokens found for entropy analysis")
            token_entropy_data['statistics'] = {
                'total_tokens': 0,
                'mean_entropy': 0.0,
                'std_entropy': 0.0,
                'min_entropy': 0.0,
                'max_entropy': 0.0,
                'percentiles': {}
            }
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(token_entropy_data, f, indent=2, ensure_ascii=False)
            print(f"\nEntropy distribution data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving entropy distribution: {e}")
        
        return token_entropy_data


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> tuple:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            tuple: (log_probs, entropys)
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []

        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        trainer_epoch = data.meta_info.get("epoch", 0)
        entropy_history = np.array(data.meta_info.get("entropy_history", []))  # Get entropy history from trainer
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        rollout_step = data.non_tensor_batch["rollout_step"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        # Split rollout_step_step simultaneously to match dataloader
        rollout_step_loader = None
        if rollout_step is not None:
            rollout_step_loader = np.array_split(rollout_step, len(dataloader))

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            # Create joint iterator to iterate over data and rollout_step simultaneously
            if rollout_step_loader is not None:
                batch_iterator = zip(dataloader, rollout_step_loader)
            else:
                batch_iterator = zip(dataloader, [None] * len(dataloader))
            
            for batch_idx, (data, rollout_step_batch) in enumerate(batch_iterator):
                # print(f"Processing batch {batch_idx}: rollout_step_batch shape = {rollout_step_batch.shape if rollout_step_batch is not None else None}")
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # Split rollout_step_batch into micro_batches
                rollout_step_micro_batches = None
                if rollout_step_batch is not None:
                    rollout_step_micro_batches = np.array_split(rollout_step_batch, len(micro_batches))

                self.actor_optimizer.zero_grad()

                for micro_idx, data in enumerate(micro_batches):
                    # Get corresponding rollout_step
                    current_micro_rollout_step = rollout_step_micro_batches[micro_idx] if rollout_step_micro_batches is not None else None
                    
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    entropy_smooth_coeff = self.config.entropy_smooth_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    # for key in data.keys():
                    #     print(f"data[{key}]: {data[key].shape}")
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    
                    # Collect entropy distribution periodically (every 10 epochs to avoid too much logging)
                    # if calculate_entropy and trainer_epoch % 10 == 0 and micro_idx == 0 and batch_idx == 0 and epoch == 0:
                    #     print(f"\n=== Collecting entropy distribution at epoch {trainer_epoch} ===")
                    #     # Get output filename from config, with fallback to default
                    #     base_filename = self.config.get("entropy_distribution_output_file", "entropy_distribution.json")
                    #     # Add epoch suffix to avoid overwriting
                    #     name_parts = base_filename.rsplit('.', 1)
                    #     if len(name_parts) == 2:
                    #         output_file = f"{name_parts[0]}_epoch_{trainer_epoch}.{name_parts[1]}"
                    #     else:
                    #         output_file = f"{base_filename}_epoch_{trainer_epoch}"
                    #     self.collect_token_entropy_distribution(entropy, responses, output_file)
                    
                    if self.config.entropy_smooth:
                        # Get max_steps from config, default to 50 if not specified
                        max_steps = self.config.get("max_step", 50)
                        
                        # Check if we should apply entropy smooth based on current step
                        should_apply_entropy_smooth = False
                        if current_micro_rollout_step is not None:
                            # Check if any step in the batch is >= max_steps // 2
                            should_apply_entropy_smooth = np.any(current_micro_rollout_step >= max_steps // 2)
                            print(f"Step check: max_steps={max_steps}, threshold={max_steps // 2}")
                            print(f"Current steps: {current_micro_rollout_step}")
                            print(f"Should apply entropy smooth: {should_apply_entropy_smooth}")
                        
                        if should_apply_entropy_smooth:
                            min_ratio = self.config.get("entropy_smooth_min_ratio", 0.8)
                            max_ratio = self.config.get("entropy_smooth_max_ratio", 1.2)
                            out_range_penalty = self.config.get("entropy_smooth_out_range_penalty", 0.1)
                            
                            entropy_mask, entropy_mask_ratio = self.generate_entropy_penalty(entropy_history, entropy, current_micro_rollout_step, mask_mode=self.config.entropy_smooth_mask_mode, min_ratio=min_ratio, max_ratio=max_ratio, out_range_penalty=out_range_penalty)


                            if self.config.enable_smooth_weights:
                                # Calculate epoch-based weight for entropy smooth
                                max_epochs = 150
                                epoch_weight = self._calculate_epoch_based_entropy_weight(trainer_epoch, max_epochs)                            
                                # Apply epoch-based weight to entropy mask
                                entropy_mask = entropy_mask * epoch_weight
                            
                            metrics["actor/entropy_mask_ratio"] = entropy_mask_ratio
                            # metrics["actor/epoch_entropy_weight"] = epoch_weight
                        else:
                            # Don't apply entropy smooth, set mask to zeros (no penalty)
                            entropy_mask = torch.zeros_like(entropy)
                            entropy_mask_ratio = 0.0
                            metrics["actor/entropy_mask_ratio"] = entropy_mask_ratio
                            print(f"Entropy smooth disabled: step < {max_steps // 2}, mask set to 0")
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        # print(f"entropy_loss after agg_loss: {entropy_loss}")
                        if self.config.entropy_smooth:
                            entropy_penalty_loss = agg_loss(loss_mat=entropy_mask, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            entropy_loss = entropy_loss - entropy_penalty_loss * entropy_smooth_coeff
                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    print("--------------------------------update policy--------------------------------")
                    # print(f"entropy: {entropy.shape}")
                    # print(f"entropy_loss: {entropy_loss}")
                    print(f"policy_loss: {policy_loss}")
                    print(f"loss: {loss}")

                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
