import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from tqdm import trange
from collections import deque
import os
import json
from datetime import datetime

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor        

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        max_prompt_length = getattr(self.config.data, 'max_prompt_length', 2048)
        safe_max_length = min(max_prompt_length, 2048 - 512)  # Reserve 512 tokens for generation
        
        # Pre-check: if prompt is extremely long, do aggressive truncation first
        prompt_tokens = len(self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False))
        if prompt_tokens > safe_max_length:
            print(f"Warning: Prompt length {prompt_tokens} exceeds safe limit {safe_max_length}, applying aggressive truncation")
            # Aggressive truncation for very long prompts
            safe_max_length = min(safe_max_length, 1536)  # Be even more conservative
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=safe_max_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.get('truncation', 'left'))
        
        

        if is_multi_modal:

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        print(f"batch_size: {batch_size}")
        print(f"total_batch_list lengths: {[len(lst) for lst in total_batch_list]}")
        
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for step_idx, data in enumerate(total_batch_list[bs]):
                # assert traj_uid[bs] == data['traj_uid'], f"data is not from the same trajectory at batch {bs}, step {step_idx}. Expected: {traj_uid[bs]}, Got: {data['traj_uid']}"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
        
        print(f"gather_rollout_data: effective_batch length: {len(effective_batch)}")
        if effective_batch:
            # Debug anchor_obs field
            sample_data = effective_batch[0]
            print(f"Sample data keys: {list(sample_data.keys())}")
            if 'anchor_obs' in sample_data:
                print(f"Sample anchor_obs type: {type(sample_data['anchor_obs'])}")
                if hasattr(sample_data['anchor_obs'], 'shape'):
                    print(f"Sample anchor_obs shape: {sample_data['anchor_obs'].shape}")
                elif hasattr(sample_data['anchor_obs'], '__len__'):
                    print(f"Sample anchor_obs length: {len(sample_data['anchor_obs'])}")
            
            # Check anchor_obs field for all data
            anchor_obs_lengths = []
            for i, data in enumerate(effective_batch):
                if 'anchor_obs' in data:
                    if hasattr(data['anchor_obs'], 'shape'):
                        anchor_obs_lengths.append(data['anchor_obs'].shape)
                    elif hasattr(data['anchor_obs'], '__len__'):
                        anchor_obs_lengths.append(len(data['anchor_obs']))
                    else:
                        anchor_obs_lengths.append("no_length_attr")
            print(f"All anchor_obs lengths/shapes: {anchor_obs_lengths[:10]}...")  # Print only first 10
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            return_entropy: bool = False,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment  
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        print("\n=== Starting vanilla_multi_turn_loop ===")
        
        obs, infos = envs.reset()
        print(f"Initial obs keys: {obs.keys()}")
        if obs.get('text') is not None:
            print(f"obs['text'] type: {type(obs['text'])}, length: {len(obs['text']) if obs['text'] else 'None'}")
        if obs.get('image') is not None:
            print(f"obs['image'] shape: {obs['image'].shape if hasattr(obs['image'], 'shape') else 'No shape attr'}")
        print(f"Initial infos length: {len(infos)}")

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        print(f"Environment observation length: {lenght_obs}")
        print(f"Original gen_batch size: {len(gen_batch.batch['input_ids']) if 'input_ids' in gen_batch.batch else 'No input_ids'}")
        
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            print(f"Repeating gen_batch {self.config.env.rollout.n} times for environment grouping")
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids'])
        print(f"Final batch_size: {batch_size}")
        batch_output = None
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
            print(f"Environment grouping enabled, group size: {self.config.env.rollout.n}")
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
            print("No environment grouping, all environments share same uid")
        
        print(f"uid_batch shape: {uid_batch.shape}")
        
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]  # Trajectory data list for each environment
        total_infos = [[] for _ in range(batch_size)]        # Info list for each environment
        episode_lengths = np.zeros(batch_size, dtype=np.int32)   # Episode length for each environment
        episode_rewards = np.zeros(batch_size, dtype=np.float32) # Cumulative reward for each environment
        
        step_entropy_list = []
        sample_entropy_mean = np.zeros(batch_size, dtype=np.float32)  # Record average entropy for each sample
        sample_entropy_count = np.zeros(batch_size, dtype=np.int32)   # Record entropy count for each sample
        max_steps = self.config.env.max_steps
        for _step in range(self.config.env.max_steps):
            
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            print("Generating responses with actor model...")
            batch_output = actor_rollout_wg.generate_sequences(batch_input)

            if return_entropy:
                
                entropy_batch_output = batch_output.select(deepcopy=True)
                entropy_batch_output.meta_info["calculate_entropy"] = True
                logprob_out = actor_rollout_wg.compute_log_prob(entropy_batch_output)
                entropys = logprob_out.batch["entropys"]
                response_length = batch_output.batch["responses"].size(1)
                attention_mask = batch_output.batch["attention_mask"]
                response_mask = attention_mask[:, -response_length:]
                ent_mean = (entropys * response_mask).sum() / response_mask.sum()
                ent_mean = ent_mean.item()
                if len(step_entropy_list) < self.config.env.max_steps:
                    step_entropy_list.append(ent_mean)
                
                sample_entropies = (entropys * response_mask).sum(dim=1) / response_mask.sum(dim=1)  # (batch_size,)
                
                for i in range(batch_size):
                    if active_masks[i]:  # Only record entropy for active samples
                        sample_entropy_count[i] += 1
                        # Calculate running average: new_avg = old_avg + (new_value - old_avg) / count
                        sample_entropy_mean[i] += (sample_entropies[i].item() - sample_entropy_mean[i]) / sample_entropy_count[i]
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            
            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            next_obs, rewards, dones, infos = envs.step(text_actions)
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            reward_increment = torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards += reward_increment
            episode_lengths[active_masks] += 1
            
            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            batch.non_tensor_batch['rollout_step'] = np.full(batch_size, _step, dtype=np.int32)
            
            
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            new_done_count = np.sum(np.logical_and(np.logical_not(is_done), dones))
            is_done = np.logical_or(is_done, dones)
                
            obs = next_obs

            if is_done.all():
                print("All environments finished, breaking loop early")
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        print("=== vanilla_multi_turn_loop completed ===\n")
        
        if return_entropy:
            return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, step_entropy_list
        else:
            return total_batch_list, episode_rewards, episode_lengths, success, traj_uid


    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            epoch: int = 0,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).
        """
        # Initial observations from the environment
        total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, step_entropy_list = \
            self.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            return_entropy=True,
        )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        
        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )
        # for key in gen_batch_output.batch.keys():
        return gen_batch_output, step_entropy_list