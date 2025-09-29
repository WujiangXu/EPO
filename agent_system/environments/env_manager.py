from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
import copy

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
        
        # Get group information for step_group method
        self.env_num = getattr(self.envs, 'env_num', None)
        self.group_n = getattr(self.envs, 'group_n', None)
        if self.env_num is None or self.group_n is None:
            # Backward compatibility: try to infer from num_processes
            total_processes = getattr(self.envs, 'num_processes', len(getattr(self.envs, 'workers', [])))
            # If unable to infer, default to assuming each environment is a separate group
            if hasattr(self.envs, 'group_n'):
                self.group_n = self.envs.group_n
                self.env_num = total_processes // self.group_n if self.group_n > 0 else total_processes
            else:
                self.env_num = total_processes
                self.group_n = 1
        
        print(f"AlfWorldEnvironmentManager initialized with {self.env_num} groups, {self.group_n} envs per group")
    
    def step_group_mask(self, group_text_actions: List[str], group_mask: List[bool]):
        """
        Extended version of step_group method that supports mask for batch-aware copy action, 
        compatible with step method, no need to constantly switch APIs, just change mask values
        
        Parameters:
        - group_text_actions: Action list
        - group_mask: Boolean mask list with length env_num
            - True: This group doesn't need copying, group_text_actions contains group_n different actions
            - False: This group needs copying, group_text_actions contains only 1 action, copied to all environments in the group
        """
        if len(group_mask) != (self.env_num):
            raise ValueError(
                f"group_mask length ({len(group_mask)}) must equal env_num ({self.env_num})"
            )
        
        text_actions: List[str] = []
        action_idx = 0  # Used to track the index of group_text_actions
        
        for group_id in range(self.env_num):
            if group_mask[group_id]:  # mask is True, no copying needed, each environment in the group uses different actions
                for _ in range(self.group_n):
                    text_actions.append(group_text_actions[action_idx])
                    action_idx += 1
            else:  # mask is False, copying needed, all environments in the group use the same action
                group_action = group_text_actions[action_idx]
                for _ in range(self.group_n):
                    text_actions.append(group_action)
                action_idx += 1
        
        # Call the original step method with the expanded actions
        return self.step(text_actions)

    def step_group(self, group_text_actions: List[str]):
        """
        Group-level step method
        
        Parameters:
        - group_text_actions (List[str]): Group-level action list with length env_num
                                         The i-th action will be distributed to all environments in the i-th group
        
        Returns:
        - Same return format as the step method
        """
        # Parameter length should equal the number of groups (env_num)
        if len(group_text_actions) != self.env_num:
            raise ValueError(
                f"group_text_actions length ({len(group_text_actions)}) must equal env_num ({self.env_num})"
            )

        # Expand group-level actions to individual environment actions.
        # All group_n environments in the i-th group execute group_text_actions[i]
        text_actions: List[str] = []
        for group_id in range(self.env_num):
            group_action = group_text_actions[group_id]
            for _ in range(self.group_n):
                text_actions.append(group_action)
        
        # Call the original step method with the expanded actions
        return self.step(text_actions)
    
    def reset(self,is_group_reset=False):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        if is_group_reset:
            # Extract distinct environment obs from full obs.
            full_text_obs = full_text_obs[::self.group_n]
            text_obs = text_obs[::self.group_n]
            if image_obs is not None:
                image_obs = image_obs[::self.group_n]
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

class SciWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name, config=None):
        self.buffers = None
        self.config = config
        self.plannings = []
        self.meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        super().__init__(envs, projection_f, env_name)

    def reset(self):
        text_obs, infos = self.envs.reset()

        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.plannings = ["No plan."] * len(text_obs)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task_descriptions(infos)

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos], init=True)
        return {'text': full_text_obs, 'anchor': text_obs}, infos

    def step(self, text_actions: List[str]):
        full_output = copy.deepcopy(text_actions)
        meta_think = self.config is not None and self.config.env.sciworld.meta_think if hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') else False
        actions, valids, action_available = self.projection_f(text_actions, meta_think=meta_think, available_actions=self.envs.get_possible_actions)

        plannings = []
        if meta_think:
            for action in text_actions:
                planning = None
                if "<planning>" in action and "</planning>" in action:
                    start_tag = "<planning>"
                    end_tag = "</planning>"
                    start_idx = action.find(start_tag)
                    end_idx = action.find(end_tag)
                    if start_idx != -1 and end_idx != -1:
                        planning = action[start_idx + len(start_tag):end_idx].strip()
                plannings.append(planning)
        else:
            plannings = [None] * len(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)
        self.save_to_history_buffer(self.pre_text_obs, actions, full_output, plannings)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, [info['available_actions'] for info in infos])

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
            info['full_output'] = full_output[i]
            info['action_available'] = to_numpy(action_available[i])
            info['score'] = info.get('score', -1)

        next_observations = {'text': full_text_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task_descriptions(self, infos: List[dict]):
        for info in infos:
            if 'task_description' in info:
                self.tasks.append(info['task_description'])
            else:
                self.tasks.append("Unknown task")

    def build_text_obs(self, text_obs: List[str], available_actions: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if self.meta_think:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS_MC
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE_MC
        else:
            _SCIWORLD_TEMPLATE_NO_HIS = SCIWORLD_TEMPLATE_NO_HIS
            _SCIWORLD_TEMPLATE = SCIWORLD_TEMPLATE

        for i in range(len(text_obs)):
            if init or history_length <= 0:
                obs = _SCIWORLD_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=available_actions[i]
                )
            else:
                all_actions = [record["action"] for record in self.buffers[i]]
                recent_history = self.buffers[i][-history_length:]
                recent_start_index = len(self.buffers[i]) - history_length
                valid_history_length = len(recent_history)
                action_history = ""

                for j in range(recent_start_index):
                    action = all_actions[j]
                    step_number = j + 1
                    action_history += f"\n[Step {step_number}, Action {step_number}: '{action}']"

                for j, record in enumerate(recent_history):
                    step_number = recent_start_index + j + 1
                    env_obs = record["text_obs"]
                    action = record["action"]
                    action_history += f"\n[Step {step_number}, Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"

                if self.config is not None and hasattr(self.config.env, 'sciworld') and hasattr(self.config.env.sciworld, 'meta_think') and self.config.env.sciworld.meta_think:
                    history_think_length = min(3, len(self.buffers[i]))
                    start_index = len(self.buffers[i]) - history_think_length
                    action_history += "\n- recent reasoning process: \n" 
                    for j, record in enumerate(self.buffers[i][-history_think_length:]):
                        step_number = start_index + j + 1
                        action_history += f"[Step {step_number}, output {step_number}: '{record['full_output']}']\n"

                    obs = _SCIWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        planning=self.plannings[i],
                        available_actions=available_actions[i]
                    )
                else:
                    obs = _SCIWORLD_TEMPLATE.format(
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                        available_actions=available_actions[i]
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions, text_actions=None, plannings=None):
        for i in range(len(actions)):
            if text_actions:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i], 'full_output': text_actions[i]})
            else:
                self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})

        if plannings:
            for i in range(len(plannings)):
                if plannings[i] is not None:
                    self.plannings[i] = plannings[i]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return

    def _set_meta_think(self, type: bool):
        self.meta_think = type

class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, env_name):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.buffers = None
        super().__init__(envs, projection_f, env_name)

    def reset(self):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.save_to_history_buffer(self.pre_text_obs, actions)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.save_to_history_buffer(self.pre_text_obs, actions)
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if init or history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    if self.is_multi_modal:
                        action_history += f"\n[Action {step_number}: '{record['action']}']"
                    else:
                        action_history += f"\n[Text Observation {step_number}: \n{record['text_obs']}\nAction {step_number}: '{record['action']}']"

                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': self.ACTION_LOOKUP[actions[i]]})


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        super().__init__(envs, projection_f, env_name)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(infos))]
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False, history_length: int = 2) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Action {step_number}: '{action}']"
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.buffers[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.buffers[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, env_name):
        self.buffers = None
        super().__init__(envs, projection_f, env_name)
    
    def reset(self):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        # initialize the history buffer
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(text_obs))]
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.save_to_history_buffer(self.pre_text_obs, actions)
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False, history_length: int = 20) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.buffers[i][-history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.buffers[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\n[Observation {step_number}: '{env_obs}', Code {step_number}: '{action}']"

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.buffers[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.buffers[i]) + 1,
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def save_to_history_buffer(self, text_obs, actions):
        for i in range(len(actions)):
            self.buffers[i].append({'text_obs': text_obs[i], 'action': actions[i]})


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True)
        _val_envs_iid = build_alfworld_envs(alf_config_path, config.env.seed + 2000, config.data.val_batch_size, 1, is_train=False, eval_mode="eval_in_distribution")
        # keep same setting for ood 
        _val_envs_ood = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, eval_mode="eval_out_of_distribution")
        # ood level 2 from paper "RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents"
        alf_test_config_path = alf_config_path.replace('config_tw.yaml', 'config_tw_test_ood.yaml')
        _val_envs_ood_l2 = build_alfworld_envs(alf_test_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, eval_mode="eval_out_of_distribution")
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs_iid = AlfWorldEnvironmentManager(_val_envs_iid, projection_f, config.env.env_name)
        val_envs_ood = AlfWorldEnvironmentManager(_val_envs_ood, projection_f, config.env.env_name)
        val_envs_ood_l2 = AlfWorldEnvironmentManager(_val_envs_ood_l2, projection_f, config.env.env_name)   
        return envs, [val_envs_iid, val_envs_ood, val_envs_ood_l2]
    elif "sciworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.sciworld import build_sciworld_envs, sciworld_projection
        import json
        variation_path_l0 = os.path.join(os.path.dirname(__file__), 'env_package/sciworld/variations_idx/L0_idx.json')
        variation_path_l1 = os.path.join(os.path.dirname(__file__), 'env_package/sciworld/variations_idx/L1_idx.json')
        variation_path_l2 = os.path.join(os.path.dirname(__file__), 'env_package/sciworld/variations_idx/L2_idx.json')

        with open(variation_path_l0, 'r') as f:
            variations_idx_l0 = json.load(f)
        with open(variation_path_l1, 'r') as f:
            variations_idx_l1 = json.load(f)
        with open(variation_path_l2, 'r') as f:
            variations_idx_l2 = json.load(f)

        simplifications_preset = "easy"
        env_step_limit = 100
        jar_path = None

        _envs = build_sciworld_envs(
            seed=config.env.seed, 
            env_num=config.data.train_batch_size, 
            group_n=group_n, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx_l0['train']
        )

        _val_envs_l0 = build_sciworld_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx_l0['test']
        )

        _val_envs_l1 = build_sciworld_envs(
            seed=config.env.seed + 1000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx_l1['test']
        )

        _val_envs_l2 = build_sciworld_envs(
            seed=config.env.seed + 2000, 
            env_num=config.data.val_batch_size, 
            group_n=1, 
            simplifications_preset=simplifications_preset,
            env_step_limit=env_step_limit,
            jar_path=jar_path,
            variations_idx=variations_idx_l2['test']
        )

        # Create projection function
        projection_f = partial(sciworld_projection)

        # Create environment managers
        envs = SciWorldEnvironmentManager(_envs, projection_f, config.env.env_name, config)
        val_envs_l0 = SciWorldEnvironmentManager(_val_envs_l0, projection_f, config.env.env_name, config)
        val_envs_l1 = SciWorldEnvironmentManager(_val_envs_l1, projection_f, config.env.env_name, config)
        val_envs_l2 = SciWorldEnvironmentManager(_val_envs_l2, projection_f, config.env.env_name, config)

        # Give some time for environments to initialize
        # import time
        # time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1)

        return envs, [val_envs_l0, val_envs_l1, val_envs_l2]
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config.env.env_name)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config.env.env_name)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)


if __name__ == "__main__":
    env_name = "alfworld"
    if env_name == "gym_cards":
        # Test GymCardEnvironmentManager
        env_num = 2
        group_n = 5
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        envs = build_gymcards_envs('gym_cards/EZPoints-v0', 0, env_num, group_n)
        projection_f = partial(gym_projection, env_name='gym_cards/EZPoints-v0')
        env_manager = GymCardEnvironmentManager(envs, projection_f, 'gym_cards/EZPoints-v0')
        obs, infos = env_manager.reset()
        for i in range(100):
            random_actions = [f'"action": {np.random.randint(0, 10)}' for i in range(len(infos))]
            obs, rewards, dones, infos = env_manager.step(random_actions)
            env_manager.save_image(obs['image'], i)
        print("completed")
    elif env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import alfworld_projection
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        import time
        alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        env_num = 2
        group_n = 3
        time1 = time.time()
        envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n)
        # val_envs = build_alfworld_envs(alf_config_path, 1000, 4)
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, 'alfworld/AlfredThorEnv')
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        # val_env_manager = AlfWorldEnvironmentManager(val_envs, alfworld_projection, 'alfworld/AlfredTWEnv')
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset(is_group_reset=True)
            print(f"obs: {obs}")
            print(f"infos: {infos}")
            obs, infos = env_manager.reset(is_group_reset=False)
            print("-"*100)
            print(f"obs: {obs}")
            print(f"infos: {infos}")
            break
            # for i in range(20):
            #     # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
            #     print("step: ", i)
            #     random_actions = [np.random.choice(env_manager.envs.get_admissible_commands[i]) for i in range(len(env_manager.envs.get_admissible_commands))]
            #     # step
            #     # obs, rewards, dones, infos = env_manager.step(random_actions)
            #     # if np.array(dones).any():
            #     #     print("Episode completed")

            #     # for k in range(len(infos)):
            #     #     assert infos[k]['won'] == False
            #     # if obs['image'] is not None:
            #     #     env_manager.save_image(obs['image'], i)

            #     random_action_group = random_actions[:env_num]
            #     obs, rewards, dones, infos = env_manager.step_group(random_action_group)
            #     if np.array(dones).any():
            #         print("Episode completed")

            #     for k in range(len(infos)):
            #         assert infos[k]['won'] == False
            #     if obs['image'] is not None:
            #         env_manager.save_image(obs['image'], i)

            #     # print("obs['image'].shape: ", obs['image'].shape)
            # time2 = time.time()
            # print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
            # print("completed")

    elif env_name == "sokoban":
        # Test SokobanEnvironmentManager
        from agent_system.environments.env_package.sokoban import sokoban_projection
        from agent_system.environments.env_package.sokoban import build_sokoban_envs
        env_num = 2
        group_n = 5
        env_kwargs = {
            'dim_room': (6, 6),
            'num_boxes': 1,
            'max_steps': 100,
            'search_depth': 30
        }
        action_pools = {
            1: "<action>up</action>",
            2: "<action>down</action>",
            3: "<action>left</action>",
            4: "<action>right</action>",
        }
        # ['tiny_rgb_array', 'list', 'state', 'rgb_array']
        envs = build_sokoban_envs(0, env_num, group_n, mode='rgb_array', is_train=True, env_kwargs=env_kwargs)
        projection_f = partial(sokoban_projection)
        env_manager = SokobanEnvironmentManager(envs, projection_f, 'sokoban')
        obs, infos = env_manager.reset()
        for i in range(100):
            random_actions = [action_pools[np.random.randint(1, 5)] for i in range(len(infos))]
            obs, rewards, dones, infos = env_manager.step(random_actions)
            if obs['image'] is not None:
                env_manager.save_image(obs['image'][0], i)
            if np.array(dones).any():
                print("Episode completed")
    elif env_name == "webshop":
        # Test WebshopEnvironmentManager
        from agent_system.environments.env_package.webshop import webshop_projection
        from agent_system.environments.env_package.webshop import build_webshop_envs
        from agent_system.environments.env_package.webshop.webshop.web_agent_site.models import RandomPolicy
        import time
        env_num = 2
        group_n = 5
        time1 = time.time()
        file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
        attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': False,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        envs = build_webshop_envs(seed=1, env_num=env_num, group_n=group_n, env_kwargs=env_kwargs, is_train=True)
        # val_envs = build_webshop_envs(1000, 4)
        env_manager = WebshopEnvironmentManager(envs, webshop_projection, 'webshop')
        policy = RandomPolicy()
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        # val_env_manager = AlfWorldEnvironmentManager(val_envs, alfworld_projection, 'alfworld/AlfredTWEnv')
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset()
            for i in range(20):
                # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
                print("step: ", i)
                random_actions = ['<action>'+policy.forward(None, info['available_actions'])+'</action>' for info in infos]
                # step
                obs, rewards, dones, infos = env_manager.step(random_actions)
                if np.array(dones).any():
                    print("Episode completed")

                if obs['image'] is not None:
                    env_manager.save_image(obs['image'], i)
                # print("obs['image'].shape: ", obs['image'].shape)
            time2 = time.time()
            print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
        print("completed")

    elif env_name == "appworld":
        # Test AppWorldEnvironmentManager
        from agent_system.environments.env_package.appworld import appworld_projection
        from agent_system.environments.env_package.appworld import build_appworld_envs
        import time
        env_num = 2
        group_n = 5
        time1 = time.time()
        envs = build_appworld_envs(dataset_name='test_normal', max_interactions=50, seed=1, env_num=env_num, group_n=group_n)
        # val_envs = build_alfworld_envs(alf_config_path, 1000, 4)
        env_manager = AppWorldEnvironmentManager(envs, appworld_projection, 'appworld')
        time2 = time.time()
        print(f"env_num: {env_num}, group_n: {group_n}, init time: ", time2 - time1)
        # val_env_manager = AlfWorldEnvironmentManager(val_envs, alfworld_projection, 'alfworld/AlfredTWEnv')
        for k in range(10):
            time1 = time.time()
            obs, infos = env_manager.reset()
            for i in range(20):
                # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
                print("step: ", i)
                random_actions = ["print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))" for i in range(len(obs['text']))]
                # print(apis.api_docs.show_api_descriptions(app_name='supervisor'))
                # step
                obs, rewards, dones, infos = env_manager.step(random_actions)
                if np.array(dones).any():
                    print("Episode completed")

                for k in range(len(infos)):
                    assert infos[k]['won'] == False
                if obs['image'] is not None:
                    env_manager.save_image(obs['image'], i)
                # print("obs['image'].shape: ", obs['image'].shape)
            time2 = time.time()
            print(f"env_num: {env_num}, group_n: {group_n}, Time elapsed: ", time2 - time1)
        print("completed")