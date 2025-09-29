import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import sys
import os
import time
import random
from typing import Union
from itertools import product

def compute_reward(info, multi_modal=False):
    reward = 10.0 * float(info['won'])
    return reward

def _worker(remote, seed, task_nums, simplifications_preset, env_step_limit, jar_path, split=None, variations_idx=None):
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    taskNames = env.get_task_names()
    random.seed(seed)
    task_id, task_variation = random.choice(variations_idx)
    prev_score = 0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                observation, reward, done, info = env.step(action)
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()
                info['score'] = info.get('score', 0.0)
                info['task_score'] = info['score']
                isCompleted = done
                prev_score = info['score']
                info["won"] = isCompleted and info["score"] > 0
                reward = compute_reward(info)
                remote.send((observation, reward, isCompleted, info))
            elif cmd == 'reset':
                if data is None:
                    task_id, task_variation = random.choice(variations_idx)
                    task_num = task_id
                    taskName = taskNames[task_num]
                else:
                    variation_idx = data
                simplification_str = simplifications_preset if simplifications_preset else ""
                env.load(taskName, task_variation, simplification_str)
                observation, info = env.reset()
                task_description = env.get_task_description()
                info['task_description'] = task_description
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()
                info['won'] = False
                info['task_num'] = task_num
                prev_score = 0
                remote.send((observation, info))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command sent to worker: {cmd}")
    finally:
        env.close()

class SciWorldMultiProcessEnv(gym.Env):
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        task_nums: list = [1], 
        split: str = "train", 
        simplifications_preset: str = "", 
        env_step_limit: int = 100,
        jar_path: str = None,
        variations_idx: list = None  
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.split = split
        self.task_nums = task_nums
        self.variations_idx = variations_idx
        self.simplifications_preset = simplifications_preset
        self.env_step_limit = env_step_limit
        self.jar_path = jar_path
        random.seed(seed)
        self._rng = np.random.RandomState(seed)
        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        ctx = mp.get_context('spawn')
        for i in range(self.num_processes):
            parent_remote, child_remote = ctx.Pipe()
            seed_i = seed + i
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, seed_i, self.task_nums, self.simplifications_preset, 
                      self.env_step_limit, self.jar_path, self.split, self.variations_idx),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
            self._parent_remotes.append(parent_remote)
            child_remote.close()
        self.prev_available_actions = [[] for _ in range(self.num_processes)]
        self.prev_possible_actions = [[] for _ in range(self.num_processes)]

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )
        for remote, action in zip(self._parent_remotes, actions):
            remote.send(('step', action))
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        variations = [None for _ in range(self.num_processes)]
        for remote, variation in zip(self._parent_remotes, variations):
            remote.send(('reset', variation))
        obs_list, info_list = [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, info_list

    @property
    def get_available_actions(self):
        return self.prev_available_actions

    @property
    def get_admissible_commands(self):
        return self.prev_available_actions

    @property
    def get_possible_actions(self):
        return self.prev_possible_actions

    def close(self):
        if getattr(self, '_closed', False):
            return
        for remote in self._parent_remotes:
            remote.send(('close', None))
        for worker in self._workers:
            worker.join()
        self._closed = True

    def __del__(self):
        self.close()

def build_sciworld_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    task_nums: Union[int, list] = 1, 
    split: str = "train", 
    simplifications_preset: str = "",
    env_step_limit: int = 100,
    jar_path: str = None,
    variations_idx: list = None
):
    return SciWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        task_nums=task_nums,
        split=split,
        simplifications_preset=simplifications_preset,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
        variations_idx=variations_idx
    ) 
