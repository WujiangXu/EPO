import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms as T
import sys

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

def worker_func(remote, config, seed, base_env):
    """
    Core loop of the subprocess:
    1. Create the actual environment object here and keep it in this process.
    2. Continuously receive commands (cmd, data) from the pipe using remote.recv(),
       then perform the corresponding env operation.
    3. Send the result back to the main process using remote.send(...).
    """

    env = base_env.init_env(batch_size=1) # Each worker holds only one sub-environment
    env.seed(seed) 

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            actions = [data] 
            
            obs, scores, dones, infos = env.step(actions)
            infos['observation_text'] = obs
            remote.send((obs, scores, dones, infos))

        elif cmd == 'reset':
            obs, infos = env.reset()
            infos['observation_text'] = obs
            remote.send((obs, infos))

        elif cmd == 'getobs':
            image = get_obs_image(env)
            image = image.cpu()  
            remote.send(image)

        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError("Unknown command: {}".format(cmd))

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, is_train=True):
        super().__init__()
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else 'eval_out_of_distribution')
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        self.parent_remotes = []
        self.workers = []

        if sys.platform.startswith("win"):
            ctx = mp.get_context('spawn')
        else:
            ctx = mp.get_context('fork')

        for i in range(self.num_processes):
            parent_remote, child_remote = mp.Pipe()
            worker = ctx.Process(
                target=worker_func,
                args=(child_remote, config, seed + (i // self.group_n), base_env)
            )
            worker.daemon = True
            worker.start()

            child_remote.close()

            self.parent_remotes.append(parent_remote)
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        for i, remote in enumerate(self.parent_remotes):
            remote.send(('step', actions[i]))

        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        for i, remote in enumerate(self.parent_remotes):
            obs, scores, dones, info = remote.recv()
            print(f"info: {info}")
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all subprocesses at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        for remote in self.parent_remotes:
            remote.send(('reset', None))

        for i, remote in enumerate(self.parent_remotes):
            obs, info = remote.recv()
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each subprocess to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        images = []
        for remote in self.parent_remotes:
            remote.send(('getobs', None))

        for remote in self.parent_remotes:
            img = remote.recv()
            images.append(img)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all subprocesses
        """
        for remote in self.parent_remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, is_train)