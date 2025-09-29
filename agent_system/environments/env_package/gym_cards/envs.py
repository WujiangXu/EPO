import gymnasium as gym
import torch.multiprocessing as mp
import numpy as np
from gym_cards.envs import Point24Env, EZPointEnv, BlackjackEnv, NumberLineEnv
import sys

def _worker(remote, env_id):
    """
    Core loop of the subprocess.
    Each subprocess holds an independent instance of gym.make(env_id),
    and communicates directly with the main process via Pipe instead of using SubprocVecEnv.
    """
    if env_id == 'gym_cards/Points24-v0':
        env = Point24Env()
    elif env_id == 'gym_cards/EZPoints-v0':
        env = EZPointEnv()
    elif env_id == 'gym_cards/Blackjack-v0':
        env = BlackjackEnv()
    elif env_id == 'gym_cards/NumberLine-v0':
        env = NumberLineEnv()
    else:
        raise NotImplementedError(f"Unknown env_id: {env_id}")

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            action = data
            obs, reward, done, _, info = env.step(action)
            remote.send((obs, reward, done, info))

        elif cmd == 'reset':
            seed_for_reset = data
            if seed_for_reset is not None:
                obs, info = env.reset(seed=seed_for_reset)
            else:
                obs, info = env.reset()

            remote.send((obs, info))

        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class GymMultiProcessEnv(gym.Env):
    """
    Custom multiprocessing parallel environment, similar to the first example.
    - env_id: Gym environment ID
    - env_num: Number of distinct environments
    - group_n: Number of replicas within each group (commonly used for multiple copies with the same seed)
    - env_kwargs: Parameters needed to create a single gym.make(env_id)
    """

    def __init__(self,
                 env_id,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 is_train=True):
        super().__init__()

        self.env_id = env_id
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n

        np.random.seed(seed)

        self.parent_remotes = []
        self.workers = []
        if sys.platform.startswith("win"):
            ctx = mp.get_context('spawn')
        else:
            ctx = mp.get_context('fork')

        for _ in range(self.num_processes):
            parent_remote, child_remote = mp.Pipe()
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, self.env_id)
            )
            worker.daemon = True
            worker.start()
            child_remote.close()

            self.parent_remotes.append(parent_remote)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list or numpy array, length must equal self.num_processes.
        :return: obs_list, reward_list, done_list, info_list
        """
        assert len(actions) == self.num_processes

        for remote, action in zip(self.parent_remotes, actions):
            remote.send(('step', action))

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for remote in self.parent_remotes:
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        
        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        Different seeds will be assigned to each environment (or the same seed within a group).
        :return: (obs_list, info_list)
        """
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # Repeat seed for environments in the same group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        for i, remote in enumerate(self.parent_remotes):
            remote.send(('reset', seeds[i]))

        obs_list, info_list = [], []
        for remote in self.parent_remotes:
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)

        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)
        return obs_list, info_list

    def close(self):
        """
        Close all subprocesses.
        """
        for remote in self.parent_remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()

    def __del__(self):
        self.close()


def build_gymcards_envs(env_name,
                        seed,
                        env_num,
                        group_n,
                        is_train=True):
    """
    Externally exposed constructor function to create parallel Gym environments.
    - env_name: [gym_cards/Blackjack-v0, gym_cards/NumberLine-v0, gym_cards/EZPoints-v0, gym_cards/Points24-v0]
    - seed: For reproducible randomness
    - env_num: Number of distinct environments
    - group_n: Number of environment replicas under the same seed
    - is_train: Determines the seed range used (train/test)
    """
    return GymMultiProcessEnv(
        env_id=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )
