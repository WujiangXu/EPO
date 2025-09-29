import os
import numpy as np
import torch.multiprocessing as mp
import sys

from appworld import AppWorld, load_task_ids, update_root

update_root(os.path.join(os.path.dirname(__file__), "appworld"))

def worker_func(remote, id, max_interactions):
    """
    Core loop for the subprocess. This actually holds an instance of AppWorld
    and operates the environment based on commands sent from the main process
    (such as 'step', 'reset', 'close', etc.), then returns the results.
    """
    env = None
    current_step_count = 0

    url_id = 8000 + id
    url = f"http://0.0.0.0:{url_id}"

    while True:
        cmd, data = remote.recv()
        if cmd == 'reset':
            if env is not None:
                env.close()

            task_id = data
            current_step_count = 0

            env = AppWorld(
                task_id=task_id,
                experiment_name=f'default_{id}',
                remote_environment_url=url,
            )

            obs = env.task.instruction
            info = {
                    "task_id": task_id,
                    "supervisor": dict(env.task.supervisor),
                    }
            remote.send((obs, info))

        elif cmd == 'step':
            action = data
            if env is None:
                raise RuntimeError("Environment not reset before step. Please call reset() first.")

            current_step_count += 1

            obs = env.execute(action)

            done = env.task_completed() or (current_step_count >= max_interactions)

            reward = 10.0 if env.task_completed() else 0.0

            info = {
                "won": env.task_completed(),
                "step_count": current_step_count
            }

            remote.send((obs, reward, done, info))

        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class AppWorldEnvs:
    """
    A multi-process wrapper for AppWorld.
    - Creates multiple subprocesses, each holding a separate AppWorld instance.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """
    def __init__(self, 
                 dataset_name,
                 max_interactions,
                 seed,
                 env_num,
                 group_n,
                 start_server_id,
                 ):
        super().__init__()

        self.dataset_name = dataset_name
        self.max_interactions = max_interactions
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.parent_remotes = []
        self.workers = []
        self.task_ids = load_task_ids(dataset_name)

        if sys.platform.startswith("win"):
            ctx = mp.get_context('spawn')
        else:
            ctx = mp.get_context('fork')
        
        for i in range(self.num_processes):
            parent_remote, child_remote = mp.Pipe()
            worker = ctx.Process(
                target=worker_func,
                args=(child_remote,
                      start_server_id + i,
                      self.max_interactions,
                      )
            )
            worker.daemon = True
            worker.start()

            child_remote.close()
            self.parent_remotes.append(parent_remote)
            self.workers.append(worker)

    def step(self, actions):
        """
        actions: Must be a list with length equal to self.num_processes, 
        each sent to the corresponding subprocess.
        
        Return format follows Gym's step() convention:
            observations, rewards, dones, infos
            adds:
            (obs, reward, done, info)
        """
        assert len(actions) == self.num_processes, "The length of actions must match the number of processes."

        for i, remote in enumerate(self.parent_remotes):
            remote.send(('step', actions[i]))

        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for remote in self.parent_remotes:
            obs, reward, done, info = remote.recv()
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Reset all subprocess environments simultaneously, 
        returning each environment's initial observation and info.
        """
        # randomly select self.env_num task_id from self.task_ids
        task_id = np.random.choice(self.task_ids, self.env_num, replace=False)
        # repeat task_id group_n times
        task_id = np.repeat(task_id, self.group_n).tolist()

        for i, remote in enumerate(self.parent_remotes):
            remote.send(('reset', task_id[i]))

        obs_list = []
        
        info_list = []

        for remote in self.parent_remotes:
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def close(self):
        """Close all subprocesses."""
        for remote in self.parent_remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()

    def render(self):
        """Implement this if visualization is needed."""
        pass

def build_appworld_envs(dataset_name="train",
                        max_interactions=50,
                        seed=0,
                        env_num=1, 
                        group_n=1,
                        start_server_id=0,
                        ):

    return AppWorldEnvs(
        dataset_name=dataset_name,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        start_server_id=start_server_id,
    )
