import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray
from typing import Dict, List, Any, Optional

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]

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

@ray.remote(num_cpus=0.25)
class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance and belongs to a group.
    """
    
    def __init__(self, config, seed, base_env, group_id: int, worker_id: int):
        self.env = base_env.init_env(batch_size=1)
        self.env.seed(seed)
        self.group_id = group_id
        self.worker_id = worker_id
        self.global_worker_id = None  # Will be set by the main environment

        # The index of this worker inside its group is simply its `worker_id`.
        self.group_worker_index = worker_id
        
        # Worker-specific state
        self.step_count = 0
        self.episode_count = 0
    
    def set_global_worker_id(self, global_id: int):
        """Set the global worker ID (index in the overall worker list)"""
        self.global_worker_id = global_id
    
    def step(self, action, shared_llm_encoding=None):
        """Execute a step in the environment"""
        actions = [action]
        
        # The optional `shared_llm_encoding` parameter is kept for interface
        # compatibility but is no longer stored because group-level shared state
        # has been removed.
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        infos['worker_info'] = {
            'group_id': self.group_id,
            'worker_id': self.worker_id,
            'global_worker_id': self.global_worker_id,
            'group_worker_index': self.group_worker_index,
            'step_count': self.step_count
        }
        
        self.step_count += 1
        if dones[0]:
            self.episode_count += 1
            
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        infos['worker_info'] = {
            'group_id': self.group_id,
            'worker_id': self.worker_id,
            'global_worker_id': self.global_worker_id,
            'group_worker_index': self.group_worker_index,
            'step_count': self.step_count
        }
        
        self.step_count = 0
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()
        return image
    
    def get_worker_info(self):
        """Get worker information"""
        return {
            'group_id': self.group_id,
            'worker_id': self.worker_id,
            'global_worker_id': self.global_worker_id,
            'group_worker_index': self.group_worker_index,
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }

# eval mode: "eval_in_distribution" or "eval_out_of_distribution"
class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed=0, env_num=1, group_n=1, is_train=True, eval_mode="eval_out_of_distribution"):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_mode)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n
        self.env_num = env_num

        # Helper wrapper to retain the original send/recv interface -----------------
        class _RayRemoteWrapper:
            """Lightweight adapter that emulates Pipe-style ``send`` / ``recv``
            semantics on top of a Ray actor handle.  Only the commands used in this
            file are supported (step / reset / getobs / close)."""

            def __init__(self, actor_handle):
                self._actor = actor_handle
                self._future: Optional[ray.ObjectRef] = None

            # PyTorch multiprocessing pipes use tuples (cmd, data)
            def send(self, msg):
                cmd, data = msg
                if cmd == 'step':
                    self._future = self._actor.step.remote(data)
                elif cmd == 'reset':
                    self._future = self._actor.reset.remote()
                elif cmd == 'getobs':
                    self._future = self._actor.getobs.remote()
                elif cmd == 'close':
                    ray.kill(self._actor)
                    self._future = None
                else:
                    raise NotImplementedError(f"Unsupported command {cmd}")

            def recv(self):
                if self._future is None:
                    raise RuntimeError("recv called before send or after close")
                result = ray.get(self._future)
                self._future = None
                return result

        # --------------------------------------------------------------------------

        # Create Ray remote actors with group information
        self.workers = []
        self.parent_remotes = []  # keep the old variable name for compatibility
        self.worker_groups = {}  # Map worker index to group info

        for i in range(self.num_processes):
            group_id = i // self.group_n
            worker_id = i % self.group_n

            worker = AlfworldWorker.remote(
                config,
                seed + group_id,
                base_env,
                group_id,
                worker_id,
            )

            # Set global worker ID
            ray.get(worker.set_global_worker_id.remote(i))

            # Store raw actor and wrapper
            self.workers.append(worker)
            self.parent_remotes.append(_RayRemoteWrapper(worker))

            self.worker_groups[i] = {
                'group_id': group_id,
                'worker_id': worker_id,
            }

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]
        self.current_step = 0

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
            # print(f"info: {info}")
            # Flatten env-specific list/tuple fields but keep nested dicts intact
            for k in list(info.keys()):
                if k == 'worker_info':
                    continue
                if isinstance(info[k], (list, tuple)) and len(info[k]) > 0:
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
        """Reset all environments"""
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                if k != 'worker_info':  # Don't modify worker_info
                    info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        self.current_step = 0
        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """Get observation images from all workers"""
        futures = [worker.getobs.remote() for worker in self.workers]
        return ray.get(futures)

    def get_worker_info(self):
        """Get information about all workers"""
        futures = [worker.get_worker_info.remote() for worker in self.workers]
        return ray.get(futures)

    @property
    def get_admissible_commands(self):
        """Return the prev_admissible_commands stored by the main process"""
        return self.prev_admissible_commands

    def close(self):
        """Close all workers"""
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, is_train=True, eval_mode="eval_out_of_distribution"):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, is_train, eval_mode)

# Test functions
def test_group_consistency(alf_config_path, seed=42, env_num=2, group_n=3):
    """
    Test that workers in the same group have consistent behavior:
    1. Check if workers in same group have same admissible commands after reset
    2. All select first action and step, then check if observations are consistent
    """
    print("Testing Group Consistency in AlfWorld Environment...")
    print(f"Creating {env_num} groups with {group_n} workers each (total: {env_num * group_n} workers)")
    
    try:
        # Create environment
        env = AlfworldEnvs(alf_config_path, seed=seed, env_num=env_num, group_n=group_n, is_train=True)
        
        print(f"\n✓ Created environment successfully")
        
        # Get worker info to understand the grouping
        worker_infos = env.get_worker_info()
        print(f"\nWorker Group Assignment:")
        groups = {}
        for i, info in enumerate(worker_infos):
            group_id = info['group_id']
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(i)
            print(f"  Worker {i}: Group {group_id}, Local ID {info['worker_id']}")
        
        print(f"\nGroup Structure:")
        for group_id, worker_indices in groups.items():
            print(f"  Group {group_id}: Workers {worker_indices}")
        
        # Step 1: Reset environment and check admissible commands consistency
        print(f"\n🔄 Resetting environment...")
        text_obs, image_obs, infos = env.reset()
        
        print(f"\n1️⃣ Checking admissible commands consistency within groups:")
        admissible_commands = env.get_admissible_commands
        
        group_consistency = True
        for group_id, worker_indices in groups.items():
            print(f"\n  Group {group_id}:")
            group_commands = []
            for worker_idx in worker_indices:
                commands = admissible_commands[worker_idx]
                group_commands.append(commands)
                print(f"    Worker {worker_idx}: {len(commands) if commands else 0} commands")
                if commands and len(commands) > 0:
                    print(f"      First few: {commands[:3]}...")
            
            # Check if all workers in this group have same admissible commands
            if len(set([tuple(cmd) if cmd else () for cmd in group_commands])) == 1:
                print(f"    ✅ Group {group_id}: All workers have consistent admissible commands")
            else:
                print(f"    ❌ Group {group_id}: Workers have different admissible commands!")
                group_consistency = False
                # Print detailed comparison
                for i, (worker_idx, commands) in enumerate(zip(worker_indices, group_commands)):
                    print(f"      Worker {worker_idx}: {commands}")
        
        if not group_consistency:
            print(f"\n⚠️ Groups are not consistent after reset!")
            env.close()
            return False
        
        # Step 2: Select first action for all workers and step
        print(f"\n2️⃣ Testing step consistency - all workers select first admissible action:")
        
        actions = []
        for i, commands in enumerate(admissible_commands):
            if commands and len(commands) > 0:
                actions.append(commands[0])  # Select first admissible action
                print(f"    Worker {i}: Selected '{commands[0]}'")
            else:
                actions.append("look")  # Fallback action
                print(f"    Worker {i}: No commands available, using 'look'")
        
        # Execute step
        print(f"\n🚀 Executing step...")
        text_obs_after, image_obs_after, rewards, dones, infos_after = env.step(actions)
        
        # Step 3: Check observations consistency within groups
        print(f"\n3️⃣ Checking observation consistency within groups after step:")
        
        obs_consistency = True
        for group_id, worker_indices in groups.items():
            print(f"\n  Group {group_id}:")
            group_observations = []
            group_rewards = []
            group_dones = []
            
            for worker_idx in worker_indices:
                obs = text_obs_after[worker_idx]
                reward = rewards[worker_idx]
                done = dones[worker_idx]
                
                group_observations.append(obs)
                group_rewards.append(reward)
                group_dones.append(done)
                
                print(f"    Worker {worker_idx}:")
                print(f"      Obs length: {len(obs) if obs else 0}")
                print(f"      Reward: {reward}")
                print(f"      Done: {done}")
                if obs:
                    print(f"      Obs preview: {obs[:100]}...")
            
            # Check consistency within group
            if len(set(group_observations)) == 1:
                print(f"    ✅ Group {group_id}: All observations are identical")
            else:
                print(f"    ❌ Group {group_id}: Observations differ!")
                obs_consistency = False
            
            if len(set(group_rewards)) == 1:
                print(f"    ✅ Group {group_id}: All rewards are identical ({group_rewards[0]})")
            else:
                print(f"    ❌ Group {group_id}: Rewards differ! {group_rewards}")
                obs_consistency = False
            
            if len(set(group_dones)) == 1:
                print(f"    ✅ Group {group_id}: All done states are identical ({group_dones[0]})")
            else:
                print(f"    ❌ Group {group_id}: Done states differ! {group_dones}")
                obs_consistency = False
        
        # Final result
        print(f"\n📊 Test Results:")
        print(f"  Admissible Commands Consistency: {'✅ PASS' if group_consistency else '❌ FAIL'}")
        print(f"  Observation Consistency: {'✅ PASS' if obs_consistency else '❌ FAIL'}")
        
        overall_success = group_consistency and obs_consistency
        print(f"  Overall: {'✅ PASS - Groups are consistent!' if overall_success else '❌ FAIL - Groups are inconsistent!'}")
        
        env.close()
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shared_llm_encoding(alf_config_path, seed=42, env_num=2, group_n=2):
    """
    Test the shared LLM encoding functionality
    """
    print("\n" + "="*60)
    print("Testing Shared LLM Encoding Functionality...")
    
    try:
        env = AlfworldEnvs(alf_config_path, seed=seed, env_num=env_num, group_n=group_n, is_train=True)
        
        # Reset environment
        env.reset()
        
        # Create mock LLM encodings for each group
        shared_encodings = []
        for i in range(env_num):
            encoding = {
                'group_id': i,
                'encoding_data': f"mock_llm_encoding_for_group_{i}",
                'timestamp': ray.util.get_current_time_ms()
            }
            shared_encodings.append(encoding)
        
        print(f"Created shared encodings: {[enc['encoding_data'] for enc in shared_encodings]}")
        
        # Execute step with shared encodings
        actions = ["look"] * env.num_processes
        text_obs, image_obs, rewards, dones, infos = env.step(actions, shared_encodings)
        
        # Check if shared data was stored correctly
        print(f"\nChecking shared data storage:")
        for group_id in range(env_num):
            shared_data = env.get_group_shared_data(group_id)
            print(f"  Group {group_id}: {len(shared_data)} shared data entries")
            for key, data in shared_data.items():
                print(f"    {key}: {data['data']}")
        
        print(f"✅ Shared LLM encoding test completed!")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Shared encoding test failed: {e}")
        return False

if __name__ == "__main__":
    # You need to provide your actual alfworld config path
    config_path = "/common/users/wx139/code/verl-agent/agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    
    print("AlfWorld Group Management Test Suite")
    print("="*60)
    
    # Test 1: Group consistency
    success1 = test_group_consistency(config_path, seed=42, env_num=2, group_n=3)
    
    # Test 2: Shared LLM encoding
    # success2 = test_shared_llm_encoding(config_path, seed=42, env_num=2, group_n=2)
    
    print(f"\n" + "="*60)
    print(f"Final Results:")
    print(f"  Group Consistency Test: {'✅ PASS' if success1 else '❌ FAIL'}")
    # print(f"  Shared Encoding Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"  Overall: {'✅ ALL TESTS PASSED' if success1 else '❌ SOME TESTS FAILED'}")