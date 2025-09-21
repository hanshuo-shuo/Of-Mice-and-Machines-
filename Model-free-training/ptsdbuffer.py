import numpy as np
import random
from typing import Any, Dict, List, Optional, Union
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from gymnasium import spaces
import torch as th
from stable_baselines3.common.vec_env import VecNormalize

class PTSDReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Calculate the number of negative and random samples
        num_negative_samples = int(batch_size * 0.7)
        num_random_samples = batch_size - num_negative_samples

        # Find indices of negative rewards
        negative_indices = np.where(self.rewards[:self.pos] < 0)[0] if not self.full else np.where(self.rewards < 0)[0]

        # Sample negative experiences
        if len(negative_indices) > 0:
            negative_sample_indices = np.random.choice(negative_indices, size=min(num_negative_samples, len(negative_indices)), replace=True)
        else:
            negative_sample_indices = np.array([])

        # Sample random experiences
        max_index = self.buffer_size if self.full else self.pos
        random_sample_indices = np.random.choice(max_index, size=num_random_samples, replace=True)

        # Combine the indices
        sample_indices = np.concatenate((negative_sample_indices, random_sample_indices))

        return self._get_samples(sample_indices, env=env)

class PTSDReplayBuffer_add(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Determine how many times to store the transition
        repeat_count = 2 if reward < 0 else 1

        for _ in range(repeat_count):
            # Reshape needed when using multiple envs with discrete observations
            if isinstance(self.observation_space, spaces.Discrete):
                obs = obs.reshape((self.n_envs, *self.obs_shape))
                next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

            # Reshape to handle multi-dim and discrete action spaces
            action = action.reshape((self.n_envs, self.action_dim))

            # Copy to avoid modification by reference
            self.observations[self.pos] = np.array(obs)

            if self.optimize_memory_usage:
                self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
            else:
                self.next_observations[self.pos] = np.array(next_obs)

            self.actions[self.pos] = np.array(action)
            self.rewards[self.pos] = np.array(reward)
            self.dones[self.pos] = np.array(done)

            if self.handle_timeout_termination:
                self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0