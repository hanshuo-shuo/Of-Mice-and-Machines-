import gym
import gymnasium
import numpy as np
import gym
import numpy as np
from typing import Tuple, Optional

class MypreyWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.model = env.model
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(3,),  
            dtype=np.float32
        )

    def step(self, action):
        if action[2] > 0.5:
            wait_pos = self.wait_action()
            action = np.array([wait_pos[0], wait_pos[1]])
        else:
            action = action[:2].copy()
            
        obs, reward, done,  _, info = self.env.step(action)
        obs = obs.astype(np.float32)
        new_obs = obs.copy()
        new_obs = np.delete(new_obs, -4)
            
        return new_obs, reward, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self, seed=None):
        obs, _ = self.env.reset()
        obs = obs.astype(np.float32)
        new_obs = obs.copy()
        new_obs = np.delete(new_obs, -4)
        return new_obs
    
    def wait_action(self):
        noise_x = np.random.uniform(-0.02, 0.02)
        noise_y = np.random.uniform(-0.02, 0.02)
        current_x = self.env.model.prey.state.location[0]
        current_y = self.env.model.prey.state.location[1]
        new_x = np.clip(current_x + noise_x, 0.0, 1.0)
        new_y = np.clip(current_y + noise_y, 0.0, 1.0)
        return tuple((new_x, new_y))
    


class MypreyWrapper_2(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.model = env.model
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(11,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(2,),  
            dtype=np.float32
        )

    def step(self, action):
        obs, reward, done,  _, info = self.env.step(action.copy())
        obs = obs.astype(np.float32)
        return obs, reward, False, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.env.render()

    def reset(self, seed=None):
        obs, _ = self.env.reset()
        obs = obs.astype(np.float32)
        return obs
