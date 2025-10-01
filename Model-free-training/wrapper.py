import gymnasium
from collections import deque
import numpy as np



class myprey_wrapper(gymnasium.Env):
    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

    def see_predator(self):
        # if the predator is visible, then it is a safe zone
        if self.model.prey_data.predator_visible:
            return True
        else:
            return False

    def step(self, action):
        if action[2] > 0.5:
            wait_pos = self.wait_action()
            action = np.array([wait_pos[0], wait_pos[1], 1])

        obs, reward, done, tr, info = self.env.step(action[:2].copy())
        obs = obs.astype(np.float32)
        new_obs = obs.copy()
        new_obs = np.delete(new_obs, -4)
        return new_obs, reward, done, tr, info

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
        return new_obs, _

    def wait_action(self):
        noise_x = np.random.uniform(-0.02, 0.02)
        noise_y = np.random.uniform(-0.02, 0.02)
        current_x = self.env.model.prey.state.location[0]
        current_y = self.env.model.prey.state.location[1]
        new_x = np.clip(current_x + noise_x, 0.0, 1.0)
        new_y = np.clip(current_y + noise_y, 0.0, 1.0)
        return tuple((new_x, new_y))
