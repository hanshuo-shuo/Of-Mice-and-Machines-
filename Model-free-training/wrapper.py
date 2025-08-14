import gymnasium
from collections import deque
import numpy as np


class uncertainty_wrapper_predator(gymnasium.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.recent_heuristic = deque(maxlen=5)
        self.model = env.model

    def wait_action(self):
        return tuple((self.env.model.prey.state.location[0],
                      self.env.model.prey.state.location[1]))

    def see_predator(self):
        # if the predator is visible, then it is a safe zone
        if self.env.model.prey_data.predator_visible:
            return True
        else:
            return False

    def uncertainty_level(self):
        # If the most recent heuristic is 0, the uncertainty is 0
        if self.recent_heuristic[-1] == 0:
            return 0
        # Try to find the position of the first occurrence of 0
        try:
            # Get the index of the first 0 in the deque
            zero_index = self.recent_heuristic.index(0)
            # Scale uncertainty level based on the distance from the most recent value
            # The further the 0, the higher the uncertainty, with a max of 0.5 if 0 is found
            return 0.5 + 0.5 * zero_index / (len(self.recent_heuristic) - 1)
        except ValueError:
            return 1

    def reset(self, seed=None):
        obs, info = self.env.reset()
        self.recent_heuristic.clear()
        # set the initial heuristic to 1
        for _ in range(5):
            self.recent_heuristic.append(1)
        return obs, info

    def step(self, action):
        uncertainty = self.uncertainty_level()
        if uncertainty == 1:
            new_action = self.wait_action()
        else:
            new_action = action
        obs, reward, done, truncated, info = self.env.step(new_action)
        uncertainty = self.uncertainty_level()
        adjusted_reward = reward - uncertainty * 0.1
        if self.see_predator():
            self.recent_heuristic.append(0)
        return obs, adjusted_reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render()



class uncertainty_wrapper_predator_2(gymnasium.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.model = env.model
        # self.env.loader.robot_start_locations = [(0.5, 0.5)]

    def wait_action(self):
        return tuple((self.env.model.prey.state.location[0],
                      self.env.model.prey.state.location[1]))

    def see_predator(self):
        # if the predator is visible, then it is a safe zone
        if self.env.model.prey_data.predator_visible:
            return True
        else:
            return False

    def uncertainty_level(self, obs):
        # Get the distance between prey and predator from the observation
        predator_prey_distance = obs[7]
        # Determine uncertainty level based on the distance
        if not self.see_predator():
            return 0
        elif predator_prey_distance < 0.1:
            return 1
        elif predator_prey_distance < 0.3:
            return 0.5
        else:
            return 0.1

    def reset(self, seed=None):
        obs, info = self.env.reset()
        # set the initial heuristic to 1
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        uncertainty = self.uncertainty_level(obs)
        adjusted_reward = reward - uncertainty*0.5
        return obs, adjusted_reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render()

class uncertainty_wrapper(gymnasium.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, seed=None):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if obs[1] < 0.49:
            reward -= 1
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render()


class safe_planning_wrapper(gymnasium.Env):
    def __init__(self, env, safe_planning_bonus):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.safe_planning_bonus = safe_planning_bonus  # New parameter for intrinsic reward

    def reset(self, seed=None):
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward += self.safe_planning_bonus if self.is_in_safe_zone(obs) else 0
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render()

    def is_in_safe_zone(self, obs):
        # if the predator is visible, then it is a safe zone
        if self.env.model.prey_data.predator_visible:
            return False
        else:
            return True

    def close(self):
        self.env.close()


class wait_wrapper(gymnasium.Env):
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
