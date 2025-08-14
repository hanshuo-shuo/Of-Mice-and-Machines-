import gym
import gymnasium
class GymnasiumToGymWrapper(gym.Env):
    def __init__(self, env, cfg):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.model = env.model
        self.cfg = cfg

    def reset(self):
        obs, info = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done or truncated, info

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()