"""
Script for training and evaluating a SAC, DQN agent to navigate a maze using CellWorld.
Uses stable-baselines3 for the implementation.
"""
import gymnasium
import cellworld_gym as cwg
from stable_baselines3 import SAC, DQN
from stable_baselines3.common.buffers import ReplayBuffer
from ptsdbuffer import PTSDReplayBuffer, PTSDReplayBuffer_add
from callback import CellworldCallback
from wrapper import (
    uncertainty_wrapper,
    safe_planning_wrapper,
    uncertainty_wrapper_predator,
    uncertainty_wrapper_predator_2,
    wait_wrapper
)
from PIL import Image
from cellworld_game.video import save_video_output
import numpy as np
import pandas as pd


def collect_buffer_data(num_episodes=10000):
    """Collect training data from a trained SAC agent and save to CSV."""
    model = SAC.load("sac_bot_evade.zip")
    env = gymnasium.make(
        "CellworldBotEvade-v0",
        world_name="21_05",
        use_lppos=False,
        use_predator=True,
        max_step=300,
        time_step=0.25,
        render=False,
        real_time=False,
        reward_function=cwg.Reward({"puffed": -1, "finished": 1}),
        action_type=cwg.BotEvadeEnv.ActionType.CONTINUOUS
    )

    trajectory_data = {
        "obs": [], "action": [], "reward": [],
        "done": [], "next_obs": [], "truncated": []
    }

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Store transition
            trajectory_data["obs"].append(obs.copy())
            trajectory_data["action"].append(action)
            trajectory_data["reward"].append(reward)
            trajectory_data["done"].append(done)
            trajectory_data["truncated"].append(truncated)
            trajectory_data["next_obs"].append(next_obs.copy())
            
            obs = next_obs

    env.close()
    pd.DataFrame(trajectory_data).to_csv("SAC_data.csv", index=False)


def make_env(discrete=False):
    """Create CellWorld environment with specified settings."""
    reward_function = cwg.Reward({"puffed": -1, "finished": 1})
    action_type = cwg.BotEvadeEnv.ActionType.DISCRETE if discrete else cwg.BotEvadeEnv.ActionType.CONTINUOUS
    
    return gymnasium.make(
        "CellworldBotEvade-v0",
        world_name="21_05",
        use_lppos=False,
        use_predator=True,
        max_step=300,
        time_step=0.25,
        render=False,
        real_time=False,
        reward_function=reward_function,
        action_type=action_type
    )


def test_random_agent(env, num_steps=1000):
    """Test environment with random actions."""
    env = uncertainty_wrapper_predator(env)
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(reward)
    
    env.close()


def train_dqn(env):
    """Train DQN agent."""
    env = make_env(discrete=True)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        train_freq=(1, "step"),
        tensorboard_log="./logs"
    )
    
    model.learn(total_timesteps=100000)
    model.save("dqn")
    env.close()


def train_sac(env):
    """Train SAC agent."""
    env = wait_wrapper(env)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        train_freq=(1, "step"),
        buffer_size=100000,
        replay_buffer_class=ReplayBuffer,
        tensorboard_log="./logs",
        policy_kwargs={"net_arch": [256, 256]}
    )
    
    callback = CellworldCallback()
    model.learn(total_timesteps=100000, log_interval=20, callback=callback)
    model.save("wait_wrapper")
    model.save_replay_buffer("wait_wrapper")
    env.close()


def evaluate_sac(num_episodes=100):
    """Evaluate trained SAC agent."""
    model = SAC.load("wait_wrapper.zip")
    env = gymnasium.make(
        "CellworldBotEvade-v0",
        world_name="21_05",
        use_lppos=False,
        use_predator=True,
        max_step=300,
        time_step=0.25,
        render=True,
        real_time=True,
        reward_function=cwg.Reward({"puffed": -1, "finished": 1}),
        action_type=cwg.BotEvadeEnv.ActionType.CONTINUOUS
    )
    env = wait_wrapper(env)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        episode_reward = 0
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        print(f"Episode {episode + 1} reward: {episode_reward}")
    
    env.close()


if __name__ == "__main__":
    env = make_env()
    evaluate_sac()