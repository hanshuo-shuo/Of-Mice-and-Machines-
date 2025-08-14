from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cellworld_gym as cwg
import gymnasium


def create_env(use_lppos: bool = True,
               use_predator: bool = False,
               max_steps: int = 300,
               time_step: float = .25,
               render: bool = False,
               real_time: bool = False,
               reward_structure: dict = {},
               **kwargs):

    return cwg.BotEvadeEnv(world_name="21_05",
                           use_lppos=use_lppos,
                           use_predator=use_predator,
                           max_step=max_steps,
                           time_step=time_step,
                           real_time=real_time,
                           render=render,
                           reward_function=cwg.BotEvadeReward(reward_structure))


def create_vec_env(environment_count: int,
                   use_lppos: bool = True,
                   use_predator: bool = False,
                   max_steps: int = 300,
                   time_step: float = 0.25,
                   reward_structure: dict = {},
                   **kwargs):
    if environment_count == 1:
        return create_env(use_lppos=use_lppos,
                          use_predator=use_predator,
                          max_steps=max_steps,
                          time_step=time_step,
                          reward_structure=reward_structure,
                          **kwargs)
    else:
        return DummyVecEnv([lambda: create_env(use_lppos=use_lppos,
                                               use_predator=use_predator,
                                               max_steps=max_steps,
                                               time_step=time_step,
                                               reward_structure=reward_structure,
                                               **kwargs)
                            for _ in range(environment_count)])

