from envs.prey_env.gym_env_bins import Environment
def random():
    env = Environment()
    env.reset()
    for i in range(10000):
        env.step(env.action_space.sample())
        env.render()
        if i % 20 == 0:
            env.reset()

if __name__=="__main__":
    random()


