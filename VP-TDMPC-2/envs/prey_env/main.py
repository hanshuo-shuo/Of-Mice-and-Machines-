from gym_env_bins import Environment

if __name__ == "__main__":
    Env = Environment()
    Env.reset()
    for i in range(50000):
        Env.step(Env.action_space.sample())
        Env.render()
        if i%50:
            Env.reset()
