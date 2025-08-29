# the environment module


def make_env(spec):
    from slm_lab.env.gym import GymEnv
    env = GymEnv(spec)
    return env
