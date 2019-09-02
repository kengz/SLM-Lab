# the environment module


def make_env(spec):
    from slm_lab.env.openai import OpenAIEnv
    env = OpenAIEnv(spec)
    return env
