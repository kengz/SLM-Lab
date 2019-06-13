# the environment module


def make_env(spec):
    try:
        from slm_lab.env.openai import OpenAIEnv
        env = OpenAIEnv(spec)
    except Exception:
        from slm_lab.env.unity import UnityEnv
        env = UnityEnv(spec)
    return env
