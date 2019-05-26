# the environment module


def make_env(spec, e=None):
    try:
        from slm_lab.env.openai import OpenAIEnv
        env = OpenAIEnv(spec, e)
    except Exception:
        from slm_lab.env.unity import UnityEnv
        env = UnityEnv(spec, e)
    return env
