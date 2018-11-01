import gym
from gym.envs.registration import register

"""
Register additional environments for OpenAI gym.
"""


def register_env(spec):
    env_name = spec['env'][0]['name']

    if env_name.lower() == 'vizdoom-v0':
        assert 'cfg_name' in spec['env'][0].keys(), 'Environment config name must be defined for vizdoom.'
        cfg_name = spec['env'][0]['cfg_name']
        register(id='vizdoom-v0',
                 entry_point='slm_lab.env.vizdoom.vizdoom_env:VizDoomEnv',
                 kwargs={'cfg_name': cfg_name})
