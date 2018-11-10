# module to register and mange multiple environment offerings
from gym.envs.registration import register
from slm_lab.lib import util
import gym
import os


def get_env_path(env_name):
    '''Get the path to Unity env binaries distributed via npm'''
    env_path = util.smart_path(f'node_modules/slm-env-{env_name}/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(env_dir), f'Missing {env_path}. See README to install from yarn.'
    return env_path


def register_env(spec):
    '''Register additional environments for OpenAI gym.'''
    env_name = spec['env'][0]['name']

    if env_name.lower() == 'vizdoom-v0':
        assert 'cfg_name' in spec['env'][0].keys(), 'Environment config name must be defined for vizdoom.'
        cfg_name = spec['env'][0]['cfg_name']
        register(id='vizdoom-v0',
                 entry_point='slm_lab.env.vizdoom.vizdoom_env:VizDoomEnv',
                 kwargs={'cfg_name': cfg_name})
