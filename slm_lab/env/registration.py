# module to register and mange multiple environment offerings
from gym.envs.registration import register
from slm_lab.lib import logger, util
import gym
import os


def get_env_path(env_name):
    '''Get the path to Unity env binaries distributed via npm'''
    env_path = util.smart_path(f'slm_lab/env/SLM-Env/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(env_dir), f'Missing {env_path}. See README to install from yarn.'
    return env_path


def try_register_env(spec):
    '''Try to additional environments for OpenAI gym.'''
    try:
        env_name = spec['env'][0]['name']
        if env_name == 'vizdoom-v0':
            assert 'cfg_name' in spec['env'][0].keys(), 'Environment config name must be defined for vizdoom.'
            cfg_name = spec['env'][0]['cfg_name']
            register(
                id=env_name,
                entry_point='slm_lab.env.vizdoom.vizdoom_env:VizDoomEnv',
                kwargs={'cfg_name': cfg_name})
        elif env_name.startswith('Unity'):
            register(
                id=env_name,
                entry_point='slm_lab.env.unity:GymUnityEnv',
                max_episode_steps=1000,  # default value different from spec
                kwargs={'name': env_name})
    except Exception as e:
        logger.exception(e)
