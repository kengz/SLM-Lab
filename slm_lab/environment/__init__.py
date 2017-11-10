'''
The environment module
Contains graduated components from experiments for building/using environment.
Provides the rich experience for agent embodiment,
reflects the curriculum and allows teaching (possibly allows teacher to enter).
To be designed by human and evolution module,
based on the curriculum and fitness metrics.
'''
import os
from slm_lab.lib import util


class Env:
    '''
    Do the above
    Also standardize logic from Unity environments
    '''
    self.max_timestep
    self.train_mode

    def __init__(self):
        return

    def reset():
        return

    def step():
        return

    def close():
        return

    def is_discrete():
        return

    def get_observable():
        '''
        What channels are observable: state, visual, sound, touch, etc.
        '''
        # TODO handle multi-brain logic
        # TODO also make clear that unity.brain.agent is not the same as RL agent here. unity agents could be seen as multiple simultaneous incarnations of this agent
        # use_observations = (brain.number_observations > 0)
        # use_states = (brain.state_space_size > 0)
        observable = {
            'state': True,
            'visual': True,
        }
        return observable


def get_env_path(env_name):
    env_path = util.smart_path(
        f'node_modules/slm-env-{env_name}/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(
        env_dir), f'Missing {env_path}. See README to install from yarn.'
    return env_path
