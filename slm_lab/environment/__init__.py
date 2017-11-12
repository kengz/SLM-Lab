'''
The environment module
Contains graduated components from experiments for building/using environment.
Provides the rich experience for agent embodiment,
reflects the curriculum and allows teaching (possibly allows teacher to enter).
To be designed by human and evolution module,
based on the curriculum and fitness metrics.
'''
import os
import pydash as _
from slm_lab.lib import util
from unityagents.brain import BrainParameters


class BrainExt:
    '''
    Unity Brain class extension, where self = brain
    to be absorbed into ml-agents Brain class later
    '''

    def is_discrete(self):
        return self.number_observations == 'discrete'

    def get_observable(self):
        '''What channels are observable: state, visual, sound, touch, etc.'''
        observable = {
            'state': self.state_space_size > 0,
            'visual': self.number_observations > 0,
        }
        return observable


def extend_unity_brain():
    '''Extend Unity BrainParameters class at runtime to add BrainExt methods'''
    ext_fn_list = util.get_fn_list(BrainExt)
    for fn in ext_fn_list:
        setattr(BrainParameters, fn, getattr(BrainExt, fn))


extend_unity_brain()


class Env:
    '''
    Do the above
    Also standardize logic from Unity environments
    '''
    max_timestep = None
    train_mode = None
    u_env = None
    agent = None

    def __init__(self):
        return

    def set_agent(self, agent):
        '''
        Make agent visible to env.
        TODO anticipate multi-agents
        '''
        self.agent = agent

    def reset():
        return

    def step():
        return

    def close():
        return


# TODO still need a single-brain env-wrapper methods


class UnityEnv:
    '''
    Unity Environment wrapper
    '''

    def get_brain(brain_name):
        return self.u_env.brains[brain_name]

    def fn_spread_brains(self, brain_fn):
        '''Call single-brain function on all for {brain_name: info}'''
        brains_info = {
            brain_name: brain_fn(brain_name)
            for brain_name in self.u_env.brains
        }
        return brains_info

    def is_discrete(self):
        return self.fn_spread_brains('is_discrete')

    def get_observable():
        observable = self.fn_spread_brains('get_observable')
        return observable

    # TODO handle multi-brain logic
    # TODO split subclass to handle unity specific logic,
    # and the other half to handle Lab specific logic
    # TODO also make clear that unity.brain.agent is not the same as RL agent here. unity agents could be seen as multiple simultaneous incarnations of this agent
    # TODO actually shd do a single-brain wrapper instead
    # then on env level call on all brains with wrapper methods, much easier


def get_env_path(env_name):
    env_path = util.smart_path(
        f'node_modules/slm-env-{env_name}/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(
        env_dir), f'Missing {env_path}. See README to install from yarn.'
    return env_path
