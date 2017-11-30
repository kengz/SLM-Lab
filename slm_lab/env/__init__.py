'''
The environment module
Contains graduated components from experiments for building/using environment.
Provides the rich experience for agent embodiment, reflects the curriculum and allows teaching (possibly allows teacher to enter).
To be designed by human and evolution module, based on the curriculum and fitness metrics.
'''
import numpy as np
import pydash as _
from slm_lab.lib import logger, util
from unityagents import UnityEnvironment
from unityagents.brain import BrainParameters
from unityagents.environment import logger as unity_logger
from slm_lab.experiment.monitor import data_space

unity_logger.setLevel('WARN')


class BrainExt:
    '''
    Unity Brain class extension, where self = brain
    TODO to be absorbed into ml-agents Brain class later
    '''

    # TODO or just set properties for all these, no method
    def is_discrete(self):
        return self.action_space_type == 'discrete'

    def get_action_dim(self):
        return self.action_space_size

    def get_observable(self):
        '''What channels are observable: state, visual, sound, touch, etc.'''
        observable = {
            'state': self.state_space_size > 0,
            'visual': self.number_observations > 0,
        }
        return observable

    def get_observable_dim(self):
        '''Get observable dimensions'''
        observable_dim = {
            'state': self.state_space_size,
            'visual': 'some np array shape, as opposed to what Arthur called size',
        }
        return observable_dim


def extend_unity_brain():
    '''Extend Unity BrainParameters class at runtime to add BrainExt methods'''
    ext_fn_list = util.get_fn_list(BrainExt)
    for fn in ext_fn_list:
        setattr(BrainParameters, fn, getattr(BrainExt, fn))


extend_unity_brain()


class RectifiedUnityEnv:
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

    # and the other half to handle Lab specific logic
    # TODO actually shd do a single-brain wrapper instead
    # then on env level call on all brains with wrapper methods, much easier
    # Remedy:
    # 1. Env class to rectify UnityEnv
    # 2. use Env class as proper
    # Rectify steps:


class OpenAIEnv:
    # TODO merge OpenAI Env in SLM Lab
    pass


class Env:
    '''
    Class for all Envs.
    Standardizes the Env design to work in Lab.
    Access Agents properties by: Agents - AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, env_space, e=0):
        self.spec = spec
        util.set_attr(self, self.spec)
        self.env_space = env_space
        self.index = e
        self.ab_proj = self.env_space.e_ab_proj[self.index]
        self.ab_proj_bodies = None

        self.u_env = UnityEnvironment(
            file_name=util.get_env_path(self.name),
            worker_id=self.index)

        # TODO expose brain methods properly to env
        agent_index = 0
        default_brain = self.u_env.brain_names[agent_index]
        brain = self.u_env.brains[default_brain]
        ext_fn_list = util.get_fn_list(brain)
        for fn in ext_fn_list:
            setattr(self, fn, getattr(brain, fn))

    def reset(self):
        # TODO need AEB space resolver
        agent_index = 0
        default_brain = self.u_env.brain_names[agent_index]
        env_info = self.u_env.reset(train_mode=self.train_mode)[default_brain]
        # TODO body-resolver:
        body_index = 0
        # TODO make spread across body
        state = [env_info.states[body_index]]
        # TODO return observables instead of just state
        return state

    def step(self, action):
        # TODO need AEB space resolver
        agent_index = 0
        default_brain = self.u_env.brain_names[agent_index]
        env_info = self.u_env.step(action)[default_brain]
        # TODO body-resolver:
        body_index = 0
        # TODO tmp make across bodies
        reward = [env_info.rewards[body_index]]
        state = [env_info.states[body_index]]
        done = [env_info.local_done[body_index]]
        return reward, state, done

    def close(self):
        self.u_env.close()


class EnvSpace:
    '''
    Subspace of AEBSpace, collection of all envs, with interface to Session logic; same methods as singleton envs.
    Access AgentSpace properties by: AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, aeb_space):
        self.spec = spec
        self.aeb_space = aeb_space
        aeb_space.env_space = self
        self.e_ab_proj = aeb_space.e_ab_proj
        self.envs = [Env(_.merge(e_spec, spec['meta']), self)
                     for e, e_spec in enumerate(spec['env'])]
        self.max_timestep = np.amax([env.max_timestep for env in self.envs])

    def get(self, e):
        return self.envs[e]

    def reset(self):
        state_proj = []
        for env in self.envs:
            state = env.reset()
            state_proj.append(state)
        state_space = self.aeb_space.add('state', state_proj)
        return state_space

    def step(self, action_space):
        reward_proj = []
        state_proj = []
        done_proj = []
        for e, env in enumerate(self.envs):
            action = action_space.get(e=e)
            reward, state, done = env.step(action)
            reward_proj.append(reward)
            state_proj.append(state)
            done_proj.append(done)
        reward_space = self.aeb_space.add('reward', reward_proj)
        state_space = self.aeb_space.add('state', state_proj)
        done_space = self.aeb_space.add('done', done_proj)
        return reward_space, state_space, done_space

    def close(self):
        for env in self.envs:
            env.close()
