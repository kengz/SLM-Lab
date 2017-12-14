'''
The environment module
Contains graduated components from experiments for building/using environment.
Provides the rich experience for agent embodiment, reflects the curriculum and allows teaching (possibly allows teacher to enter).
To be designed by human and evolution module, based on the curriculum and fitness metrics.
'''
from slm_lab.experiment.monitor import info_space
from slm_lab.lib import logger, util
from unityagents import UnityEnvironment
from unityagents.brain import BrainParameters
from unityagents.environment import logger as unity_logger
import gym
import numpy as np
import os
import pydash as _

gym.logger.setLevel('ERROR')
unity_logger.setLevel('ERROR')


class BrainExt:
    '''
    Unity Brain class extension, where self = brain
    TODO to be absorbed into ml-agents Brain class later
    TODO or just set properties for all these, no method
    '''

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


class OpenAIEnv:
    # TODO check done on solve_mean_rewards
    def __init__(self, spec, env_space, e=0):
        self.spec = spec
        util.set_attr(self, self.spec)
        self.name = self.spec['name']
        self.env_space = env_space
        self.index = e
        self.ab_proj = self.env_space.e_ab_proj[self.index]
        self.bodies = None  # consistent with ab_proj, set in aeb_space.init_body_space()
        self.u_env = gym.make(self.name)
        self.max_timestep = self.max_timestep or self.u_env.spec.tags.get(
            'wrapper_config.TimeLimit.max_episode_steps')

    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        pass

    def is_discrete(self, a):
        '''Check if an agent (brain) is subject to discrete actions'''
        return self.u_env.action_space.__class__.__name__ != 'Box'  # continuous

    def get_action_dim(self, a):
        '''Get the action dim for an agent (brain) in env'''
        if self.is_discrete(a=0):
            action_dim = self.u_env.action_space.n
        else:
            action_dim = self.u_env.action_space.shape[0]
        return action_dim

    def get_observable(self, a):
        '''Get the observable for an agent (brain) in env'''
        # TODO detect if is pong from pixel
        return {'state': True, 'visual': False}

    def get_observable_dim(self, a):
        '''Get the observable dim for an agent (brain) in env'''
        state_dim = self.u_env.observation_space.shape[0]
        if (len(self.u_env.observation_space.shape) > 1):
            state_dim = self.u_env.observation_space.shape
        return {'state': state_dim}

    def reset(self):
        self.done = False
        state = []
        body_state = self.u_env.reset()
        for a, b in self.ab_proj:
            state.append(body_state)
        assert len(state) == 1, 'OpenAI Gym supports only single body'
        return state

    def step(self, action):
        # TODO hack for mismaching env timesteps
        if self.done:
            self.reset()
        assert len(action) == 1, 'OpenAI Gym supports only single body'
        if not self.train_mode:
            self.u_env.render()
        body_action = action[0]
        body_state, body_reward, body_done, _info = self.u_env.step(
            body_action)
        reward = [body_reward]
        state = [body_state]
        done = [body_done]
        self.done = body_done
        return reward, state, done

    def close(self):
        self.u_env.close()


class Env:
    '''
    Class for all Envs.
    Standardizes the Env design to work in Lab.
    Access Agents properties by: Agents - AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, env_space, e=0):
        self.spec = spec
        util.set_attr(self, self.spec)
        self.name = self.spec['name']
        self.env_space = env_space
        self.index = e
        self.ab_proj = self.env_space.e_ab_proj[self.index]
        self.bodies = None  # consistent with ab_proj, set in aeb_space.init_body_space()
        worker_id = int(f'{os.getpid()}{self.index}'[-4:])
        self.u_env = UnityEnvironment(
            file_name=util.get_env_path(self.name), worker_id=worker_id)
        self.check_u_brain_to_agent()

    def check_u_brain_to_agent(self):
        '''Check the size match between unity brain and agent'''
        u_brain_num = self.u_env.number_brains
        agent_num = util.get_aeb_shape(self.ab_proj)[0]
        assert u_brain_num == agent_num, f'There must be a Unity brain for each agent; failed check brain: {u_brain_num} == agent: {agent_num}.'

    def check_u_agent_to_body(self, a_env_info, a):
        '''Check the size match between unity agent and body'''
        u_agent_num = len(a_env_info.agents)
        a_body_num = len(_.filter_(self.ab_proj, lambda ab: ab[0] == a))
        assert u_agent_num == a_body_num, f'There must be a Unity agent for each body; failed check agent: {u_agent_num} == body: {a_body_num}.'

    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        pass

    def get_brain(self, a):
        '''Get the unity-equivalent of agent, i.e. brain, to access its info'''
        a_name = self.u_env.brain_names[a]
        a_brain = self.u_env.brains[a_name]
        return a_brain

    def is_discrete(self, a):
        '''Check if an agent (brain) is subject to discrete actions'''
        return self.get_brain(a).is_discrete()

    def get_action_dim(self, a):
        '''Get the action dim for an agent (brain) in env'''
        return self.get_brain(a).get_action_dim()

    def get_observable(self, a):
        '''Get the observable for an agent (brain) in env'''
        return self.get_brain(a).get_observable()

    def get_observable_dim(self, a):
        '''Get the observable dim for an agent (brain) in env'''
        return self.get_brain(a).get_observable_dim()

    def reset(self):
        env_info_dict = self.u_env.reset(
            train_mode=self.train_mode, config=self.spec.get('unity'))
        state = []
        for a, b in self.ab_proj:
            a_name = self.u_env.brain_names[a]
            a_env_info = env_info_dict[a_name]
            self.check_u_agent_to_body(a_env_info, a)
            body_state = a_env_info.states[b]
            state.append(body_state)
        return state

    def step(self, action):
        env_info_dict = self.u_env.step(action)
        reward = []
        state = []
        done = []
        for a, b in self.ab_proj:
            a_name = self.u_env.brain_names[a]
            a_env_info = env_info_dict[a_name]
            body_reward = a_env_info.rewards[b]
            reward.append(body_reward)
            body_state = a_env_info.states[b]
            state.append(body_state)
            body_done = a_env_info.local_done[b]
            done.append(body_done)
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
        self.envs = []
        for e, e_spec in enumerate(spec['env']):
            try:
                env = OpenAIEnv(_.merge(spec['meta'].copy(), e_spec), self, e)
            except gym.error.Error:
                env = Env(_.merge(spec['meta'].copy(), e_spec), self, e)
            self.envs.append(env)
        self.max_timestep = np.amax([env.max_timestep for env in self.envs])

    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        for env in self.envs:
            env.post_body_init()

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
