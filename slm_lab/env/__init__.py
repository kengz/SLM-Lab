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
        self.body_e = None
        self.flat_nonan_body_e = None  # flatten_nonan version of bodies
        self.u_env = gym.make(self.name)
        self.max_timestep = self.max_timestep or self.u_env.spec.tags.get(
            'wrapper_config.TimeLimit.max_episode_steps')

    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        self.flat_nonan_body_e = util.flatten_nonan(self.body_e)

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
        state = np.full(self.body_e.shape, np.nan, dtype=object)
        for (a, b), body in np.ndenumerate(self.body_e):
            if body is np.nan:
                continue
            body_state = self.u_env.reset()
            # set body_data
            state[(a, b)] = body_state
        non_nan_cnt = util.count_nonnan(state)
        assert util.count_nonnan(
            state) == 1, 'OpenAI Gym supports only single body'
        return state

    def step(self, action):
        # TODO hack for mismaching env timesteps
        if self.done:
            self.reset()
        # TODO spread action from agent
        assert len(action) == 1, 'OpenAI Gym supports only single body'
        if not self.train_mode:
            self.u_env.render()
        body_action = action[(0, 0)]
        (body_state, body_reward, body_done, _info) = self.u_env.step(body_action)
        reward = np.full(self.body_e.shape, np.nan)
        state = np.full(self.body_e.shape, np.nan, dtype=object)
        done = reward.copy()
        for (a, b), body in np.ndenumerate(self.body_e):
            if body is np.nan:
                continue
            # set body_data
            reward[(a, b)] = body_reward
            state[(a, b)] = body_state
            done[(a, b)] = body_done
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
        # TODO rename with consistent semantics and data_space, maybe body_e
        self.body_e = None
        self.flat_nonan_body_e = None  # flatten_nonan version of bodies
        worker_id = int(f'{os.getpid()}{self.index}'[-4:])
        self.u_env = UnityEnvironment(
            file_name=util.get_env_path(self.name), worker_id=worker_id)
        # TODO experiment to find out optimal benchmarking max_timestep, set

    def check_u_brain_to_agent(self):
        '''Check the size match between unity brain and agent'''
        u_brain_num = self.u_env.number_brains
        agent_num = len(self.body_e)
        assert u_brain_num == agent_num, f'There must be a Unity brain for each agent; failed check brain: {u_brain_num} == agent: {agent_num}.'

    def check_u_agent_to_body(self, a_env_info, a):
        '''Check the size match between unity agent and body'''
        u_agent_num = len(a_env_info.agents)
        a_body_num = util.count_nonnan(self.body_e[a])
        assert u_agent_num == a_body_num, f'There must be a Unity agent for each body; failed check agent: {u_agent_num} == body: {a_body_num}.'

    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        self.flat_nonan_body_e = util.flatten_nonan(self.body_e)
        self.check_u_brain_to_agent()

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

    def get_env_info(self, env_info_dict, a):
        a_name = self.u_env.brain_names[a]
        a_env_info = env_info_dict[a_name]
        return a_env_info

    def reset(self):
        env_info_dict = self.u_env.reset(
            train_mode=self.train_mode, config=self.spec.get('unity'))
        state = np.full(self.body_e.shape, np.nan, dtype=object)
        for (a, b), body in np.ndenumerate(self.body_e):
            # TODO refactor this
            if body is np.nan:
                continue
            a_env_info = self.get_env_info(env_info_dict, a)
            self.check_u_agent_to_body(a_env_info, a)
            # set body_data
            state[(a, b)] = a_env_info.states[b]
        return state

    def step(self, action):
        # TODO spread action from agent
        env_info_dict = self.u_env.step(action)
        reward = np.full(self.body_e.shape, np.nan)
        state = np.full(self.body_e.shape, np.nan, dtype=object)
        done = reward.copy()
        for (a, b), body in np.ndenumerate(self.body_e):
            if body is np.nan:
                continue
            a_env_info = self.get_env_info(env_info_dict, a)
            # set body_data
            reward[(a, b)] = a_env_info.rewards[b]
            state[(a, b)] = a_env_info.states[b]
            done[(a, b)] = a_env_info.local_done[b]
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
        self.aeb_shape = aeb_space.aeb_shape
        aeb_space.env_space = self
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
        state_data = np.full(self.aeb_shape, np.nan, dtype=object)
        for env in self.envs:
            state = env.reset()
            state_data[env.index] = state
        state_space = self.aeb_space.add('state', state_data)
        return state_space

    def step(self, action_space):
        reward_data = np.full(self.aeb_shape, np.nan)
        state_data = np.full(self.aeb_shape, np.nan, dtype=object)
        done_data = reward_data.copy()
        for env in self.envs:
            e = env.index
            action = action_space.get(e=e)
            reward, state, done = env.step(action)
            reward_data[e] = reward
            state_data[e] = state
            done_data[e] = done
        reward_space = self.aeb_space.add('reward', reward_data)
        state_space = self.aeb_space.add('state', state_data)
        done_space = self.aeb_space.add('done', done_data)
        return reward_space, state_space, done_space

    def close(self):
        for env in self.envs:
            env.close()
