'''
The environment module
Contains graduated components from experiments for building/using environment.
Provides the rich experience for agent embodiment, reflects the curriculum and allows teaching (possibly allows teacher to enter).
To be designed by human and evolution module, based on the curriculum and fitness metrics.
'''
from abc import ABC, abstractmethod
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from unityagents import brain, UnityEnvironment
import gym
import logging
import numpy as np
import os
import pydash as ps

ENV_DATA_NAMES = ['reward', 'state', 'done']
logger = logger.get_logger(__name__)


def make_env(spec):
    try:
        env = OpenAIEnv(spec)
    except gym.error.Error:
        env = UnityEnv(spec)
    return env


def set_gym_space_attr(gym_space):
    '''Set missing gym space attributes for standardization'''
    if isinstance(gym_space, gym.spaces.Box):
        pass
    elif isinstance(gym_space, gym.spaces.Discrete):
        setattr(gym_space, 'low', 0)
        setattr(gym_space, 'high', gym_space.n)
    elif isinstance(gym_space, gym.spaces.MultiBinary):
        setattr(gym_space, 'low', np.full(gym_space.n, 0))
        setattr(gym_space, 'high', np.full(gym_space.n, 2))
    elif isinstance(gym_space, gym.spaces.MultiDiscrete):
        setattr(gym_space, 'low', np.zeros_like(nvec))
        setattr(gym_space, 'high', np.array(gym_space.nvec))
    else:
        raise ValueError('gym_space not recognized')


class Clock:
    '''Clock class for each env and space to keep track of relative time. Ticking and control loop is such that reset is at t=0, but epi begins at 1, env step begins at 1.'''

    def __init__(self, clock_speed=1):
        self.clock_speed = int(clock_speed)
        self.ticks = 0  # multiple ticks make a timestep; used for clock speed
        self.t = 0
        self.total_t = 0
        self.epi = 0

    def to_step(self):
        '''Step signal from clock_speed. Step only if the base unit of time in this clock has moved. Used to control if env of different clock_speed should step()'''
        return self.ticks % self.clock_speed == 0

    def tick(self, unit='t'):
        if unit == 't':  # timestep
            if self.to_step():
                self.t += 1
                self.total_t += 1
            else:
                pass
            self.ticks += 1
        elif unit == 'epi':  # episode, reset timestep
            self.epi += 1
            self.t = 0
        else:
            raise KeyError

    def get(self, unit='t'):
        return getattr(self, unit)


# TODO move these classes out to files
class BaseEnv(ABC):
    '''The base Env class with API and helper methods. Use this to implement your env class that is compatible with the Lab APIs'''

    def __init__(self, spec, e=None):
        self.e = e or 0  # for compatibility with env_space
        self.clock = Clock()
        self.done = False
        self.env_spec = spec['env'][self.e]
        util.set_attr(self, self.env_spec, [
            'name',
            'max_timestep',
            'max_episode',
            'save_epi_frequency',
        ])

    def _get_spaces(self, u_env):
        '''Helper to set the extra attributes to, and get, observation and action spaces'''
        observation_space = u_env.observation_space
        action_space = u_env.action_space
        set_gym_space_attr(observation_space)
        set_gym_space_attr(action_space)
        return observation_space, action_space

    def _get_observable_dim(self, observation_space):
        '''Get the observable dim for an agent in env'''
        state_dim = observation_space.shape
        if len(state_dim) == 1:
            state_dim = state_dim[0]
        return {'state': state_dim}

    def _get_action_dim(self, action_space):
        '''Get the action dim for an action_space for agent to use'''
        if isinstance(action_space, gym.spaces.Box):
            assert len(action_space.shape) == 1
            action_dim = action_space.shape[0]
        elif isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiBinary)):
            action_dim = action_space.n
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            action_dim = action_space.nvec.tolist()
        else:
            raise ValueError('action_space not recognized')
        return action_dim

    def _is_discrete(self, action_space):
        '''Check if an action space is discrete'''
        return util.get_class_name(action_space) != 'Box'

    @abstractmethod
    @lab_api
    def reset(self):
        '''Reset method, return _reward, state, done'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def step(self, action):
        '''Step method, return reward, state, done'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def close(self):
        '''Method to close and cleanup env'''
        raise NotImplementedError

    @lab_api
    def space_init(self, env_space):
        '''Post init override for space env. Note that aeb is already correct from __init__'''
        raise NotImplementedError

    @lab_api
    def space_reset(self):
        '''Space (multi-env) reset method, return _reward_e, state_e, done_e'''
        raise NotImplementedError

    @lab_api
    def space_step(self, action_e):
        '''Space (multi-env) step method, return reward_e, state_e, done_e'''
        raise NotImplementedError


class OpenAIEnv(BaseEnv):
    def __init__(self, spec, e=None):
        super(OpenAIEnv, self).__init__(spec, e)

        self.u_env = gym.make(self.name)
        self.observation_space, self.action_space = self._get_spaces(self.u_env)
        self.observable_dim = self._get_observable_dim(self.observation_space)
        self.action_dim = self._get_action_dim(self.action_space)
        self.is_discrete = self._is_discrete(self.action_space)
        self.max_timestep = self.max_timestep or self.u_env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        if e is None:  # no e specified, is in singleton mode
            logger.info(util.self_desc(self))

    @lab_api
    def reset(self):
        _reward = np.nan
        state = self.u_env.reset()
        self.done = done = False
        if util.get_lab_mode() == 'dev':
            self.u_env.render()
        logger.debug(f'Env {self.e} reset reward: {_reward}, state: {state}, done: {done}')
        self.aeb_space.add_single(ENV_DATA_NAMES, (_reward, state, done))
        return _reward, state, done

    @lab_api
    def step(self, action):
        if not self.is_discrete:  # guard for continuous
            action = np.array([action])
        state, reward, done, _info = self.u_env.step(action)
        if util.get_lab_mode() == 'dev':
            self.u_env.render()
        self.done = done = done or self.clock.get('t') > self.max_timestep
        logger.debug(f'Env {self.e} step reward: {reward}, state: {state}, done: {done}')
        self.aeb_space.add_single(ENV_DATA_NAMES, (reward, state, done))
        return reward, state, done

    @lab_api
    def close(self):
        self.u_env.close()

    @lab_api
    def space_init(self, env_space):
        '''Post init override for space env. Note that aeb is already correct from __init__'''
        self.env_space = env_space
        self.observation_spaces = [self.observation_space]
        self.action_spaces = [self.action_space]

        # TODO have the body initialized and set directly
        self.nanflat_body_e = util.nanflatten(self.body_e)
        for idx, body in enumerate(self.nanflat_body_e):
            body.nanflat_e_idx = idx
        self.body_num = len(self.nanflat_body_e)
        assert len(self.body_num) == 1, 'OpenAI Gym supports only single body'

        # self.body_e = None
        # self.nanflat_body_e = None  # nanflatten version of bodies
        # self.body_num = None
        logger.info(util.self_desc(self))

    @lab_api
    def space_reset(self):
        _reward_e, state_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for ab, body in util.ndenumerate_nonan(self.body_e):
            state = self.u_env.reset()
            state_e[ab] = state
            done_e[ab] = self.done = False
        if util.get_lab_mode() == 'dev':
            self.u_env.render()
        logger.debug(f'Env {self.e} reset reward_e: {_reward_e}, state_e: {state_e}, done_e: {done_e}')
        return _reward_e, state_e, done_e

    @lab_api
    def space_step(self, action_e):
        action = action_e[(0, 0)]  # single body
        if self.done:  # space envs run continually without a central reset signal
            return self.reset()
        if not self.is_discrete:
            action = np.array([action])
        (state, reward, done, _info) = self.u_env.step(action)
        if util.get_lab_mode() == 'dev':
            self.u_env.render()
        self.done = done = done or self.clock.get('t') > self.max_timestep
        reward_e, state_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for ab, body in util.ndenumerate_nonan(self.body_e):
            reward_e[ab] = reward
            state_e[ab] = state
            done_e[ab] = done
        logger.debug(f'Env {self.e} step reward_e: {reward_e}, state_e: {state_e}, done_e: {done_e}')
        return reward_e, state_e, done_e


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

    def get_observable_types(self):
        '''What channels are observable: state, image, sound, touch, etc.'''
        observable = {
            'state': self.state_space_size > 0,
            'image': self.number_observations > 0,
        }
        return observable

    def get_observable_dim(self):
        '''Get observable dimensions'''
        observable_dim = {
            'state': self.state_space_size,
            'image': 'some np array shape, as opposed to what Arthur called size',
        }
        return observable_dim


# Extend Unity BrainParameters class at runtime to add BrainExt methods
util.monkey_patch(brain.BrainParameters, BrainExt)


class UnityEnv:
    '''
    Class for all Envs.
    Standardizes the UnityEnv design to work in Lab.
    Access Agents properties by: Agents - AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, env_spec, env_space, e=0):
        self.env_spec = env_spec
        self.env_space = env_space
        self.info_space = env_space.info_space
        self.e = e
        util.set_attr(self, self.env_spec)
        self.name = self.env_spec['name']
        self.body_e = None
        self.nanflat_body_e = None  # nanflatten version of bodies
        self.body_num = None

        worker_id = int(f'{os.getpid()}{self.e+int(ps.unique_id())}'[-4:])
        self.u_env = UnityEnvironment(file_name=util.get_env_path(self.name), worker_id=worker_id)
        # spaces for NN auto input/output inference
        logger.warn('Unity environment observation_space and action_space are constructed with invalid range. Use only their shapes.')
        self.observation_spaces = []
        self.action_spaces = []
        for a in range(len(self.u_env.brain_names)):
            observation_shape = (self.get_observable_dim(a)['state'],)
            if self.get_brain(a).state_space_type == 'discrete':
                observation_space = gym.spaces.Box(low=0, high=1, shape=observation_shape, dtype=np.int32)
            else:
                observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=observation_shape, dtype=np.float32)
            self.observation_spaces.append(observation_space)
            if self.is_discrete(a):
                action_space = gym.spaces.Discrete(self.get_action_dim(a))
            else:
                action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_spaces.append(action_space)
        for observation_space, action_space in zip(self.observation_spaces, self.action_spaces):
            set_gym_space_attr(observation_space)
            set_gym_space_attr(action_space)

        # TODO experiment to find out optimal benchmarking max_timestep, set
        # TODO ensure clock_speed from env_spec
        self.clock_speed = 1
        self.clock = Clock(self.clock_speed)
        self.done = False

    def check_u_brain_to_agent(self):
        '''Check the size match between unity brain and agent'''
        u_brain_num = self.u_env.number_brains
        agent_num = len(self.body_e)
        assert u_brain_num == agent_num, f'There must be a Unity brain for each agent. e:{self.e}, brain: {u_brain_num} != agent: {agent_num}.'

    def check_u_agent_to_body(self, env_info_a, a):
        '''Check the size match between unity agent and body'''
        u_agent_num = len(env_info_a.agents)
        body_num = util.count_nonan(self.body_e[a])
        assert u_agent_num == body_num, f'There must be a Unity agent for each body; a:{a}, e:{self.e}, agent_num: {u_agent_num} != body_num: {body_num}.'

    def get_brain(self, a):
        '''Get the unity-equivalent of agent, i.e. brain, to access its info'''
        name_a = self.u_env.brain_names[a]
        brain_a = self.u_env.brains[name_a]
        return brain_a

    def get_env_info(self, env_info_dict, a):
        name_a = self.u_env.brain_names[a]
        env_info_a = env_info_dict[name_a]
        return env_info_a

    @lab_api
    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        self.nanflat_body_e = util.nanflatten(self.body_e)
        for idx, body in enumerate(self.nanflat_body_e):
            body.nanflat_e_idx = idx
        self.body_num = len(self.nanflat_body_e)
        self.check_u_brain_to_agent()
        logger.info(util.self_desc(self))

    def is_discrete(self, a):
        '''Check if an agent (brain) is subject to discrete actions'''
        return self.get_brain(a).is_discrete()

    def get_action_dim(self, a):
        '''Get the action dim for an agent (brain) in env'''
        return self.get_brain(a).get_action_dim()

    def get_action_space(self, a):
        return self.action_spaces[a]

    def get_observable_dim(self, a):
        '''Get the observable dim for an agent (brain) in env'''
        return self.get_brain(a).get_observable_dim()

    def get_observable_types(self, a):
        '''Get the observable for an agent (brain) in env'''
        return self.get_brain(a).get_observable_types()

    def get_observation_space(self, a):
        return self.observation_spaces[a]

    @lab_api
    def reset(self):
        self.done = False
        env_info_dict = self.u_env.reset(train_mode=(util.get_lab_mode() != 'dev'), config=self.env_spec.get('unity'))
        _reward_e, state_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for (a, b), body in util.ndenumerate_nonan(self.body_e):
            env_info_a = self.get_env_info(env_info_dict, a)
            self.check_u_agent_to_body(env_info_a, a)
            state = env_info_a.states[b]
            state_e[(a, b)] = state
            done_e[(a, b)] = self.done
        logger.debug(f'Env {self.e} reset reward_e: {_reward_e}, state_e: {state_e}, done_e: {done_e}')
        return _reward_e, state_e, done_e

    @lab_api
    def step(self, action_e):
        # TODO implement clock_speed: step only if self.clock.to_step()
        if self.done:
            return self.reset()
        action_e = util.nanflatten(action_e)
        env_info_dict = self.u_env.step(action_e)
        reward_e, state_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for (a, b), body in util.ndenumerate_nonan(self.body_e):
            env_info_a = self.get_env_info(env_info_dict, a)
            reward_e[(a, b)] = env_info_a.rewards[b]
            state_e[(a, b)] = env_info_a.states[b]
            done_e[(a, b)] = env_info_a.local_done[b]
        self.done = (util.nonan_all(done_e) or self.clock.get('t') > self.max_timestep)
        logger.debug(f'Env {self.e} step reward_e: {reward_e}, state_e: {state_e}, done_e: {done_e}')
        return reward_e, state_e, done_e

    @lab_api
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
        self.env_spec = spec['env']
        self.info_space = aeb_space.info_space
        self.envs = []
        for e, env_spec in enumerate(self.env_spec):
            env_spec = ps.merge(spec['meta'].copy(), env_spec)
            env = make_env(self.spec)
            env.e = e
            self.envs.append(env)

    @lab_api
    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        for env in self.envs:
            env.post_body_init()
        logger.info(util.self_desc(self))

    def get(self, e):
        return self.envs[e]

    def get_base_clock(self):
        '''Get the clock with the finest time unit, i.e. ticks the most cycles in a given time, or the highest clock_speed'''
        fastest_env = ps.max_by(self.envs, lambda env: env.clock_speed)
        clock = fastest_env.clock
        return clock

    @lab_api
    def reset(self):
        logger.debug3('EnvSpace.reset')
        _reward_v, state_v, done_v = self.aeb_space.init_data_v(ENV_DATA_NAMES)
        for env in self.envs:
            _reward_e, state_e, done_e = env.reset()
            state_v[env.e, 0:len(state_e)] = state_e
            done_v[env.e, 0:len(done_e)] = done_e
        _reward_space, state_space, done_space = self.aeb_space.add(ENV_DATA_NAMES, (_reward_v, state_v, done_v))
        logger.debug3(f'\nstate_space: {state_space}')
        return _reward_space, state_space, done_space

    @lab_api
    def step(self, action_space):
        reward_v, state_v, done_v = self.aeb_space.init_data_v(ENV_DATA_NAMES)
        for env in self.envs:
            e = env.e
            action_e = action_space.get(e=e)
            reward_e, state_e, done_e = env.step(action_e)
            reward_v[e, 0:len(reward_e)] = reward_e
            state_v[e, 0:len(state_e)] = state_e
            done_v[e, 0:len(done_e)] = done_e
        reward_space, state_space, done_space = self.aeb_space.add(ENV_DATA_NAMES, (reward_v, state_v, done_v))
        logger.debug3(f'\nreward_space: {reward_space}\nstate_space: {state_space}\ndone_space: {done_space}')
        return reward_space, state_space, done_space

    @lab_api
    def close(self):
        logger.info('EnvSpace.close')
        for env in self.envs:
            env.close()
