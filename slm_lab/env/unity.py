from gym import spaces
from slm_lab.env.base import BaseEnv, ENV_DATA_NAMES, set_gym_space_attr
from slm_lab.env.registration import get_env_path
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from unityagents import brain, UnityEnvironment
import numpy as np
import os
import pydash as ps

logger = logger.get_logger(__name__)


class BrainExt:
    '''Unity Brain class extension, where self = brain'''

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


class UnityEnv(BaseEnv):
    '''
    Wrapper for Unity ML-Agents env to work with the Lab.

    e.g. env_spec
    "env": [{
      "name": "gridworld",
      "max_t": 20,
      "max_tick": 3,
      "unity": {
        "gridSize": 6,
        "numObstacles": 2,
        "numGoals": 1
      }
    }],
    '''

    def __init__(self, spec, e=None, env_space=None):
        super(UnityEnv, self).__init__(spec, e, env_space)
        util.set_attr(self, self.env_spec, ['unity'])
        worker_id = int(f'{os.getpid()}{self.e+int(ps.unique_id())}'[-4:])
        seed = ps.get(spec, 'meta.random_seed')
        # TODO update Unity ml-agents to use seed=seed below
        self.u_env = UnityEnvironment(file_name=get_env_path(self.name), worker_id=worker_id)
        self.patch_gym_spaces(self.u_env)
        self._set_attr_from_u_env(self.u_env)
        assert self.max_t is not None
        if env_space is None:  # singleton mode
            pass
        else:
            self.space_init(env_space)

        logger.info(util.self_desc(self))

    def patch_gym_spaces(self, u_env):
        '''
        For standardization, use gym spaces to represent observation and action spaces for Unity.
        This method iterates through the multiple brains (multiagent) then constructs and returns lists of observation_spaces and action_spaces
        '''
        observation_spaces = []
        action_spaces = []
        for a in range(len(u_env.brain_names)):
            brain = self._get_brain(u_env, a)
            observation_shape = (brain.get_observable_dim()['state'],)
            if brain.is_discrete():
                dtype = np.int32
                action_space = spaces.Discrete(brain.get_action_dim())
            else:
                dtype = np.float32
                action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=dtype)
            observation_space = spaces.Box(low=0, high=1, shape=observation_shape, dtype=dtype)
            set_gym_space_attr(observation_space)
            set_gym_space_attr(action_space)
            observation_spaces.append(observation_space)
            action_spaces.append(action_space)
        # set for singleton
        u_env.observation_space = observation_spaces[0]
        u_env.action_space = action_spaces[0]
        return observation_spaces, action_spaces

    def _get_brain(self, u_env, a):
        '''Get the unity-equivalent of agent, i.e. brain, to access its info'''
        name_a = u_env.brain_names[a]
        brain_a = u_env.brains[name_a]
        return brain_a

    def _check_u_brain_to_agent(self):
        '''Check the size match between unity brain and agent'''
        u_brain_num = self.u_env.number_brains
        agent_num = len(self.body_e)
        assert u_brain_num == agent_num, f'There must be a Unity brain for each agent. e:{self.e}, brain: {u_brain_num} != agent: {agent_num}.'

    def _check_u_agent_to_body(self, env_info_a, a):
        '''Check the size match between unity agent and body'''
        u_agent_num = len(env_info_a.agents)
        body_num = util.count_nonan(self.body_e[a])
        assert u_agent_num == body_num, f'There must be a Unity agent for each body; a:{a}, e:{self.e}, agent_num: {u_agent_num} != body_num: {body_num}.'

    def _get_env_info(self, env_info_dict, a):
        '''Unity API returns a env_info_dict. Use this method to pull brain(env)-specific usable for lab API'''
        name_a = self.u_env.brain_names[a]
        env_info_a = env_info_dict[name_a]
        return env_info_a

    @lab_api
    def reset(self):
        self.done = False
        env_info_dict = self.u_env.reset(train_mode=(util.get_lab_mode() != 'dev'), config=self.env_spec.get('unity'))
        a, b = 0, 0  # default singleton aeb
        env_info_a = self._get_env_info(env_info_dict, a)
        state = env_info_a.states[b]
        logger.debug(f'Env {self.e} reset state: {state}')
        return state

    @lab_api
    def step(self, action):
        env_info_dict = self.u_env.step(action)
        a, b = 0, 0  # default singleton aeb
        env_info_a = self._get_env_info(env_info_dict, a)
        state = env_info_a.states[b]
        reward = env_info_a.rewards[b] * self.reward_scale
        done = env_info_a.local_done[b]
        self.done = done = done or self.clock.t > self.max_t
        logger.debug(f'Env {self.e} step state: {state}, reward: {reward}, done: {done}')
        return state, reward, done, env_info_a

    @lab_api
    def close(self):
        self.u_env.close()

    # NOTE optional extension for multi-agent-env

    @lab_api
    def space_init(self, env_space):
        '''Post init override for space env. Note that aeb is already correct from __init__'''
        self.env_space = env_space
        self.aeb_space = env_space.aeb_space
        self.observation_spaces = [self.observation_space]
        self.action_spaces = [self.action_space]

    @lab_api
    def space_reset(self):
        self._check_u_brain_to_agent()
        self.done = False
        env_info_dict = self.u_env.reset(train_mode=(util.get_lab_mode() != 'dev'), config=self.env_spec.get('unity'))
        state_e, _reward_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for (a, b), body in util.ndenumerate_nonan(self.body_e):
            env_info_a = self._get_env_info(env_info_dict, a)
            self._check_u_agent_to_body(env_info_a, a)
            state = env_info_a.states[b]
            state_e[(a, b)] = state
            done_e[(a, b)] = self.done
        logger.debug(f'Env {self.e} reset reward_e: {_reward_e}, state_e: {state_e}, done_e: {done_e}')
        return _reward_e, state_e, done_e

    @lab_api
    def space_step(self, action_e):
        # TODO implement clock_speed: step only if self.clock.to_step()
        if self.done:
            _reward_e, state_e, done_e = self.space_reset()
            return state_e, _reward_e, done_e, None
        action_e = util.nanflatten(action_e)
        env_info_dict = self.u_env.step(action_e)
        state_e, reward_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for (a, b), body in util.ndenumerate_nonan(self.body_e):
            env_info_a = self._get_env_info(env_info_dict, a)
            state_e[(a, b)] = env_info_a.states[b]
            reward_e[(a, b)] = env_info_a.rewards[b] * self.reward_scale
            done_e[(a, b)] = env_info_a.local_done[b]
        info_e = env_info_dict
        self.done = (util.nonan_all(done_e) or self.clock.t > self.max_t)
        logger.debug(f'Env {self.e} step state_e: {state_e}, reward_e: {reward_e}, done_e: {done_e}')
        return state_e, reward_e, done_e, info_e
