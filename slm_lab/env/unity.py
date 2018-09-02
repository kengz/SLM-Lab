from slm_lab.env import ENV_DATA_NAMES
from slm_lab.env.base import BaseEnv, Clock, ENV_DATA_NAMES
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from unityagents import brain, UnityEnvironment
import numpy as np
import os

logger = logger.get_logger(__name__)


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


class UnityEnv(BaseEnv):
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

    # @lab_api
    # def space_init(self, env_space):
    #     '''Post init override for space env. Note that aeb is already correct from __init__'''
    #     self.observation_spaces = [self.observation_space]
    #     self.action_spaces = [self.action_space]
    #     self.env_space = env_space
    #     self.check_u_brain_to_agent()
    #     logger.info(util.self_desc(self))

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
