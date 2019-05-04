from slm_lab.env.base import BaseEnv, ENV_DATA_NAMES
from slm_lab.env.wrapper import make_gym_env
from slm_lab.env.vec_env import make_gym_venv
from slm_lab.env.registration import try_register_env
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import gym
import numpy as np
import pydash as ps
import roboschool


logger = logger.get_logger(__name__)


class OpenAIEnv(BaseEnv):
    '''
    Wrapper for OpenAI Gym env to work with the Lab.

    e.g. env_spec
    "env": [{
      "name": "CartPole-v0",
      "num_envs": null,
      "max_t": null,
      "max_tick": 10000,
    }],
    '''

    def __init__(self, spec, e=None, env_space=None):
        super().__init__(spec, e, env_space)
        try_register_env(spec)  # register if it's a custom gym env
        seed = ps.get(spec, 'meta.random_seed')
        stack_len = ps.get(spec, 'agent.0.memory.stack_len')
        if self.is_venv:  # make vector environment
            self.u_env = make_gym_venv(self.name, seed, stack_len, self.num_envs)
        else:
            self.u_env = make_gym_env(self.name, seed, stack_len)
        self._set_attr_from_u_env(self.u_env)
        self.max_t = self.max_t or self.u_env.spec.max_episode_steps
        assert self.max_t is not None
        if env_space is None:  # singleton mode
            pass
        else:
            self.space_init(env_space)
        logger.info(util.self_desc(self))

    @lab_api
    def reset(self):
        self.done = False
        state = self.u_env.reset()
        if self.to_render:
            self.u_env.render()
        return state

    @lab_api
    def step(self, action):
        if not self.is_discrete and self.action_dim == 1:  # guard for continuous with action_dim 1, make array
            action = np.expand_dims(action, axis=-1)
        state, reward, done, info = self.u_env.step(action)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        if self.to_render:
            self.u_env.render()
        if not self.is_venv and self.clock.t > self.max_t:
            done = True
        self.done = done
        return state, reward, done, info

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
        self.done = False
        state_e, = self.env_space.aeb_space.init_data_s(['state'], e=self.e)
        for ab, body in util.ndenumerate_nonan(self.body_e):
            state = self.u_env.reset()
            state_e[ab] = state
        if self.to_render:
            self.u_env.render()
        return state_e

    @lab_api
    def space_step(self, action_e):
        action = action_e[(0, 0)]  # single body
        if self.done:  # space envs run continually without a central reset signal
            state_e = self.space_reset()
            _reward_e, done_e = self.env_space.aeb_space.init_data_s(['reward', 'done'], e=self.e)
            return state_e, _reward_e, done_e, None
        if not self.is_discrete and self.action_dim == 1:  # guard for continuous with action_dim 1, make array
            action = np.expand_dims(action, axis=-1)
        state, reward, done, info = self.u_env.step(action)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        if self.to_render:
            self.u_env.render()
        if not self.is_venv and self.clock.t > self.max_t:
            done = True
        self.done = done
        state_e, reward_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for ab, body in util.ndenumerate_nonan(self.body_e):
            state_e[ab] = state
            reward_e[ab] = reward
            done_e[ab] = done
        info_e = info
        return state_e, reward_e, done_e, info_e
