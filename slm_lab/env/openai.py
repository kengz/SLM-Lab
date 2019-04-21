from slm_lab.env.base import BaseEnv, ENV_DATA_NAMES
from slm_lab.env.wrapper import make_gym_env
from slm_lab.env.vec_env import make_gym_venv
from slm_lab.env.registration import register_env
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import gym
import numpy as np
import pydash as ps

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
        super(OpenAIEnv, self).__init__(spec, e, env_space)
        try:
            # register any additional environments first. guard for re-registration
            register_env(spec)
        except Exception as e:
            pass
        seed = ps.get(spec, 'meta.random_seed')
        stack_len = ps.get(spec, 'agent.0.memory.stack_len')
        num_envs = ps.get(spec, f'env.{self.e}.num_envs')
        if num_envs is None:
            self.u_env = make_gym_env(self.name, seed, stack_len)
        else:  # make vector environment
            self.u_env = make_gym_venv(self.name, seed, stack_len, num_envs)
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
        _reward = np.nan
        state = self.u_env.reset()
        self.done = done = False
        if util.to_render():
            self.u_env.render()
        logger.debug(f'Env {self.e} reset reward: {_reward}, state: {state}, done: {done}')
        return _reward, state, done

    @lab_api
    def step(self, action):
        if not self.is_discrete:  # guard for continuous
            action = np.array([action])
        state, reward, done, info = self.u_env.step(action)
        reward *= self.reward_scale
        if util.to_render():
            self.u_env.render()
        done = done or self.clock.t > self.max_t
        self.done = done
        logger.debug(f'Env {self.e} step state: {state}, reward: {reward}, done: {done}')
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
        state_e, _reward_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for ab, body in util.ndenumerate_nonan(self.body_e):
            state = self.u_env.reset()
            state_e[ab] = state
            done_e[ab] = self.done = False
        if util.to_render():
            self.u_env.render()
        logger.debug(f'Env {self.e} reset reward_e: {_reward_e}, state_e: {state_e}, done_e: {done_e}')
        return _reward_e, state_e, done_e

    @lab_api
    def space_step(self, action_e):
        action = action_e[(0, 0)]  # single body
        if self.done:  # space envs run continually without a central reset signal
            _reward_e, state_e, done_e = self.space_reset()
            return state_e, _reward_e, done_e, None
        if not self.is_discrete:
            action = np.array([action])
        state, reward, done, info = self.u_env.step(action)
        reward *= self.reward_scale
        if util.to_render():
            self.u_env.render()
        self.done = done = done or self.clock.t > self.max_t
        state_e, reward_e, done_e = self.env_space.aeb_space.init_data_s(ENV_DATA_NAMES, e=self.e)
        for ab, body in util.ndenumerate_nonan(self.body_e):
            state_e[ab] = state
            reward_e[ab] = reward
            done_e[ab] = done
        info_e = info
        logger.debug(f'Env {self.e} step state_e: {state_e}, reward_e: {reward_e}, done_e: {done_e}')
        return state_e, reward_e, done_e, info_e
