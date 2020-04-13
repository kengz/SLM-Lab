from slm_lab.env.base import BaseEnv
from slm_lab.env.wrapper import make_gym_env
from slm_lab.env.vec_env import make_gym_venv
from slm_lab.env.registration import try_register_env
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import gym
import numpy as np
import pydash as ps
# import roboschool


logger = logger.get_logger(__name__)


class OpenAIEnv(BaseEnv):
    '''
    Wrapper for OpenAI Gym env to work with the Lab.

    e.g. env_spec
    "env": [{
        "name": "PongNoFrameskip-v4",
        "frame_op": "concat",
        "frame_op_len": 4,
        "normalize_state": false,
        "reward_scale": "sign",
        "num_envs": 8,
        "max_t": null,
        "max_frame": 1e7
    }],
    '''

    def __init__(self, spec):
        super().__init__(spec)
        try_register_env(spec)  # register if it's a custom gym env
        seed = ps.get(spec, 'meta.random_seed')
        episode_life = not util.in_eval_lab_modes()
        if self.is_venv:  # make vector environment
            self.u_env = make_gym_venv(name=self.name, num_envs=self.num_envs, seed=seed, frame_op=self.frame_op, frame_op_len=self.frame_op_len, image_downsize=self.image_downsize, reward_scale=self.reward_scale, normalize_state=self.normalize_state, episode_life=episode_life)
        else:
            self.u_env = make_gym_env(name=self.name, seed=seed, frame_op=self.frame_op, frame_op_len=self.frame_op_len, image_downsize=self.image_downsize, reward_scale=self.reward_scale, normalize_state=self.normalize_state, episode_life=episode_life)
        self.NUM_AGENTS = self.u_env.NUM_AGENTS if hasattr(self.u_env, "NUM_AGENTS") else 1
        if self.name.startswith('Unity'):
            # Unity is always initialized as singleton gym env, but the Unity runtime can be vec_env
            self.num_envs = self.u_env.num_envs
            # update variables dependent on num_envs
            self._infer_venv_attr()
            self._set_clock()
        self._set_attr_from_u_env(self.u_env)
        self.max_t = self.max_t or self.u_env.spec.max_episode_steps
        assert self.max_t is not None

        self.extra_env_log_info_col = self.get_extra_training_log_info().keys()

        logger.info(util.self_desc(self))

    def seed(self, seed):
        print("OpenAIEnv set seed", seed)
        self.u_env.seed(seed)

    @lab_api
    def reset(self):
        self.done = False
        state = self.u_env.reset()

        state = self._convert_discrete_state_to_one_hot_numpy(state)

        if self.to_render:
            try:
                self.u_env.render()
            except NotImplementedError:
                logger.warning("env.render method is not implemented")
                self.to_render = False


        if self.NUM_AGENTS == 1:  # Adapt to env with single agent
            state = [state]

        return state

    @lab_api
    def step(self, action):

        if self.NUM_AGENTS == 1:  # Support env with single agent
            action = action[0]

        if not self.action_space_is_discrete and self.action_dim == 1:  # guard for continuous with action_dim 1, make array
            action = np.expand_dims(action, axis=-1)
        state, reward, done, info = self.u_env.step(action)

        if isinstance(info, dict) and 'extra_info_to_log' in info.keys():
            self.extra_info_to_log = info['extra_info_to_log']

        state = self._convert_discrete_state_to_one_hot_numpy(state)

        self._update_total_reward(info)
        if self.to_render:
            try:
                self.u_env.render()
            except NotImplementedError:
                logger.warning("env.render method is not implemented")
                self.to_render = False
        if not self.is_venv and self.clock.t > self.max_t:
            done = True
        self.done = done

        if self.NUM_AGENTS == 1:  # Support env with single agent
            state = [state]
            reward = [reward]

        # logger.info(f"state {state} {type(state)}")
        # logger.info(f"reward {reward} {type(reward)}")
        # logger.info(f"info {info}")

        return state, reward, done, info

    def _convert_discrete_state_to_one_hot_numpy(self, state):
        if self.observation_space_is_discrete:
            def to_one_hot(array, n_values):
                return np.eye(n_values)[array.astype(np.int)]

            # TODO check support MultiDiscrete env
            assert len(self.observable_dim) ==1
            observation_space = self.observable_dim[0]

            state = np.array(state)
            state = to_one_hot(state, observation_space)
        return state

    @lab_api
    def close(self):
        self.u_env.close()


    def _is_discrete(self, space):
        '''Check if an space is discrete'''
        is_a_multi_agent_env = True if hasattr(self.u_env, "NUM_AGENTS") and self.u_env.NUM_AGENTS > 1 else False
        space = space[0] if is_a_multi_agent_env else space

        # logger.debug("util.get_class_name(space) {}".format(util.get_class_name(space)))
        # return util.get_class_name(space) != 'Box'
        return "Discrete" in util.get_class_name(space)

    def get_extra_training_log_info(self):
        if hasattr(self, "extra_info_to_log"):
            return self.extra_info_to_log
        else:
            return {}