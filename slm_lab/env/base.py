from abc import ABC, abstractmethod
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
import pydash as ps
import time

logger = logger.get_logger(__name__)


# Removed set_gym_space_attr - use gymnasium space attributes directly


class Clock:
    '''Clock class for each env and space to keep track of relative time. Ticking and control loop is such that reset is at t=0 and epi=0'''

    def __init__(self, max_frame: int = int(1e7), clock_speed: int = 1):
        self.max_frame = max_frame
        self.clock_speed = int(clock_speed)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.frame = 0  # i.e. total_t
        self.epi = 0
        self.start_wall_t = time.time()
        self.wall_t = 0
        self.batch_size = 1  # multiplier to accurately count opt steps
        self.opt_step = 0  # count the number of optimizer updates

    def load(self, train_df: pd.DataFrame) -> None:
        '''Load clock from the last row of body.train_df'''
        last_row = train_df.iloc[-1]
        last_clock_vals = ps.pick(last_row, *['epi', 't', 'wall_t', 'opt_step', 'frame'])
        util.set_attr(self, last_clock_vals)
        self.start_wall_t -= self.wall_t  # offset elapsed wall_t

    def get(self, unit: str = 'frame') -> int:
        return getattr(self, unit)

    def get_elapsed_wall_t(self) -> int:
        '''Calculate the elapsed wall time (int seconds) since self.start_wall_t'''
        return int(time.time() - self.start_wall_t)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def tick(self, unit: str = 't') -> None:
        if unit == 't':  # timestep
            self.t += self.clock_speed
            self.frame += self.clock_speed
            self.wall_t = self.get_elapsed_wall_t()
        elif unit == 'epi':  # episode, reset timestep
            self.epi += 1
            self.t = 0
        elif unit == 'opt_step':
            self.opt_step += self.batch_size
        else:
            raise KeyError


class BaseEnv(ABC):
    '''
    The base Env class with API and helper methods. Use this to implement your env class that is compatible with the Lab APIs

    e.g. env_spec
    "env": [{
        "name": "PongNoFrameskip-v4",
        "num_envs": 8,
        "max_t": null,
        "max_frame": 1e7
    }],
    '''

    def __init__(self, spec: dict[str, Any]):
        self.env_spec = spec['env'][0]  # idx 0 for single-env
        # set default
        util.set_attr(self, dict(
            eval_frequency=10000,
            log_frequency=10000,
            num_envs=1,
        ))
        util.set_attr(self, spec['meta'], [
            'eval_frequency',
            'log_frequency',
        ])
        util.set_attr(self, self.env_spec, [
            'name',
            'num_envs',
            'max_t',
            'max_frame',
        ])
        if util.get_lab_mode() == 'eval':  # override if env is for eval
            self.num_envs = ps.get(spec, 'meta.rigorous_eval')
        self._infer_frame_attr(spec)
        self._infer_venv_attr()
        self._set_clock()
        self.done = False
        self.total_reward = np.nan


    def _infer_frame_attr(self, spec: dict[str, Any]) -> None:
        '''Infer frame attributes'''
        if spec['meta']['distributed'] != False:  # divide max_frame for distributed
            self.max_frame = int(self.max_frame / spec['meta']['max_session'])

    def _infer_venv_attr(self) -> None:
        '''Infer vectorized env attributes'''
        self.is_venv = (self.num_envs is not None and self.num_envs > 1)


    def _set_clock(self) -> None:
        self.clock_speed = 1 * (self.num_envs or 1)  # tick with a multiple of num_envs to properly count frames
        self.clock = Clock(self.max_frame, self.clock_speed)

    def _set_attr_from_u_env(self, u_env: Union['gym.Env', 'gym.vector.VectorEnv']) -> None:
        '''Set the observation, action dimensions and action type from u_env'''
        # Set canonical spaces - use single env spaces if vector env
        if isinstance(u_env, VectorEnv):
            self.observation_space = u_env.single_observation_space
            self.action_space = u_env.single_action_space
        else:
            self.observation_space = u_env.observation_space  
            self.action_space = u_env.action_space
            
        # Extract state dimension from canonical observation space
        if isinstance(self.observation_space, spaces.Box):
            self.state_dim = self.observation_space.shape[0] if len(self.observation_space.shape) == 1 else self.observation_space.shape
        else:
            self.state_dim = getattr(self.observation_space, 'n', self.observation_space.shape)
            
        # Extract action properties from canonical action space
        if isinstance(self.action_space, spaces.Discrete):
            self.action_dim = self.action_space.n
            self.is_discrete = True
            self.is_multi = False
        elif isinstance(self.action_space, spaces.Box):
            self.action_dim = self.action_space.shape[0] if len(self.action_space.shape) == 1 else self.action_space.shape
            self.is_discrete = False
            self.is_multi = len(self.action_space.shape) > 1 or self.action_space.shape[0] > 1
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.action_dim = self.action_space.nvec.tolist()
            self.is_discrete = True
            self.is_multi = True
        elif isinstance(self.action_space, spaces.MultiBinary):
            self.action_dim = self.action_space.n
            self.is_discrete = True
            self.is_multi = True
        else:
            raise NotImplementedError(f'Action space type {type(self.action_space)} not supported')
            

    def _update_total_reward(self, info: dict[str, Any]) -> None:
        '''Extract total_reward from info (set in wrapper) into self.total_reward for single and vec env'''
        if isinstance(info, dict):
            self.total_reward = info['total_reward']
        else:  # vec env tuple of infos
            self.total_reward = np.array([i.get('total_reward', np.nan) for i in info])

    @abstractmethod
    @lab_api
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict[str, Any]]:
        '''Reset method, return state, info'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def step(self, action: np.ndarray) -> tuple[np.ndarray, Union[float, np.ndarray], Union[bool, np.ndarray], Union[bool, np.ndarray], dict[str, Any]]:
        '''Step method, return state, reward, term, trunct, info'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def close(self) -> None:
        '''Method to close and cleanup env'''
        raise NotImplementedError
