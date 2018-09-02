'''
The environment module
Contains graduated components from experiments for building/using environment.
Provides the rich experience for agent embodiment, reflects the curriculum and allows teaching (possibly allows teacher to enter).
To be designed by human and evolution module, based on the curriculum and fitness metrics.
'''
from slm_lab.env.base import Clock, ENV_DATA_NAMES
from slm_lab.env.openai import OpenAIEnv
from slm_lab.env.unity import UnityEnv
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import pydash as ps


logger = logger.get_logger(__name__)


def make_env(spec, e=None, env_space=None):
    try:
        env = OpenAIEnv(spec, e, env_space)
    except Exception:
        env = UnityEnv(spec, e, env_space)
    return env


class EnvSpace:
    '''
    Subspace of AEBSpace, collection of all envs, with interface to Session logic; same methods as singleton envs.
    Access AgentSpace properties by: AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, aeb_space):
        self.spec = spec
        self.aeb_space = aeb_space
        aeb_space.env_space = self
        self.info_space = aeb_space.info_space
        self.envs = []
        for e in range(len(self.spec['env'])):
            env = make_env(self.spec, e, env_space=self)
            self.envs.append(env)
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
            _reward_e, state_e, done_e = env.space_reset()
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
            reward_e, state_e, done_e = env.space_step(action_e)
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
