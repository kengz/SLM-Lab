from gym_unity.envs import UnityEnv
from slm_lab.env.registration import get_env_path
from slm_lab.lib import util
import os
import pydash as ps


class GymUnityEnv(UnityEnv):
    '''Wrapper to make UnityEnv register-able under gym'''
    spec = None

    def __init__(self, name):
        worker_id = int(f'{os.getpid()}{int(ps.unique_id())}'[-4:])
        super().__init__(get_env_path(name), worker_id, no_graphics=not util.to_render())
