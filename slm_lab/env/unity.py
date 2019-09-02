from gym_unity.envs import UnityEnv
from slm_lab.env.registration import get_env_path
from slm_lab.lib import util
import numpy as np
import os
import pydash as ps

# NOTE: stack-frames used in ml-agents:
# 3DBallHard 9
# Hallways 3
# PushBlock 3
# Walker 5


class GymUnityEnv(UnityEnv):
    '''Wrapper to make UnityEnv register-able under gym'''
    spec = None

    def __init__(self, name):
        worker_id = int(f'{os.getpid()}{int(ps.unique_id())}'[-4:])
        super().__init__(get_env_path(name), worker_id, no_graphics=not util.to_render(), multiagent=True)
        self.num_envs = self.number_agents

    def reset(self):
        state = super().reset()
        # Unity returns list, we need array
        return np.array(state)

    def step(self, action):
        # Unity wants list instead of numpy
        action = list(action)
        state, reward, done, info = super().step(action)
        # Unity returns list, we need array
        state = np.array(state)
        reward = np.array(reward)
        done = np.array(done)
        return state, reward, done, info

    def close(self):
        try:  # guard repeated call to close()
            super().close()
        except Exception as e:
            pass
