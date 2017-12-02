'''
The random agent algorithm
For basic dev purpose.
'''
import numpy as np
from slm_lab.agent.algorithm.base import Algorithm


class Random(Algorithm):
    '''
    Example Random agent that works in both discrete and continuous envs
    '''

    def body_act_discrete(self, body, body_state):
        '''Random discrete action'''
        body_action = np.random.randint(body.action_dim)
        return body_action

    def body_act_continuous(self, body, body_state):
        '''Random continuous action'''
        body_action = np.random.randn(body.action_dim)
        return body_action

    def update(self, action, reward, state, done):
        return
