'''
The random agent algorithm
For basic dev purpose.
'''
from slm_lab.agent.algorithm.base import Algorithm
import numpy as np


class Random(Algorithm):
    '''
    Example Random agent that works in both discrete and continuous envs
    '''

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        pass

    def body_act_discrete(self, body, body_state):
        '''Random discrete action'''
        body_action = np.random.randint(body.action_dim)
        return body_action

    def body_act_continuous(self, body, body_state):
        '''Random continuous action'''
        body_action = np.random.randn(body.action_dim)
        return body_action

    def train(self):
        return

    def update(self):
        return
