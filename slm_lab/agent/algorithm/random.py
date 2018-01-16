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
        self.flat_nonan_explore_var_a = [
            np.nan] * len(self.agent.flat_nonan_body_a)

    def body_act_discrete(self, body, state):
        '''Random discrete action'''
        action = np.random.randint(body.action_dim)
        return action

    def body_act_continuous(self, body, state):
        '''Random continuous action'''
        action = np.random.randn(body.action_dim)
        return action

    def train(self):
        loss = np.nan
        return loss

    def update(self):
        return self.flat_nonan_explore_var_a
