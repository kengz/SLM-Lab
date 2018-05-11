'''
The random agent algorithm
For basic dev purpose.
'''
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib.decorator import lab_api
import numpy as np


class Random(Algorithm):
    '''
    Example Random agent that works in both discrete and continuous envs
    '''

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        pass

    @lab_api
    def init_nets(self):
        '''Initialize the neural network from the spec'''
        pass

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        pass

    @lab_api
    def body_act_discrete(self, body, state):
        '''Random discrete action'''
        action = np.random.randint(body.action_dim)
        return action

    @lab_api
    def body_act_continuous(self, body, state):
        '''Random continuous action'''
        action = np.random.randn(body.action_dim)
        return action

    @lab_api
    def sample(self):
        batch = np.nan
        return batch

    @lab_api
    def train(self):
        self.sample()
        loss = np.nan
        return loss

    @lab_api
    def update(self):
        explore_var = np.nan
        return explore_var
