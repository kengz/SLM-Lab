'''
The random agent algorithm
For basic dev purpose.
'''
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib import logger
from slm_lab.lib.decorator import lab_api
import numpy as np

logger = logger.get_logger(__name__)


class Random(Algorithm):
    '''
    Example Random agent that works in both discrete and continuous envs
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        self.to_train = 0
        self.training_frequency = 1
        self.training_start_step = 0

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network from the spec'''
        self.net_names = []

    @lab_api
    def act(self, state):
        '''Random action'''
        body = self.body
        if body.env.is_venv and not util.in_eval_lab_modes():
            action = np.array([body.action_space.sample() for _ in range(body.env.num_envs)])
        else:
            action = body.action_space.sample()
        return action

    @lab_api
    def sample(self):
        self.body.memory.sample()
        batch = np.nan
        return batch

    @lab_api
    def train(self):
        self.sample()
        loss = np.nan
        return loss

    @lab_api
    def update(self):
        self.body.explore_var = np.nan
        return self.body.explore_var
