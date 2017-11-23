'''
The random agent algorithm
For basic dev purpose.
'''
import numpy as np
from slm_lab.agent.algorithm.base import Algorithm

# TODO make as proper to the module design, this shd handle solely just the action, policy, using net for architecture and training
# then in Agent class instantiation, the below should supplement


class Random(Algorithm):
    '''
    Example Random agent that works in both discrete and continuous envs
    '''

    def act_discrete(self, state):
        '''Random discrete action'''
        # TODO AEB space resolver, lineup index
        action = np.random.randint(
            0, self.agent.env.get_action_dim(), size=(self.agent.body_num))
        return action

    def act_continuous(self, state):
        '''Random continuous action'''
        # TODO AEB space resolver, lineup index
        action = np.random.randn(
            self.agent.body_num, self.agent.env.get_action_dim())
        return action

    def update(self):
        return
