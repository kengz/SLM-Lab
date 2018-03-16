from slm_lab.agent.memory.replay import Replay
from slm_lab.agent.memory.memory_utils import SumTree
from slm_lab.lib import util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as _


class PrioritizedReplay(Replay):
    '''
    Stores agent experiences and samples from them for agent training according to each experience's priority

    The memory has the same behaviour and storage structure as Replay memory with the addition of a SumTree to store and sample the priorities.
    '''
    def __init__(self, body):
        super(PrioritizedReplay, self).__init__(body)

    def reset(self):
        super(PrioritizedReplay, self).reset()
        self.tree = SumTree(self.max_size)

    @lab_api
    def update(self, action, reward, state, done, priority):
        '''Interface method to update memory'''
        # TODO
        pass

    def add_experience(self, state, action, reward, next_state, done, priority=1):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        super(PrioritizedReplay, self).add_experience(state, action, reward, next_state, done, priority)
        # TODO
        pass

    def sample_idxs(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        # TODO
        pass

    def update_priorities(self, priorities):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        super(PrioritizedReplay, self).update_priorities(priorities)
        # TODO
        pass
