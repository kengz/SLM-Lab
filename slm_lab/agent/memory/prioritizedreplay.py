from slm_lab.agent.memory.replay import Replay
from slm_lab.agent.memory.memory_utils import SumTree
from slm_lab.lib import util
from slm_lab.lib.decorator import lab_api
import torch
import numpy as np
import pydash as _
import random


class PrioritizedReplay(Replay):
    '''
    Stores agent experiences and samples from them for agent training according to each experience's priority

    The memory has the same behaviour and storage structure as Replay memory with the addition of a SumTree to store and sample the priorities.
    '''
    def __init__(self, body):
        super(PrioritizedReplay, self).__init__(body)
        self.e = self.body.agent.spec['memory']['e']
        self.e = torch.zeros(1).fill_(self.e)
        self.alpha = self.body.agent.spec['memory']['alpha']
        self.alpha = torch.zeros(1).fill_(self.alpha)

    def reset(self):
        super(PrioritizedReplay, self).reset()
        self.tree = SumTree(self.max_size)

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        super(PrioritizedReplay, self).update(action, reward, state, done)

    def add_experience(self, state, action, reward, next_state, done, error=100000):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.'''
        error = torch.zeros(1).fill_(error)
        priority = self.get_priority(error)
        super(PrioritizedReplay, self).add_experience(state, action, reward, next_state, done, priority)
        self.tree.add(priority, self.head)

    def get_priority(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        p = torch.pow(error + self.e, self.alpha)
        return p.numpy()

    def sample_idxs(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.
        Implementation follows the approach in the paper "Prioritized Experience Replay", Schaul et al 2015" and is Jaromír Janisch's with minor adaptations. See memory_utils.py for the license and link to Jaromír's excellent blog'''
        batch_idxs = []
        tree_idxs = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs.append(idx)
            tree_idxs.append(tree_idx)

        batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs = tree_idxs
        return batch_idxs

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        priorities = self.get_priority(errors)
        super(PrioritizedReplay, self).update_priorities(priorities)
        for p, i in zip(priorities, self.tree_idxs):
            self.tree.update(i, p)
