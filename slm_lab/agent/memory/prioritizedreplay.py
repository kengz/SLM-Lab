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

    e.g. memory_spec
    "memory": {
        "name": "Replay",
        "alpha": 1,
        "epsilon": 0,
        "batch_size": 32,
        "max_size": 10000
    }
    '''

    def __init__(self, memory_spec, algorithm, body):
        util.set_attr(self, self.memory_spec, [
            'alpha',
            'epsilon',
            'batch_size',
            'max_size',
        ])
        self.epsilon = torch.full((1,), self.epsilon)
        self.alpha = torch.full((1,), self.alpha)
        super(PrioritizedReplay, self).__init__(memory_spec, algorithm, body)

    def reset(self):
        super(PrioritizedReplay, self).reset()
        self.tree = SumTree(self.max_size)

    def add_experience(self, state, action, reward, next_state, done, error=100000):
        '''
        Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.
        '''
        error = torch.full_like((1,), error)
        priority = self.get_priority(error)
        super(PrioritizedReplay, self).add_experience(state, action, reward, next_state, done, priority)
        self.tree.add(priority, self.head)

    def get_priority(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        p = torch.pow(error + self.epsilon, self.alpha)
        return p.numpy()

    def sample_idxs(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.
        Implementation follows the approach in the paper "Prioritized Experience Replay", Schaul et al 2015" and is Jaromír Janisch's with minor adaptations. See memory_utils.py for the license and link to Jaromír's excellent blog'''
        batch_idxs = np.zeros(batch_size)
        tree_idxs = np.zeros(batch_size)
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx

        batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs = tree_idxs
        return batch_idxs

    def get_body_errors(self, errors):
        '''Get the slice of errors belonging to a body in network output'''
        body_idx = self.body.nanflat_a_idx
        start_idx = body_idx * self.batch_size
        end_idx = start_idx + self.batch_size
        body_errors = errors[start_idx:end_idx]
        return body_errors

    def update_priorities(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        body_errors = self.get_body_errors(errors)
        priorities = self.get_priority(body_errors)
        super(PrioritizedReplay, self).update_priorities(priorities)
        for p, i in zip(priorities, self.tree_idxs):
            self.tree.update(i, p)
