from slm_lab.agent.memory.replay import Replay, AtariReplay
from slm_lab.lib import util
from slm_lab.lib.decorator import lab_api
import numpy as np
import random
import torch


class SumTree:
    '''
    Helper class for PrioritizedReplay

    This implementation is, with minor adaptations, Jaromír Janisch's. The license is reproduced below.
    For more information see his excellent blog series "Let's make a DQN" https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

    MIT License

    Copyright (c) 2018 Jaromír Janisch

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.indices[indexIdx])

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Prio: {self.tree[j]}')


class PrioritizedReplay(Replay):
    '''
    Prioritized Experience Replay

    Implementation follows the approach in the paper "Prioritized Experience Replay", Schaul et al 2015" https://arxiv.org/pdf/1511.05952.pdf and is Jaromír Janisch's with minor adaptations.
    See memory_util.py for the license and link to Jaromír's excellent blog

    Stores agent experiences and samples from them for agent training according to each experience's priority

    The memory has the same behaviour and storage structure as Replay memory with the addition of a SumTree to store and sample the priorities.

    e.g. memory_spec
    "memory": {
        "name": "PrioritizedReplay",
        "alpha": 1,
        "epsilon": 0,
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''

    def __init__(self, memory_spec, body):
        util.set_attr(self, memory_spec, [
            'alpha',
            'epsilon',
            'batch_size',
            'max_size',
            'use_cer',
        ])
        super(PrioritizedReplay, self).__init__(memory_spec, body)

        self.epsilon = torch.full((1,), self.epsilon)
        self.alpha = torch.full((1,), self.alpha)
        # adds a 'priorities' scalar to the data_keys and call reset again
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']
        self.reset()

    def reset(self):
        super(PrioritizedReplay, self).reset()
        self.tree = SumTree(self.max_size)

    def add_experience(self, state, action, reward, next_state, done, error=100000):
        '''
        Implementation for update() to add experience to memory, expanding the memory size if necessary.
        All experiences are added with a high priority to increase the likelihood that they are sampled at least once.
        '''
        super(PrioritizedReplay, self).add_experience(state, action, reward, next_state, done)
        error = torch.zeros(1).fill_(error)
        priority = self.get_priority(error)
        self.priorities[self.head] = priority
        self.tree.add(priority, self.head)

    def get_priority(self, error):
        '''Takes in the error of one or more examples and returns the proportional priority'''
        p = torch.pow(error.cpu().detach() + self.epsilon, self.alpha)
        return p.squeeze().detach().numpy()

    def sample_idxs(self, batch_size):
        '''Samples batch_size indices from memory in proportional to their priority.'''
        batch_idxs = np.zeros(batch_size)
        tree_idxs = np.zeros(batch_size, dtype=np.int)

        for i in range(batch_size):
            s = random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx

        batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs = tree_idxs
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.head
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
        assert len(priorities) == self.batch_idxs.size
        self.priorities[self.batch_idxs] = priorities
        for p, i in zip(priorities, self.tree_idxs):
            self.tree.update(i, p)


class AtariPrioritizedReplay(PrioritizedReplay, AtariReplay):
    '''Make a Prioritized AtariReplay via nice multi-inheritance (python magic)'''
    pass
