from collections import deque
from copy import deepcopy
from slm_lab.agent.memory.base import Memory
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps

logger = logger.get_logger(__name__)


def sample_next_states(head, max_size, ns_idx_offset, batch_idxs, states, ns_buffer):
    '''Method to sample next_states from states, with proper guard for next_state idx being out of bound'''
    # idxs for next state is state idxs with offset, modded
    ns_batch_idxs = (batch_idxs + ns_idx_offset) % max_size
    # if head < ns_idx <= head + ns_idx_offset, ns is stored in ns_buffer
    ns_batch_idxs = ns_batch_idxs % max_size
    buffer_ns_locs = np.argwhere(
        (head < ns_batch_idxs) & (ns_batch_idxs <= head + ns_idx_offset)).flatten()
    # find if there is any idxs to get from buffer
    to_replace = buffer_ns_locs.size != 0
    if to_replace:
        # extract the buffer_idxs first for replacement later
        # given head < ns_idx <= head + offset, and valid buffer idx is [0, offset)
        # get 0 < ns_idx - head <= offset, or equiv.
        # get -1 < ns_idx - head - 1 <= offset - 1, i.e.
        # get 0 <= ns_idx - head - 1 < offset, hence:
        buffer_idxs = ns_batch_idxs[buffer_ns_locs] - head - 1
        # set them to 0 first to allow sampling, then replace later with buffer
        ns_batch_idxs[buffer_ns_locs] = 0
    # guard all against overrun idxs from offset
    ns_batch_idxs = ns_batch_idxs % max_size
    next_states = util.cond_multiget(states, ns_batch_idxs)
    if to_replace:
        # now replace using buffer_idxs and ns_buffer
        buffer_ns = util.cond_multiget(ns_buffer, buffer_idxs)
        next_states[buffer_ns_locs] = buffer_ns
    return next_states


class Replay(Memory):
    '''
    Stores agent experiences and samples from them for agent training

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
        - This is implemented as a circular buffer so that inserting experiences are O(1)
        - Each element of an experience is stored as a separate array of size N * element dim

    When a batch of experiences is requested, K experiences are sampled according to a random uniform distribution.

    If 'use_cer', sampling will add the latest experience.

    e.g. memory_spec
    "memory": {
        "name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''

    def __init__(self, memory_spec, body):
        super().__init__(memory_spec, body)
        util.set_attr(self, self.memory_spec, [
            'batch_size',
            'max_size',
            'use_cer',
        ])
        self.is_episodic = False
        self.batch_idxs = None
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        self.head = -1  # index of most recent experience
        # generic next_state buffer to store last next_states (allow for multiple for venv)
        self.ns_idx_offset = self.body.env.num_envs if body.env.is_venv else 1
        self.ns_buffer = deque(maxlen=self.ns_idx_offset)
        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.reset()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        # set self.states, self.actions, ...
        for k in self.data_keys:
            if k != 'next_states':  # reuse self.states
                # list add/sample is over 10x faster than np, also simpler to handle
                setattr(self, k, [None] * self.max_size)
        self.size = 0
        self.head = -1
        self.ns_buffer.clear()

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Interface method to update memory'''
        if self.body.env.is_venv:
            for sarsd in zip(state, action, reward, next_state, done):
                self.add_experience(*sarsd)
        else:
            self.add_experience(state, action, reward, next_state, done)

    def add_experience(self, state, action, reward, next_state, done):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        self.states[self.head] = state.astype(np.float16)
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.ns_buffer.append(next_state.astype(np.float16))
        self.dones[self.head] = done
        # Actually occupied size of memory
        if self.size < self.max_size:
            self.size += 1
        self.seen_size += 1
        # set to_train using memory counters head, seen_size instead of tick since clock will step by num_envs when on venv; to_train will be set to 0 after training step
        algorithm = self.body.agent.algorithm
        algorithm.to_train = algorithm.to_train or (self.seen_size > algorithm.training_start_step and self.head % algorithm.training_frequency == 0)

    @lab_api
    def sample(self):
        '''
        Returns a batch of batch_size samples. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the corresponding sampled elements
        e.g.
        batch = {
            'states'     : states,
            'actions'    : actions,
            'rewards'    : rewards,
            'next_states': next_states,
            'dones'      : dones}
        '''
        self.batch_idxs = self.sample_idxs(self.batch_size)
        batch = {}
        for k in self.data_keys:
            if k == 'next_states':
                batch[k] = sample_next_states(self.head, self.max_size, self.ns_idx_offset, self.batch_idxs, self.states, self.ns_buffer)
            else:
                batch[k] = util.cond_multiget(getattr(self, k), self.batch_idxs)
        return batch

    def sample_idxs(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        batch_idxs = np.random.randint(self.size, size=batch_size)
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.head
        return batch_idxs


class AtariReplay(Replay):
    '''
    Preprocesses an state to be the concatenation of the last four states, after converting the 210 x 160 x 3 image to 84 x 84 x 1 grayscale image, and clips all rewards to [-10, 10] as per "Playing Atari with Deep Reinforcement Learning", Mnih et al, 2013
    Note: Playing Atari with Deep RL clips the rewards to + / - 1
    '''

    def add_experience(self, state, action, reward, next_state, done):
        # clip reward, done here to minimize change to only training data data
        super().add_experience(state, action, np.sign(reward), next_state, done)
