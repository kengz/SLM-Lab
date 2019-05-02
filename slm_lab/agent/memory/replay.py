from collections import deque
from copy import deepcopy
from slm_lab.agent.memory.base import Memory
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps

logger = logger.get_logger(__name__)


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
        super(Replay, self).__init__(memory_spec, body)
        util.set_attr(self, self.memory_spec, [
            'batch_size',
            'max_size',
            'use_cer',
        ])
        self.state_buffer = deque(maxlen=0)  # for API consistency
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
        self.state_buffer.clear()
        self.ns_buffer.clear()
        for _ in range(self.state_buffer.maxlen):
            self.state_buffer.append(np.zeros(self.body.state_dim))

    def epi_reset(self, state):
        '''Method to reset at new episode'''
        super(Replay, self).epi_reset(self.preprocess_state(state, append=False))

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Interface method to update memory'''
        if not self.body.env.is_venv and np.isnan(reward):  # start of episode (venv is not episodic)
            self.epi_reset(next_state)
        else:
            # prevent conflict with preprocess in epi_reset
            state = self.preprocess_state(state, append=False)
            next_state = self.preprocess_state(next_state, append=False)
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
        # set to_train
        tick = self.body.env.clock.get()
        algorithm = self.body.agent.algorithm
        # set to self to handle venv stepping multiple ticks; to_train will be set to 0 after training step
        # TODO This is unsafe
        algorithm.to_train = algorithm.to_train or (tick > algorithm.training_start_step and self.head % algorithm.training_frequency == 0)

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
                batch[k] = self._sample_next_states(self.batch_idxs)
            else:
                batch[k] = util.cond_multiget(getattr(self, k), self.batch_idxs)
        return batch

    def _sample_next_states(self, batch_idxs):
        '''Method to sample next_states from states, with proper guard for next_state idx being out of bound'''
        # idxs for next state is state idxs with offset
        ns_batch_idxs = batch_idxs + self.ns_idx_offset
        # if head < ns_idx <= head + ns_idx_offset, ns is stored in self.ns_buffer
        buffer_ns_locs = np.argwhere(
            (self.head < ns_batch_idxs) & (ns_batch_idxs <= self.head + self.ns_idx_offset)).flatten()
        # find if there is any idxs to get from buffer
        to_replace = buffer_ns_locs.size != 0
        if to_replace:
            # extract the buffer_idxs first for replacement later
            # given head < ns_idx <= head + offset, and valid buffer idx is [0, offset)
            # get 0 < ns_idx - head <= offset, or equiv.
            # get -1 < ns_idx - head - 1 <= offset - 1, i.e.
            # get 0 <= ns_idx - head - 1 < offset, hence:
            buffer_idxs = ns_batch_idxs[buffer_ns_locs] - self.head - 1
            # set them to 0 first to allow sampling, then replace later with buffer
            ns_batch_idxs[buffer_ns_locs] = 0
        # guard all against overrun idxs from offset
        ns_batch_idxs = ns_batch_idxs % self.max_size
        next_states = util.cond_multiget(self.states, ns_batch_idxs)
        if to_replace:
            # now replace using buffer_idxs and ns_buffer
            buffer_ns = util.cond_multiget(self.ns_buffer, buffer_idxs)
            next_states[buffer_ns_locs] = buffer_ns
        return next_states

    def sample_idxs(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        batch_idxs = np.random.randint(self.size, size=batch_size)
        if self.use_cer:  # add the latest sample
            batch_idxs[-1] = self.head
        return batch_idxs


class SeqReplay(Replay):
    '''
    Preprocesses a state to be the stacked sequence of the last n states. Otherwise the same as Replay memory

    e.g. memory_spec
    "memory": {
        "name": "SeqReplay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    * seq_len provided by net_spec
    '''

    def __init__(self, memory_spec, body):
        super(SeqReplay, self).__init__(memory_spec, body)
        self.seq_len = self.body.agent.agent_spec['net']['seq_len']
        self.state_buffer = deque(maxlen=self.seq_len)
        self.reset()

    def preprocess_state(self, state, append=True):
        '''Transforms the raw state into format that is fed into the network'''
        # append when state is first seen when acting in policy_util, don't append elsewhere in memory
        self.preprocess_append(state, append)
        return np.stack(self.state_buffer)


class ConcatReplay(Replay):
    '''
    Preprocesses a state to be the concatenation of the last n states. Otherwise the same as Replay memory

    e.g. memory_spec
    "memory": {
        "name": "ConcatReplay",
        "batch_size": 32,
        "max_size": 10000,
        "concat_len": 4,
        "use_cer": true
    }
    '''

    def __init__(self, memory_spec, body):
        util.set_attr(self, memory_spec, [
            'batch_size',
            'max_size',
            'concat_len',  # number of stack states
            'use_cer',
        ])
        self.raw_state_dim = deepcopy(body.state_dim)  # used for state_buffer
        body.state_dim = body.state_dim * self.concat_len  # modify to use for net init for concat input
        super(ConcatReplay, self).__init__(memory_spec, body)
        self.state_buffer = deque(maxlen=self.concat_len)
        self.reset()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        super(ConcatReplay, self).reset()
        self.state_buffer.clear()
        for _ in range(self.state_buffer.maxlen):
            self.state_buffer.append(np.zeros(self.raw_state_dim))

    def epi_reset(self, state):
        '''Method to reset at new episode'''
        super(ConcatReplay, self).epi_reset(state)
        # reappend buffer with custom shape
        self.state_buffer.clear()
        for _ in range(self.state_buffer.maxlen):
            self.state_buffer.append(np.zeros(self.raw_state_dim))

    def preprocess_state(self, state, append=True):
        '''Transforms the raw state into format that is fed into the network'''
        # append when state is first seen when acting in policy_util, don't append elsewhere in memory
        self.preprocess_append(state, append)
        return np.concatenate(self.state_buffer)


class AtariReplay(Replay):
    '''
    Preprocesses an state to be the concatenation of the last four states, after converting the 210 x 160 x 3 image to 84 x 84 x 1 grayscale image, and clips all rewards to [-10, 10] as per "Playing Atari with Deep Reinforcement Learning", Mnih et al, 2013
    Note: Playing Atari with Deep RL clips the rewards to + / - 1

    e.g. memory_spec
    "memory": {
        "name": "AtariReplay",
        "batch_size": 32,
        "max_size": 250000,
        "stack_len": 4,
        "use_cer": true
    }
    '''

    def __init__(self, memory_spec, body):
        util.set_attr(self, memory_spec, [
            'batch_size',
            'max_size',
            'stack_len',  # number of stack states
            'use_cer',
        ])
        Replay.__init__(self, memory_spec, body)

    def add_experience(self, state, action, reward, next_state, done):
        # clip reward, done here to minimize change to only training data data
        super(AtariReplay, self).add_experience(state, action, np.sign(reward), next_state, done)


class ImageReplay(Replay):
    '''
    An off policy replay buffer that normalizes (preprocesses) images through
    division by 255 and subtraction of 0.5.
    '''

    def __init__(self, memory_spec, body):
        super(ImageReplay, self).__init__(memory_spec, body)

    def preprocess_state(self, state, append=True):
        state = util.normalize_image(state) - 0.5
        return state
