from collections import deque
from copy import deepcopy
from slm_lab.agent.memory.base import Memory
from slm_lab.lib import logger, util
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
        self.true_size = 0  # to number of experiences stored
        self.seen_size = 0  # the number of experiences seen, including those stored and discarded
        self.head = -1  # index of most recent experience
        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.scalar_shape = (self.max_size,)
        self.states_shape = self.scalar_shape + tuple(np.reshape(self.body.state_dim, -1))
        self.actions_shape = self.scalar_shape + self.body.action_space.shape
        self.reset()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        # set data keys as self.{data_keys}
        for k in self.data_keys:
            if k == 'states':
                setattr(self, k, np.zeros(self.states_shape, dtype=np.float16))
            elif k == 'next_states':
                # don't store next_states, but create a place holder to track it for sampling
                self.latest_next_state = None
            elif k == 'actions':
                setattr(self, k, np.zeros(self.actions_shape, dtype=self.body.action_space.dtype))
            else:
                setattr(self, k, np.zeros(self.scalar_shape))
        self.true_size = 0
        self.head = -1
        self.state_buffer.clear()
        for _ in range(self.state_buffer.maxlen):
            self.state_buffer.append(np.zeros(self.body.state_dim))

    def epi_reset(self, state):
        '''Method to reset at new episode'''
        super(Replay, self).epi_reset(self.preprocess_state(state, append=False))

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        self.base_update(action, reward, state, done)
        state = self.preprocess_state(state, append=False)  # prevent conflict with preprocess in epi_reset
        if not np.isnan(reward):  # not the start of episode
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state

    def add_experience(self, state, action, reward, next_state, done):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        self.states[self.head] = state
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.latest_next_state = next_state
        self.dones[self.head] = done
        # Actually occupied size of memory
        if self.true_size < self.max_size:
            self.true_size += 1
        self.seen_size += 1

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
                batch[k] = getattr(self, k)[self.batch_idxs]
        return batch

    def _sample_next_states(self, batch_idxs):
        '''Method to sample next_states from states, with proper guard for last idx (out of bound)'''
        # idxs for next state is state idxs + 1
        ns_batch_idxs = batch_idxs + 1
        # find the locations to be replaced with latest_next_state
        latest_ns_locs = np.argwhere(ns_batch_idxs == self.true_size).flatten()
        to_replace = latest_ns_locs.size != 0
        # set to 0, a safe sentinel for ns_batch_idxs due to the +1 above
        # then sample safely from self.states, and replace at locs with latest_next_state
        if to_replace:
            ns_batch_idxs[latest_ns_locs] = 0
        next_states = self.states[ns_batch_idxs]
        if to_replace:
            next_states[latest_ns_locs] = self.latest_next_state
        return next_states

    def sample_idxs(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        batch_idxs = np.random.randint(self.true_size, size=batch_size)
        # TODO fancy sampling only took early experiences somehow. could be due to binning stuff
        # batch_idxs = util.fast_uniform_sample(self.true_size, batch_size)
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
        # update states_shape and call reset again
        self.states_shape = self.scalar_shape + tuple(np.reshape([self.seq_len, self.body.state_dim], -1))
        self.reset()

    def preprocess_state(self, state, append=True):
        '''Transforms the raw state into format that is fed into the network'''
        # append when state is first seen when acting in policy_util, don't append elsewhere in memory
        self.preprocess_append(state, append)
        return np.stack(self.state_buffer)


class SILReplay(Replay):
    '''
    Special Replay for SIL, which adds the returns calculated from its OnPolicyReplay

    e.g. memory_spec
    "memory": {
        "name": "SILReplay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''

    def __init__(self, memory_spec, body):
        super(SILReplay, self).__init__(memory_spec, body)
        # adds a 'rets' scalar to the data_keys and call reset again
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'rets']
        self.reset()

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        raise AssertionError('Do not call SIL memory in main API control loop')

    def add_experience(self, state, action, reward, next_state, done, ret):
        '''Used to add memory from onpolicy memory'''
        super(SILReplay, self).add_experience(state, action, reward, next_state, done)
        self.rets[self.head] = ret


class SILSeqReplay(SILReplay, SeqReplay):
    '''
    Preprocesses a state to be the stacked sequence of the last n states. Otherwise the same as SILReplay memory

    e.g. memory_spec
    "memory": {
        "name": "SILSeqReplay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    * seq_len provided by net_spec
    '''
    pass


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
            self.state_buffer.append(state)

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
        # state_dim = (1, 84, 84) from env
        self.raw_state_dim = body.state_dim[1:]
        body.state_dim = (self.stack_len,) + self.raw_state_dim  # greyscale downsized, stacked
        Replay.__init__(self, memory_spec, body)


class ImageReplay(Replay):
    '''
    An off policy replay buffer that normalizes (preprocesses) images through
    division by 256 and subtraction of 0.5.
    '''

    def __init__(self, memory_spec, body):
        super(ImageReplay, self).__init__(memory_spec, body)

    def preprocess_state(self, state, append=True):
        state = (state.astype('float32') / 256.) - 0.5
        return state
