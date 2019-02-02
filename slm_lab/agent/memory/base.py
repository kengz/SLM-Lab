from abc import ABC, abstractmethod
from collections import deque
from slm_lab.lib import logger, util
import numpy as np
import pydash as ps

logger = logger.get_logger(__name__)


class Memory(ABC):
    '''
    Abstract class ancestor to all Memories,
    specifies the necessary design blueprint for agent body to work in Lab.
    Mostly, implement just the abstract methods and properties.
    Memory is singleton to each body for modularity, and there is no gains to do multi-body memory now. Shall be constructed when body_space is built.
    '''

    def __init__(self, memory_spec, body):
        '''
        @param {*} body is the unit that stores its experience in this memory. Each body has a distinct memory.
        '''
        from slm_lab.experiment import analysis
        self.memory_spec = memory_spec
        self.body = body

        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']
        # the basic variables for every memory
        self.last_state = None
        # method to log size warning only once to prevent spamming log
        self.warn_size_once = ps.once(lambda msg: logger.warn(msg))
        # for API consistency, reset to some max_len in your specific memory class
        self.state_buffer = deque(maxlen=0)
        # total_reward and its history over episodes
        self.total_reward = 0
        self.total_reward_h = []
        self.avg_total_reward = 0
        self.avg_total_reward_h = []
        self.avg_window = analysis.MA_WINDOW

    @abstractmethod
    def reset(self):
        '''Method to fully reset the memory storage and related variables'''
        raise NotImplementedError

    def epi_reset(self, state):
        '''Method to reset at new episode'''
        self.last_state = state
        self.body.epi_reset()
        self.total_reward = 0
        self.state_buffer.clear()
        for _ in range(self.state_buffer.maxlen):
            self.state_buffer.append(np.zeros(self.body.state_dim))

    def base_update(self, action, reward, state, done):
        '''Method to do base memory update, like stats'''
        if np.isnan(reward):  # the start of episode
            self.epi_reset(state)
            return

        self.total_reward += reward
        if done:
            self.total_reward_h.append(self.total_reward)
            self.avg_total_reward = np.mean(self.total_reward_h[-self.avg_window:])
            self.avg_total_reward_h.append(self.avg_total_reward_h)
        return

    @abstractmethod
    def update(self, action, reward, state, done):
        '''Implement memory update given the full info from the latest timestep. Hint: use self.last_state to construct SARS. NOTE: guard for np.nan reward and done when individual env resets.'''
        self.base_update(action, reward, state, done)
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        '''Implement memory sampling mechanism'''
        raise NotImplementedError

    def preprocess_append(self, state, append=True):
        '''Method to conditionally append to state buffer'''
        if append:
            assert id(state) != id(self.state_buffer[-1]), 'Do not append to buffer other than during action'
            self.state_buffer.append(state)

    def preprocess_state(self, state, append=True):
        '''Transforms the raw state into format that is fed into the network'''
        return state

    def print_memory_info(self):
        '''Prints size of all of the memory arrays'''
        for k in self.data_keys:
            d = getattr(self, k)
            logger.info(f'Memory for body {self.body.aeb}: {k} :shape: {d.shape}, dtype: {d.dtype}, size: {util.sizeof(d)}MB')
