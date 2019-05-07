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
        self.memory_spec = memory_spec
        self.body = body

        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']

    @abstractmethod
    def reset(self):
        '''Method to fully reset the memory storage and related variables'''
        raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        '''Implement memory update given the full info from the latest timestep. NOTE: guard for np.nan reward and done when individual env resets.'''
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        '''Implement memory sampling mechanism'''
        raise NotImplementedError

    def print_memory_info(self):
        '''Prints size of all of the memory arrays'''
        for k in self.data_keys:
            d = getattr(self, k)
            logger.info(f'Memory for body {self.body.aeb}: {k} :shape: {d.shape}, dtype: {d.dtype}, size: {util.sizeof(d)}MB')
