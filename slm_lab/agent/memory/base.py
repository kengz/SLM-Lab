from abc import ABC, abstractmethod, abstractproperty
from collections import deque


class Memory(ABC):
    '''
    Abstract class ancestor to all Memories,
    specifies the necessary design blueprint for agent body to work in Lab.
    Mostly, implement just the abstract methods and properties.
    Memory is singleton to each body for modularity, and there is no gains to do multi-body memory now. Shall be constructed when body_space is built.
    '''

    def __init__(self, memory_spec, algorithm, body):
        '''
        @param {*} body is the unit that stores its experience in this memory. Each body has a distinct memory.
        '''
        self.memory_spec = memory_spec
        self.algorithm = algorithm
        self.body = body
        self.agent_spec = body.agent.agent_spec
        self.last_state = None
        self.state_buffer = deque(maxlen=0)  # for API consistency

    @abstractmethod
    def reset(self):
        '''Method to fully reset the memory storage and related variables'''
        raise NotImplementedError

    def epi_reset(self, state):
        '''Method to reset at new episode'''
        self.last_state = state
        self.state_buffer.clear()

    @abstractmethod
    def update(self, action, reward, state, done):
        '''Implement memory update given the full info from the latest timestep. Hint: use self.last_state to construct SARS. NOTE: guard for np.nan reward and done when individual env resets.'''
        if np.isnan(reward):  # the start of episode
            self.epi_reset(state)
        else:
            pass
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        '''Implement memory sampling mechanism'''
        raise NotImplementedError
