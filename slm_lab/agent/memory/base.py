from abc import ABC, abstractmethod
from slm_lab.lib import logger

logger = logger.get_logger(__name__)


class Memory(ABC):
    '''Abstract Memory class to define the API methods'''

    def __init__(self, memory_spec, agent):
        '''
        @param {*} agent is the unit that stores its experience in this memory. Each agent has a distinct memory.
        '''
        self.memory_spec = memory_spec
        self.agent = agent
        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']

    @abstractmethod
    def reset(self):
        '''Method to fully reset the memory storage and related variables'''
        raise NotImplementedError

    @abstractmethod
    def update(self, state, action, reward, next_state, done, terminated, truncated):
        '''Implement memory update given the full info from the latest timestep. NOTE: guard for np.nan reward and done when individual env resets.'''
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        '''Implement memory sampling mechanism'''
        raise NotImplementedError
