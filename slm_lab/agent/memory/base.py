from abc import ABC, abstractmethod, abstractproperty


class Memory(ABC):
    '''
    Abstract class ancestor to all Memories,
    specifies the necessary design blueprint for agent body to work in Lab.
    Mostly, implement just the abstract methods and properties.
    Memory is singleton to each body for modularity, and there is no gains to do multi-body memory now. Shall be constructed when body_space is built.
    '''

    def __init__(self, body):
        self.body = body
        self.spec = body.agent.spec
        self.memory_spec = self.spec['memory']
        self.last_state = None

    def reset_last_state(self, state):
        '''Do reset of body memory per session during agent_space.reset() to set last_state'''
        self.last_state = state

    @abstractmethod
    def reset(self):
        '''Method specifically to reset the memory storage and related variables'''
        raise NotImplementedError

    @abstractmethod
    def update(self, action, reward, state, done):
        '''Implement memory update given the full info from the latest timestep. Hint: use self.last_state to construct SARS. NOTE: guard for np.nan reward and done when individual env resets.'''
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        '''Implement memory sampling mechanism'''
        raise NotImplementedError

    # TODO standardize sample method
