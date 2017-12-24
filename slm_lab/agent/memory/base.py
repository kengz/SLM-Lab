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
        self.last_state = None

    def reset_last_state(self, state):
        '''Episodic reset of memory, update last_state to the reset_state from env.'''
        # TODO this is per body, need to generalize
        self.last_state = state

    @abstractmethod
    def update(self, action, reward, state, done):
        '''Implement memory update given the full info from the latest timestep. Hint: use self.last_state to construct SARS.'''
        raise NotImplementedError

    # TODO standardize sample method
