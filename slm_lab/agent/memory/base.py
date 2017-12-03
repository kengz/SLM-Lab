from abc import ABC, abstractmethod, abstractproperty


class Memory(ABC):
    '''
    Abstract class ancestor to all Algorithms,
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, agent):
        self.agent = agent
        self.last_state = None

    @abstractmethod
    def post_body_init(self):
        '''Initializes the part of memory needing a body to exist first.'''
        raise NotImplementedError

    def reset_last_state(self, state):
        '''Episodic reset of memory, update last_state to the reset_state from env.'''
        # TODO this is per body, need to generalize
        self.last_state = state

    @abstractmethod
    def update(self, action, reward, state, done):
        '''Implement memory update given the full info from the latest timestep. Hint: use self.last_state to construct SARS.'''
        raise NotImplementedError

    # TODO standardize sample method
