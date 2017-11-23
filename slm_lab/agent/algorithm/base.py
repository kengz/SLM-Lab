from abc import ABC, abstractmethod, abstractproperty


class Algorithm(ABC):
    '''
    Abstract class ancestor to all Algorithms,
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''
    # reference to agent for accessing the other agent components
    agent = None

    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def act_discrete(self, state):
        '''Implement discrete action, or throw NotImplementedError'''
        # TODO auto AEB space resolver from atomic method
        raise NotImplementedError
        return action

    @abstractmethod
    def act_continuous(self, state):
        '''Implement continuous action, or throw NotImplementedError'''
        raise NotImplementedError
        return action

    def act(self, state):
        '''Standard act method. Actions should be implemented in submethods'''
        if self.agent.env.is_discrete():
            return self.act_discrete(state)
        else:
            return self.act_continuous(state)

    @abstractmethod
    def update(self):
        '''Implement algorithm update, or throw NotImplementedError'''
        raise NotImplementedError
