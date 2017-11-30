from abc import ABC, abstractmethod, abstractproperty


class Algorithm(ABC):
    '''
    Abstract class ancestor to all Algorithms,
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def act_discrete(self, state):
        '''Implement discrete action, or throw NotImplementedError'''
        raise NotImplementedError
        return action

    @abstractmethod
    def act_continuous(self, state):
        '''Implement continuous action, or throw NotImplementedError'''
        raise NotImplementedError
        return action

    def act(self, state):
        '''Standard act method. Actions should be implemented in submethods'''
        if self.agent.bodies[0].env.is_discrete():
            return self.act_discrete(state)
        else:
            return self.act_continuous(state)

    @abstractmethod
    def update(self, reward, state, done):
        '''Implement algorithm update, or throw NotImplementedError'''
        raise NotImplementedError
