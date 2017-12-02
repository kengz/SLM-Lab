from abc import ABC, abstractmethod, abstractproperty


class Memory(ABC):
    '''
    Abstract class ancestor to all Algorithms,
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''
    # TODO implement by extraction from replay

    def __init__(self, agent):
        self.agent = agent
