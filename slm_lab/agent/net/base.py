from abc import ABC, abstractmethod, abstractproperty


class Net(ABC):
    '''
    Abstract class ancestor to all Nets,
    specifies the necessary design blueprint for algorithm to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.spec = algorithm.agent.spec
        self.net_spec = self.spec['net']
