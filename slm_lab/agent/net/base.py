from abc import ABC, abstractmethod, abstractproperty


class Net(ABC):
    '''
    Abstract class ancestor to all Nets,
    specifies the necessary design blueprint for algorithm to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, net_spec, algorithm, body):
        '''
        @param {*} algorithm is the module that uses network to act or train
        @param {*} body has properties like observation_space, action_space and dim for constructing the input/output layers of network. This param could also be an array for hydra architecture.
        '''
        self.net_spec = net_spec
        self.algorithm = algorithm
        self.body = body
        self.agent_spec = algorithm.agent.agent_spec
