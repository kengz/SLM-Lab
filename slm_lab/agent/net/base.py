from abc import ABC, abstractmethod, abstractproperty


class Net(ABC):
    '''
    Abstract class ancestor to all Nets,
    specifies the necessary design blueprint for algorithm to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, net_spec, algorithm, in_dim, out_dim):
        '''
        @param {dict} net_spec is the spec for the net
        @param {*} algorithm is the module that uses network to act or train
        @param {int|list} in_dim is the input dimension(s) for the network. Usually use in_dim=body.state_dim
        @param {int|list} out_dim is the output dimension(s) for the network. Usually use out_dim=body.action_dim
        '''
        self.net_spec = net_spec
        self.algorithm = algorithm
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.agent_spec = algorithm.agent.agent_spec
