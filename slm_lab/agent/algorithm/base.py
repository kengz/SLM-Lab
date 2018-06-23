from abc import ABC, abstractmethod, abstractproperty
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np

logger = logger.get_logger(__name__)


class Algorithm(ABC):
    '''
    Abstract class ancestor to all Algorithms,
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, agent):
        '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        '''
        self.agent = agent
        self.agent_spec = agent.agent_spec
        self.algorithm_spec = self.agent_spec['algorithm']
        self.memory_spec = self.agent_spec['memory']
        self.net_spec = self.agent_spec['net']
        self.last_loss = np.nan  # to record the loss from the last train() step

    @abstractmethod
    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.init_algorithm_params()
        self.init_nets()
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def init_nets(self):
        '''Initialize the neural network from the spec'''
        raise NotImplementedError

    @lab_api
    def post_init_nets(self):
        '''
        Method to conditionally load models.
        Call at the end of init_net() after setting self.net_names
        '''
        assert hasattr(self, 'net_names')
        if util.get_lab_mode() == 'enjoy':
            logger.info('Loaded algorithm models for lab_mode: enjoy')
            self.load()
        else:
            logger.info(f'Initialized algorithm models for lab_mode: {util.get_lab_mode()}')

    @lab_api
    def calc_pdparam(self, x, evaluate=True):
        '''
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        raise NotImplementedError

    @lab_api
    def body_act(self, body, state):
        '''Standard atomic body_act method. Atomic body logic should be implemented in submethods.'''
        raise NotImplementedError
        return action

    @lab_api
    def act(self, state_a):
        '''Interface-level agent act method for all its bodies. Resolves state to state; get action and compose into action.'''
        data_names = ['action']
        action_a, = self.agent.agent_space.aeb_space.init_data_s(data_names, a=self.agent.a)
        for (e, b), body in util.ndenumerate_nonan(self.agent.body_a):
            state = state_a[(e, b)]
            action_a[(e, b)] = self.body_act(body, state)
        return action_a

    def nanflat_to_data_a(self, data_name, nanflat_data_a):
        '''Reshape nanflat_data_a, e.g. action_a, from a single pass back into the API-conforming data_a'''
        data_names = [data_name]
        data_a, = self.agent.agent_space.aeb_space.init_data_s(data_names, a=self.agent.a)
        for body, data in zip(self.agent.nanflat_body_a, nanflat_data_a):
            e, b = body.e, body.b
            data_a[(e, b)] = data
        return data_a

    @abstractmethod
    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        raise NotImplementedError
        return batch

    @abstractmethod
    @lab_api
    def train(self):
        '''Implement algorithm train, or throw NotImplementedError'''
        if util.get_lab_mode() == 'enjoy':
            return np.nan
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def update(self):
        '''Implement algorithm update, or throw NotImplementedError'''
        raise NotImplementedError

    @lab_api
    def save(self, epi=None):
        '''Save net models for algorithm given the required property self.net_names'''
        if not hasattr(self, 'net_names'):
            logger.info('No net declared in self.net_names in init_nets(); no models to save.')
        else:
            net_util.save_algorithm(self, epi=epi)

    @lab_api
    def load(self):
        '''Load net models for algorithm given the required property self.net_names'''
        if not hasattr(self, 'net_names'):
            logger.info('No net declared in self.net_names in init_nets(); no models to load.')
        else:
            net_util.load_algorithm(self)
