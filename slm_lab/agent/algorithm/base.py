from abc import ABC, abstractmethod
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

    def __init__(self, agent, global_nets=None):
        '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        '''
        self.agent = agent
        self.algorithm_spec = agent.agent_spec['algorithm']
        self.name = self.algorithm_spec['name']
        self.memory_spec = agent.agent_spec['memory']
        self.net_spec = agent.agent_spec['net']
        self.body = self.agent.body
        self.init_algorithm_params()
        self.init_nets(global_nets)
        logger.info(util.self_desc(self))

    @abstractmethod
    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network from the spec'''
        raise NotImplementedError

    @lab_api
    def post_init_nets(self):
        '''
        Method to conditionally load models.
        Call at the end of init_nets() after setting self.net_names
        '''
        assert hasattr(self, 'net_names')
        for net_name in self.net_names:
            assert net_name.endswith('net'), f'Naming convention: net_name must end with "net"; got {net_name}'
        if util.in_eval_lab_modes():
            self.load()
            logger.info(f'Loaded algorithm models for lab_mode: {util.get_lab_mode()}')
        else:
            logger.info(f'Initialized algorithm models for lab_mode: {util.get_lab_mode()}')

    @lab_api
    def calc_pdparam(self, x, net=None):
        '''
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        raise NotImplementedError

    @lab_api
    def act(self, state):
        '''Standard act method.'''
        raise NotImplementedError
        return action

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
        if util.in_eval_lab_modes():
            return np.nan
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def update(self):
        '''Implement algorithm update, or throw NotImplementedError'''
        raise NotImplementedError

    @lab_api
    def save(self, ckpt=None):
        '''Save net models for algorithm given the required property self.net_names'''
        if not hasattr(self, 'net_names'):
            logger.info('No net declared in self.net_names in init_nets(); no models to save.')
        else:
            net_util.save_algorithm(self, ckpt=ckpt)

    @lab_api
    def load(self):
        '''Load net models for algorithm given the required property self.net_names'''
        if not hasattr(self, 'net_names'):
            logger.info('No net declared in self.net_names in init_nets(); no models to load.')
        else:
            net_util.load_algorithm(self)
        # set decayable variables to final values
        for k, v in vars(self).items():
            if k.endswith('_scheduler') and hasattr(v, 'end_val'):
                var_name = k.replace('_scheduler', '')
                setattr(self.body, var_name, v.end_val)
