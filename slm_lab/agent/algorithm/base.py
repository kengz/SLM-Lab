from abc import ABC, abstractmethod
from slm_lab.agent.net import net_util
from slm_lab.lib import logger
from slm_lab.lib.env_var import lab_mode
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch

logger = logger.get_logger(__name__)


class Algorithm(ABC):
    '''Abstract Algorithm class to define the API methods'''

    def __init__(self, agent, global_nets=None):
        '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        '''
        self.agent = agent
        self.algorithm_spec = agent.agent_spec['algorithm']
        self.name = self.algorithm_spec['name']
        self.memory_spec = agent.agent_spec['memory']
        self.net_spec = agent.agent_spec['net']
        self.init_algorithm_params()
        self.init_nets(global_nets)

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters and schedulers'''
        # Initialize common scheduler attributes
        if hasattr(self, 'explore_var_spec') and self.explore_var_spec is not None:
            from slm_lab.agent.algorithm import policy_util
            self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
            self.agent.explore_var = self.explore_var_scheduler.start_val
            # Register for logging
            self.agent.mt.register_algo_var('explore_var', self.agent)

    @abstractmethod
    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network from the spec'''
        raise NotImplementedError

    @lab_api
    def end_init_nets(self):
        '''Checkers and conditional loaders called at the end of init_nets()'''
        # check all nets naming
        assert hasattr(self, 'net_names')
        for net_name in self.net_names:
            assert net_name.endswith('net'), f'Naming convention: net_name must end with "net"; got {net_name}'

        # load algorithm if is in train@ resume or enjoy mode
        if self.agent.spec['meta']['resume'] or lab_mode() == 'enjoy':
            self.load()

    @lab_api
    def calc_pdparam(self, x, net=None):
        '''
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        raise NotImplementedError

    def to_action(self, action: torch.Tensor) -> np.ndarray:
        '''Convert tensor action to numpy with gymnasium-compatible shapes
        
        Handles 8 action type combinations:
        1. Single CartPole (2 actions): (1,) → scalar int
        2. Vector CartPole (2 actions): (2,) → (2,) 
        3. Single LunarLander (4 actions): (1,) → scalar int
        4. Vector LunarLander (4 actions): (2,) → (2,)
        5. Single Pendulum (1D): (1, 1) → (1,)
        6. Vector Pendulum (1D): (2, 1) → (2, 1)
        7. Single BipedalWalker (4D): (1, 4) → (4,)
        8. Vector BipedalWalker (4D): (2, 4) → (2, 4)
        '''
        action_np = action.cpu().numpy()
        
        # Single environments need scalars for discrete, squeezed arrays for continuous
        if not self.agent.env.is_venv:
            if self.agent.env.is_discrete and action_np.size == 1:
                action_np = action_np.item()  # (1,) or scalar → int
            elif not self.agent.env.is_discrete and action_np.ndim == 2:
                action_np = action_np.squeeze(0)  # (1, action_dim) → (action_dim,)
        
        # Vector continuous environments need (num_envs, action_dim) shape
        elif self.agent.env.is_venv and not self.agent.env.is_discrete:
            if action_np.ndim == 1:  # Got (num_envs*action_dim,), need (num_envs, action_dim)
                action_np = action_np.reshape(self.agent.env.num_envs, self.agent.env.action_dim)
        
        return action_np

    @lab_api
    def act(self, state):
        '''Standard act method.'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def train(self):
        '''Implement algorithm train, or throw NotImplementedError'''
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
        # set decayable variables to initial values
        for k, v in vars(self).items():
            if k.endswith('_scheduler') and hasattr(v, 'start_val'):
                var_name = k.replace('_scheduler', '')
                setattr(self.agent, var_name, v.start_val)
