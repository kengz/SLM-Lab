from abc import ABC, abstractmethod

import numpy as np
import pydash as ps

from slm_lab.agent import memory
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from slm_lab.agent.agent import agent_util
from slm_lab.env.base import Clock
from slm_lab.agent.algorithm import policy_util
logger = logger.get_logger(__name__)


class Algorithm(ABC):
    '''Abstract Algorithm class to define the API methods'''

    def __init__(self, agent, global_nets, algorithm_spec, memory_spec, net_spec, algo_idx=0):
        '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        :param algo_idx:
        '''
        self.algo_idx = algo_idx
        self.agent = agent
        self._set_internal_clock()
        self.algorithm_spec = algorithm_spec
        self.net_spec = net_spec
        self.memory_spec = memory_spec

        self.name = self.algorithm_spec['name']
        self.body = self.agent.body

        # self.action_type = policy_util.get_space_type(self.agent.body.action_space)
        self.action_pdtype = ps.get(algorithm_spec, f'action_pdtype')
        if self.action_pdtype in (None, 'default'):
            self.action_pdtype = policy_util.ACTION_PDS[self.agent.body.action_type][0]
        self.ActionPD = policy_util.get_action_pd_cls(self.action_pdtype, self.agent.body.action_type)

        self.init_memory()
        self.init_algorithm_params()
        self.init_nets(global_nets)

        # Extra customizable training log
        # self.algo_temp_info = {} if not hasattr(self, "algo_temp_info") else self.algo_temp_info
        self.to_log = {}

        # Welfare function can also be changed by algorithm
        # TODO change this ( set by agent, algo and meta algo...)
        update_welfare_fn = ps.get(self.algorithm_spec, 'welfare_function', None)
        if update_welfare_fn is not None:
            agent.welfare_function = getattr(agent_util, update_welfare_fn)

        util.set_attr(self, dict(
            display_gradflow=False,
            boltzmann_cluster_threshold=0.25,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'display_gradflow',
            'boltzmann_cluster_threshold',
        ])
        self.episilon = 1e-12

    def init_memory(self, add_to_self=True):
        if self.memory_spec is not None:
            MemoryClass = getattr(memory, ps.get(self.memory_spec, 'name'))
            memory_obj = MemoryClass(self.memory_spec, self)
        else:
            memory_obj = None

        if add_to_self:
            self.memory = memory_obj
        return memory_obj

    def _set_internal_clock(self):
        # tick with a multiple of num_envs to properly count frames
        self.clock_speed = 1 * (self.agent.body.env.num_envs or 1)
        self.internal_clock = Clock(self.agent.body.env.max_frame, self.clock_speed)

    @abstractmethod
    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        raise NotImplementedError

    @abstractmethod
    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network from the spec
        '''
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
    def proba_distrib_params(self, x, net=None):
        '''
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        raise NotImplementedError

    @lab_api
    def act(self, state):
        '''Standard act method.'''
        raise NotImplementedError
        return action, action_pd

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
        '''Save net models for algorithm given the required property self.net_names
        '''
        if not hasattr(self, 'net_names'):
            logger.info('No net declared in self.net_names in init_nets(); no models to save.')
        else:
            net_util.save_algorithm(self, ckpt=ckpt, prefix=self.identifier + '_')

    @property
    def identifier(self):
        identifier = f"agent_n{self.agent.agent_idx}_algo_n{self.algo_idx}"
        return identifier

    @lab_api
    def load(self):
        '''Load net models for algorithm given the required property self.net_names
        '''
        if not hasattr(self, 'net_names'):
            logger.info('No net declared in self.net_names in init_nets(); no models to load.')
        else:
            net_util.load_algorithm(self, prefix=self.identifier + '_')
        # set decayable variables to final values
        for k, v in vars(self).items():
            if k.endswith('_scheduler') and hasattr(v, 'end_val'):
                var_name = k.replace('_scheduler', '')
                setattr(self.body, var_name, v.end_val)

    def memory_update(self, state, action, welfare, next_state, done):
        # TODO Support vectorized env += n vectorized, done ?
        self.internal_clock.tick(unit='t')
        return self.memory.update(state, action, welfare, next_state, done)

    def get_log_values(self):
        if hasattr(self, "net"):
            self.to_log['opt_step'] = self.net.opt_step

        if hasattr(self, "entropy_coef_scheduler"):
            self.to_log["entropy_coef"] = self.entropy_coef_scheduler.val

        to_log = self.to_log
        self.to_log = {}
        # self._reset_temp_info()
        return to_log

    # def _reset_temp_info(self):
    #     for k, v in self.algo_temp_info.items():
    #         if isinstance(v, str):
    #             self.algo_temp_info[k] = ""
    #         else:
    #             self.algo_temp_info[k] = np.nan

    def log_grad_norm(self):
        grad_norms = net_util.get_grad_norms(self)
        self.to_log['grad_norm'] = np.nan if ps.is_empty(grad_norms) else np.mean(grad_norms)
