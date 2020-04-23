from abc import abstractmethod

import numpy as np
import pydash as ps

from slm_lab.agent import algorithm
from slm_lab.lib import logger
from slm_lab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


# TODO add some docs

class MetaAlgorithm(algorithm.Algorithm):
    ''' Abstract Meta Algorithm class to define the API methods '''

    def __init__(self, agent, global_nets, algorithm_spec,
                 memory_spec, net_spec, algo_idx=0, create_sub_algo=True):

        super().__init__(agent, global_nets=None, algorithm_spec=algorithm_spec,
                         memory_spec=None, net_spec=None,
                         algo_idx=algo_idx)
        self.meta_algorithm_spec = algorithm_spec
        self.algorithms = []
        self.algo_idx = algo_idx
        if create_sub_algo:
            # TODO manage global nets (needed in distributed training)
            # TODO why are we not using algorithm_spec instead of agent.agent_spec['algorithm']['contained_algorithms']?
            for algo_idx, virtual_agent_spec in enumerate(self.meta_algorithm_spec['contained_algorithms']):
                AlgorithmClass = getattr(algorithm, virtual_agent_spec['name'])
                algo = AlgorithmClass(agent,
                                      global_nets[algo_idx] if global_nets is not None else None,
                                      algorithm_spec=ps.get(virtual_agent_spec, 'algorithm'),
                                      memory_spec=ps.get(virtual_agent_spec, 'memory', None),
                                      net_spec=ps.get(virtual_agent_spec, 'net', None),
                                      algo_idx=algo_idx)
                self.algorithms.append(algo)

    @lab_api
    def save(self, ckpt=None):
        '''Save net models for each algorithms given the required property self.net_names'''
        for algorithm in self.algorithms:
            algorithm.save(ckpt=ckpt)

    @lab_api
    def load(self):
        '''Load net models for each algorithms given the required property self.net_names'''
        for algorithm in self.algorithms:
            algorithm.load()

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        pass

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network from the spec
        '''
        pass

    @abstractmethod
    def memory_update(self, state, action, welfare, next_state, done):
        raise NotImplementedError()

    @property
    def algorithm(self):
        raise NotImplementedError()

    @abstractmethod
    def act(self, state):
        '''Standard act method.'''
        raise NotImplementedError()

    @abstractmethod
    def sample(self):
        '''Samples a batch from memory'''
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        '''Implement algorithm train, or throw NotImplementedError'''
        raise NotImplementedError()

    @abstractmethod
    def update(self):
        '''Implement algorithm update, or throw NotImplementedError'''
        raise NotImplementedError()

    @property
    def net(self):
        raise NotImplementedError()

    @property
    def explore_var_scheduler(self):
        raise NotImplementedError()

    @property
    def entropy_coef_scheduler(self):
        raise NotImplementedError()

    def get_log_values(self):
        for idx, algo in enumerate(self.algorithms):
            if idx > 0:
                for k, v in algo.get_log_values().items():
                    k_meta = f'{k}_alg{idx}'
                    assert k_meta not in self.to_log.keys()
                    self.to_log[k_meta] = v
            else:
                self.to_log.update(algo.get_log_values())

        self._reset_temp_info()
        extra_training_info_to_log = self.to_log
        self.to_log = {}
        return extra_training_info_to_log

    def log_grad_norm(self):
        for algo in self.algorithms:
            algo.log_grad_norm()

    @property
    def name(self):
        raise NotImplementedError()

    @name.setter
    def name(self, value):
        raise NotImplementedError()

    @property
    def training_frequency(self):
        raise NotImplementedError()


class OneOfNAlgoActived(MetaAlgorithm):
    ''' OneOfNAlgoActived class to define the API methods. This meta-algo apply the curenlty activated algorithm. No
    heuristic are implemented in this class to change the activated algorithm'''

    def __init__(self, agent, global_nets, algorithm_spec,
                 memory_spec, net_spec, algo_idx=0):
        '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        :param algo_idx:
        '''

        super().__init__(agent, global_nets, algorithm_spec, memory_spec, net_spec, algo_idx)
        self.active_algo_idx = 0

    @property
    def name(self):
        return f'{self.meta_algo_name }({self.active_algo_idx} {self.algorithms[self.active_algo_idx].name})'

    @name.setter
    def name(self, value):
        'setting'
        self.meta_algo_name = value

    @property
    def training_frequency(self):
        return self.algorithms[self.active_algo_idx].training_frequency

    @lab_api
    def act(self, state):
        '''Standard act method.'''
        return self.algorithms[self.active_algo_idx].act(state)

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        return self.algorithms[self.active_algo_idx].sample()

    @property
    def net(self):
        return self.algorithms[self.active_algo_idx].net

    @lab_api
    def train(self):
        '''Implement algorithm train, or throw NotImplementedError'''
        losses = []

        for idx, algo in enumerate(self.algorithms):
            if self.agent.world.deterministic:
                self.agent.world._set_rd_state(self.agent.world.rd_seed)
            losses.append(algo.train())

        # TODO clip_eps_scheduler

        losses = [el for el in losses if not np.isnan(el)]
        loss = sum(losses) if len(losses) > 0 else np.nan

        if not np.isnan(loss):
            logger.debug(f"{self.active_algo_idx} loss {loss}")

        return loss

    @lab_api
    def update(self):
        '''Implement algorithm update, or throw NotImplementedError'''
        explore_vars = []
        for algo in self.algorithms:
            explore_vars.append(algo.update())
        explore_vars = [el for el in explore_vars if not np.isnan(el)]
        explore_var = sum(explore_vars) if len(explore_vars) > 0 else np.nan
        return explore_var

    @lab_api
    def memory_update(self, state, action, welfare, next_state, done):
        return self.algorithms[self.active_algo_idx].memory_update(state, action, welfare, next_state, done)

    @property
    def explore_var_scheduler(self):
        return self.algorithms[self.active_algo_idx].explore_var_scheduler

    @property
    def entropy_coef_scheduler(self):
        return self.algorithms[self.active_algo_idx].entropy_coef_scheduler
