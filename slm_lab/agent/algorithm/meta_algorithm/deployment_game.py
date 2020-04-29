from abc import ABC, abstractmethod

import numpy as np
import pydash as ps
import copy

from slm_lab.agent.algorithm.meta_algorithm import MetaAlgorithm
from abc import ABC, abstractmethod
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from slm_lab.agent import algorithm, memory, world
from slm_lab.agent.agent import agent_util, observability

logger = logger.get_logger(__name__)


class DeploymentGame(MetaAlgorithm):
    ''' OneOfNAlgoActived class to define the API methods. This meta-algo apply the curenlty activated algorithm. No
    heuristic are implemented in this class to change the activated algorithm'''

    def __init__(self, agent, global_nets, algorithm_spec,
                 memory_spec, net_spec, algo_idx=0):
        '''
        @param {*} agent is the container for algorithm and related components, and interfaces with env.
        :param algo_idx:
        '''

        self.algorithms = [] # To prevent crash
        super().__init__(agent, global_nets,
                         algorithm_spec=algorithm_spec,
                         memory_spec=None,
                         net_spec=None,
                         create_sub_algo=False)

        # Save for later redeployments
        self.agent = agent
        self.algo_to_choose_from = copy.deepcopy(self.meta_algorithm_spec['contained_algorithms'])
        self.global_nets = global_nets

        # Create the algo which choose which sub algo to deploy
        self.algorithms = [None, None]
        self.deploy_algo_idx = 1
        DeployerAlgoClass = getattr(algorithm, ps.get(self.meta_algorithm_spec, 'deploy_algo.name'))
        self.algorithms[self.deploy_algo_idx] = DeployerAlgoClass(self.agent, algorithm_spec=ps.get(
                                                                    self.meta_algorithm_spec,'deploy_algo'))


        # Deploy for the 1st time
        self.active_algo_idx = 0 # Not rely used / always equals 0 since we are picking one algo from the list of algo
        self.redeploy_every_n_epi = self.meta_algorithm_spec['redeploy_every_n_epi']
        self.epi_since_last_deploy = 0
        self.epi_tot_reward = 0

        self.deploy_sub_algo(init = True)


    @property
    def name(self):
        return f'{self.meta_algo_name }({self.active_algo_idx} {self.algorithms[self.active_algo_idx].name} {self.agent.welfare_function.__name__})'

    @name.setter
    def name(self, value):
        'setting'
        self.meta_algo_name = value

    @property
    def coop_algo_idx(self):
        return self.algorithms[self.active_algo_idx].coop_algo_idx

    @property
    def training_frequency(self):
        return self.algorithms[self.active_algo_idx].training_frequency

    def deploy_sub_algo(self, init=False):
        if not init:
            self.to_log.update({"deployed_algo_idx": self.algo_to_redeploy_idx})
            self.to_log.update({"undisc_tot_r": self.epi_tot_reward/self.epi_since_last_deploy})
            self.algorithms[self.deploy_algo_idx].memory_update(state=None, action=None,
                                           welfare=self.epi_tot_reward/self.epi_since_last_deploy,
                                           next_state=None, done=None)

        self.algo_to_redeploy_idx, _ = self.algorithms[self.deploy_algo_idx].act(state=None)

        algo_to_deploy_spec = self.algo_to_choose_from[self.algo_to_redeploy_idx]
        AlgorithmClass = getattr(algorithm, algo_to_deploy_spec['name'])
        algo = AlgorithmClass(self.agent,
                              self.global_nets[self.algo_idx][self.algo_to_redeploy_idx] if self.global_nets is not None else None,
                              algorithm_spec=ps.get(algo_to_deploy_spec, 'algorithm', None),
                              memory_spec=ps.get(algo_to_deploy_spec, 'memory', None),
                              net_spec=ps.get(algo_to_deploy_spec, 'net', None),
                              algo_idx=self.algo_to_redeploy_idx)
        self.algorithms = [algo, self.algorithms[self.deploy_algo_idx]]

        # Change welfare to fit the one in the algo spec
        update_welfare_fn = ps.get(algo_to_deploy_spec, 'welfare_function', None)
        if update_welfare_fn is not None:
            self.agent.welfare_function = getattr(agent_util, update_welfare_fn)

    @lab_api
    def act(self, state):
        '''Standard act method.'''

        if self.epi_since_last_deploy == self.redeploy_every_n_epi:
            self.deploy_sub_algo()
            self.epi_since_last_deploy = 0
            self.epi_tot_reward = 0

        if self.agent.world.deterministic:
            self.agent.world._set_rd_state(self.agent.world.rd_seed)
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
        losses = [ el for el in losses if not np.isnan(el)]
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
        self.epi_tot_reward += self.agent.reward
        if done:
            self.epi_since_last_deploy += 1
        return self.algorithms[self.active_algo_idx].memory_update(state, action, welfare, next_state, done)

    @property
    def explore_var_scheduler(self):
        return self.algorithms[self.active_algo_idx].explore_var_scheduler

    @property
    def entropy_coef_scheduler(self):
        return self.algorithms[self.active_algo_idx].entropy_coef_scheduler

