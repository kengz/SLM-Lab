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
from slm_lab.agent.algorithm.meta_algorithm.learning_equilibrium import copy_weights_between_networks

logger = logger.get_logger(__name__)


class SLConstrainedByRL(MetaAlgorithm):
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
                         algorithm_spec,
                         memory_spec, net_spec)

        util.set_attr(self, dict(
            penalty_coefficient=1.0
        ))
        util.set_attr(self, self.algorithm_spec, [
            'penalty_coefficient'
        ])

        # Save for later redeployments
        self.agent = agent
        self.algo_to_choose_from = copy.deepcopy(self.meta_algorithm_spec['contained_algorithms'])
        self.global_nets = global_nets

        self.spl_algo_idx = 0
        self.rl_algo_idx = 1



    @property
    def name(self):
        return f'{self.meta_algo_name }({self.active_algo_idx} {self.algorithms[self.active_algo_idx].name} {self.agent.welfare_function.__name__})'

    @name.setter
    def name(self, value):
        'setting'
        self.meta_algo_name = value

    # @property
    # def coop_algo_idx(self):
    #     return self.algorithms[self.active_algo_idx].coop_algo_idx

    @property
    def training_frequency(self):
        return self.algorithms[self.active_algo_idx].training_frequency

    @lab_api
    def act(self, state):
        '''Standard act method.'''
        if self.agent.world.deterministic:
            self.agent.world._set_rd_state(self.agent.world.rd_seed)
        return self.algorithms[self.spl_algo_idx].act(state)

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
            if algo.to_train == 1 and idx == self.spl_algo_idx:
                # temp_weigths = { k:v.clone().detach() for k,v in self.algorithms[
                #     self.rl_algo_idx].net.named_parameters()}
                # temp_net = algo.net.clone()
                # algo.net.attribute = list(algo.net.attribute)
                # temp_net = copy.deepcopy(algo.net)
                import pickle
                temp_net = pickle.loads(pickle.dumps(algo.net))

                if hasattr(self, "previous_spl_net"):
                    current_spl_net_weights = dict(self.algorithms[self.spl_algo_idx].net.named_parameters())
                    # print("self.algorithms[self.spl_algo_idx].net.parameters[0]", list(
                    #     self.algorithms[self.spl_algo_idx].net.parameters())[0][0,4])

                    loss_penalty = sum((current_spl_net_weights[k] - v.detach()).norm()
                                  for k, v in self.last_rl_net_weights.items())
                    # print("self.penalty_coefficient", self.penalty_coefficient)
                    loss_penalty *= self.penalty_coefficient
                else:
                    loss_penalty = 0
                print("in loss_penalty", loss_penalty)
                if self.agent.world.deterministic:
                    self.agent.world._set_rd_state(self.agent.world.rd_seed)
                # print("in spl_algo_idx")
                losses.append(algo.train(loss_penalty))
                # print("after self.algorithms[self.rl_algo_idx].net.parameters[0]",
                #       list(self.algorithms[self.rl_algo_idx].net.parameters())[0][0,4])

                # self.previous_spl_net_weights = temp_weigths
                self.previous_spl_net = temp_net
            elif algo.to_train == 1 and idx == self.rl_algo_idx:
                # print("self.previous_spl_net.parameters[0]", list(self.previous_spl_net.parameters())[0][0,4])
                # print("before copy algo.net.parameters[0]", list(algo.net.parameters())[0][0,4])
                copy_weights_between_networks(copy_from_net=self.previous_spl_net,
                                              copy_to_net=algo.net)
                # print("algo.net.parameters[0]", list(algo.net.parameters())[0][0,4])
                losses.append(algo.train())
                # print("after train algo.net.parameters[0]", list(algo.net.parameters())[0][0,4])

                self.last_rl_net_weights = dict(algo.net.named_parameters())

        losses = [ el for el in losses if not np.isnan(el)]
        loss = sum(losses) if len(losses) > 0 else np.nan

        if not np.isnan(loss):
            logger.debug(f"loss {loss}")

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
    def memory_update(self, state, action, welfare, next_state, done, idx):
        # for algo in self.algorithms:
        #     algo.memory_update(state, action, welfare, next_state, done)
        self.algorithms[idx].memory_update(state, action, welfare, next_state, done)

    @property
    def explore_var_scheduler(self):
        return self.algorithms[self.active_algo_idx].explore_var_scheduler

    @property
    def entropy_coef_scheduler(self):
        return self.algorithms[self.active_algo_idx].entropy_coef_scheduler

