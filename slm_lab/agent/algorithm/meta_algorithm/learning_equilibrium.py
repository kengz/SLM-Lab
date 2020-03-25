import numpy as np
import copy
from abc import ABC, abstractmethod

import torch

from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from slm_lab.agent import algorithm, memory, world
from slm_lab.agent.algorithm import meta_algorithm
from slm_lab.agent.agent import agent_util

logger = logger.get_logger(__name__)


class LE(meta_algorithm.OneOfNAlgoActived):
    # TODO add docs

    def __init__(self, agent, global_nets=None, algorithm_spec=None, memory_spec=None, net_spec=None, algo_idx=0):

        super().__init__(agent, global_nets, algorithm_spec, memory_spec, net_spec, algo_idx)

        util.set_attr(self, dict(
            punishement_time=10,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'punishement_time',
        ])

        assert len(self.algorithms) == 2
        self.coop_algo_idx = 0
        self.punisher_algo_idx = 1
        self.remeaning_punishing_time = 0
        self.min_coop_time = 3

    @lab_api
    def update(self):
        explore_vars = []
        for algo in self.algorithms:
            explore_vars.append(algo.update())
        explore_var = sum(explore_vars) if len(explore_vars) > 0 else np.nan

        if self.agent.world.shared_dict['done']:
            if self.remeaning_punishing_time > - self.min_coop_time :
                self.remeaning_punishing_time -= 1
            if self.remeaning_punishing_time == 0:
                self.active_algo_idx = self.coop_algo_idx
        return explore_var

    @lab_api
    def memory_update(self, state, action, welfare, next_state, done):

        if done:

            if self.remeaning_punishing_time <= - self.min_coop_time:
                current_agent_action = action
                other_agents_actions = agent_util.get_from_other_agents(self.agent, key="action", default=[])
                detected_defection = False
                # TODO BUG: Currently only the last action of the epi is used to detect defection (only OK if each epi is one step long)
                for other_agent_action in other_agents_actions:
                    if other_agent_action != current_agent_action:
                        welfare = 0
                        detected_defection = True
                        break
                if detected_defection:
                    self.active_algo_idx = self.punisher_algo_idx
                    self.remeaning_punishing_time = self.punishement_time
                    logger.debug("DEFECTION DETECTED")

        logger.debug(f'self.active_algo_idx {self.active_algo_idx}')

        if self.remeaning_punishing_time > 0:
            other_agents_rewards = agent_util.get_from_other_agents(self.agent, key="reward", default=[])
            welfare = - sum(other_agents_rewards)

        logger.debug(len(getattr(self.algorithms[self.active_algo_idx].memory, "cur_epi_data")['states']))
        return self.algorithms[self.active_algo_idx].memory_update(state, action, welfare, next_state, done)

