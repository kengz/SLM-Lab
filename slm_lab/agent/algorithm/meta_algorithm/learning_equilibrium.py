import numpy as np
import torch
import copy
from torch.distributions.kl import kl_divergence

from slm_lab.agent.agent import agent_util
from slm_lab.agent.algorithm import meta_algorithm
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from collections import deque

logger = logger.get_logger(__name__)


class LE(meta_algorithm.OneOfNAlgoActived):
    """
    Env must returns symmetrical agent states (giving agent 2 state to agent 1 should be fine)
    """

    # TODO add docs
    # TODO make it work for epi (how to divide epi when defection in during an epi ? (=> wait for end ?)

    def __init__(self, agent, global_nets=None, algorithm_spec=None,
                 memory_spec=None, net_spec=None, algo_idx=0):



        super().__init__(agent, global_nets, algorithm_spec, memory_spec, net_spec, algo_idx)

        util.set_attr(self, dict(
            punishement_time=10,
            min_coop_time=0,
            kl_divergence_puni_thr=1.0
        ))
        util.set_attr(self, self.algorithm_spec, [
            'punishement_time',
            'min_coop_time',
            'kl_divergence_puni_thr'
        ])

        assert len(self.algorithms) == 2
        self.coop_algo_idx = 0
        self.punish_algo_idx = 1
        self.remeaning_punishing_time = 0
        self.detected_defection = False
        self.kl_div_tot = 0
        self.kl_div_steps = 0
        # self.new_epi = True


        self.action_pd_coop = deque(maxlen=100)
        self.action_pd_punish = deque(maxlen=100)
        self.algo_temp_info = {
            "log_coop_time": 0,
            "log_punishement_time": 0,
            "log_kl_div_tot": 0,
            "log_kl_steps": 0
        }
        self.epsilon = 1e-6

    def act(self, state):
        action, action_pd = self.algorithms[self.active_algo_idx].act(state)

        if self.active_algo_idx == self.coop_algo_idx:
            self.action_pd_coop.append(action_pd.probs[0])
        elif self.active_algo_idx == self.punish_algo_idx:
            self.action_pd_punish.append(action_pd.probs[0])

        return action, action_pd

    def act_coop(self, state):
        return self.algorithms[self.coop_algo_idx].act(state)

    @lab_api
    def memory_update(self, state, action, welfare, next_state, done):

        if self.remeaning_punishing_time <= - (self.min_coop_time-1):

            other_ag_states = agent_util.get_from_other_agents(self.agent, key="state", default=[])
            other_ag_algorithms = agent_util.get_from_other_agents(self.agent, key="algorithm", default=[])

            with torch.no_grad():
                for other_ag_s, other_ag_algo in zip(other_ag_states, other_ag_algorithms):

                    _, current_ag_act_prob_distrib = self.act_coop(other_ag_s)
                    if isinstance(other_ag_algo, LE):
                        _, other_ag_act_prob_distrib = other_ag_algo.act_coop(other_ag_s)
                    else:
                        _, other_ag_act_prob_distrib = other_ag_algo.act(other_ag_s)

                    kl_div = kl_divergence(other_ag_act_prob_distrib, current_ag_act_prob_distrib)
                    self.algo_temp_info['log_kl_div_tot'] += kl_div
                    self.algo_temp_info['log_kl_steps'] += 1

                    # kl = ','.join([ "{:g}".format(el) for el in kl_div])
                    # other = ','.join([ "{:g}".format(el) for el in other_ag_act_prob_distrib.probs.tolist()[0]])
                    # current = ','.join([ "{:g}".format(el) for el in current_ag_act_prob_distrib.probs.tolist()[0]])
                    # print("state", other_ag_s)
                    # print('kl_div', kl, 'other', other, 'current', current)

                    # assert kl_div < 50.0
                    self.kl_div_tot += kl_div
                    self.kl_div_steps += 1

        assert (self.remeaning_punishing_time > 0) == (self.active_algo_idx == self.punish_algo_idx)
        assert (self.remeaning_punishing_time <= 0) == (self.active_algo_idx == self.coop_algo_idx)

        if self.remeaning_punishing_time > 0:
            other_agents_rewards = agent_util.get_from_other_agents(self.agent, key="reward", default=[])
            welfare = - sum(other_agents_rewards)
            self.algo_temp_info['log_punishement_time'] += 1
        else:
            self.algo_temp_info['log_coop_time'] += 1

        outputs = self.algorithms[self.active_algo_idx].memory_update(state, action, welfare, next_state, done)

        if done:

            if self.remeaning_punishing_time <= - (self.min_coop_time - 1):
                if self.kl_div_tot / (self.kl_div_steps + 1e-6) > self.kl_divergence_puni_thr:
                    self.detected_defection = True
                self.kl_div_tot = 0
                self.kl_div_steps = 0

            if self.remeaning_punishing_time > - self.min_coop_time:
                self.remeaning_punishing_time -= 1

            # Switch from coop to punishement only at the start of epi
            if self.detected_defection:
                self.active_algo_idx = self.punish_algo_idx
                self.remeaning_punishing_time = self.punishement_time
                self.detected_defection = False
                logger.debug("DEFECTION DETECTED")
                # print("DEFECTION DETECTED")

            if self.remeaning_punishing_time <= 0:
                self.active_algo_idx = self.coop_algo_idx
        return outputs

    def get_extra_training_log_info(self):
        extra_training_info_to_log = {
            "kl_div": round(float((self.algo_temp_info['log_kl_div_tot'] /
                            (self.algo_temp_info['log_kl_steps'] + self.epsilon))),2)
        }

        if self.algo_temp_info['log_coop_time'] == 0 and self.algo_temp_info['log_punishement_time'] == 0:
            extra_training_info_to_log["coop_frac"] = 0.5
        else:
            extra_training_info_to_log["coop_frac"] = round(float((self.algo_temp_info['log_coop_time'] /
                                                                  (self.algo_temp_info['log_punishement_time'] +
                                                                   self.algo_temp_info['log_coop_time']))),2)

        if self.agent.body.action_space_is_discrete:
            action_pd_coop = list(copy.deepcopy(self.action_pd_coop))
            action_pd_punish = list(copy.deepcopy(self.action_pd_punish))
            for act_idx in range(self.agent.body.action_dim):
                n_action_i = sum([el[act_idx] for el in action_pd_coop])
                extra_training_info_to_log[f'ca{act_idx}'] = round(float(n_action_i /
                                                                     (len(action_pd_coop) + self.epsilon)),2)
            for act_idx in range(self.agent.body.action_dim):
                n_action_i = sum([el[act_idx] for el in action_pd_punish])
                extra_training_info_to_log[f'pa{act_idx}'] = round(float(n_action_i /
                                                                     (len(action_pd_punish) + self.epsilon)),2)
        else:
            raise NotImplementedError()


        for algo in self.algorithms:
            extra_training_info_to_log.update(algo.get_extra_training_log_info())

        self._reset_temp_info()
        return extra_training_info_to_log
