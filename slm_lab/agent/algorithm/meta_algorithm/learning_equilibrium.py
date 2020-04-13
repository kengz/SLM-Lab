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


def compare_two_identical_net(net1, net2, fn=lambda x: x):
    aggregated_value = 0
    for (n1, p1), (n2, p2) in zip(net1.named_parameters(), net2.named_parameters()):
        aggregated_value += fn(p1.data - p2.data).abs().sum()
    return aggregated_value


def copy_weights_between_networks(copy_from_net, copy_to_net):
    copy_to_net.load_state_dict(copy_from_net.state_dict())
    compare_two_identical_net(copy_to_net, copy_to_net)


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
            defection_detection_mode="action_pd_kullback_leibler_div",
            punishement_time=10,
            min_coop_time=0,
            defection_carac_threshold=1.0
        ))
        util.set_attr(self, self.algorithm_spec, [
            'defection_detection_mode',
            'punishement_time',
            'min_coop_time',
            'defection_carac_threshold'
        ])

        self.all_defection_detection_modes = ["network_weights",
                                             "action_pd_kullback_leibler_div"]

        assert self.defection_detection_mode in self.all_defection_detection_modes
        if self.defection_detection_mode == self.all_defection_detection_modes[0]:
            assert len(self.algorithms) == 3
            self.is_fully_init = False
        if self.defection_detection_mode == self.all_defection_detection_modes[1]:
            assert len(self.algorithms) == 2, str(len(self.algorithms))
            # assert len(self.agent.world.agents) == 2, str(self.agent.world.agents) + str(len(self.agent.world.agents))
            self.is_fully_init = True

        self.coop_algo_idx = 0
        self.punish_algo_idx = 1

        self.remeaning_punishing_time = 0
        self.detected_defection = False

        self.defection_carac_tot = 0
        self.defection_carac_steps = 0


        self.action_pd_coop = deque(maxlen=100)
        self.action_pd_punish = deque(maxlen=100)
        self.algo_temp_info = {
            "log_coop_time": 0,
            "log_punishement_time": 0,
            "log_defection_carac_tot": 0,
            "log_defection_carac_steps": 0
        }
        self.epsilon = 1e-6


    def act(self, state):
        action, action_pd = self.algorithms[self.active_algo_idx].act(state)

        # To log action prob distrib
        if self.active_algo_idx == self.coop_algo_idx:
            self.action_pd_coop.append(action_pd.probs[0])
        elif self.active_algo_idx == self.punish_algo_idx:
            self.action_pd_punish.append(action_pd.probs[0])

        return action, action_pd

    def act_coop(self, state):
        return self.algorithms[self.coop_algo_idx].act(state)

    def detect_defection(self, state, action, welfare, next_state, done):
        if not isinstance(self.defection_carac_tot, torch.Tensor) and np.isnan(self.defection_carac_tot):
            self.defection_carac_tot = 0
            self.defection_carac_steps = 0
            self.algo_temp_info['log_defection_carac_tot'] = 0
            self.algo_temp_info['log_defection_carac_steps'] = 0

        if self.defection_detection_mode == self.all_defection_detection_modes[0]:
            return self.defection_from_network_weights(state, action, welfare, next_state, done)
        elif self.defection_detection_mode == self.all_defection_detection_modes[1]:
            return self.defection_from_action_pd_kl_div(state, action, welfare, next_state, done)

    def defection_from_network_weights(self, state, action, welfare, next_state, done):
        # Update the coop networks simulating the opponents
        other_ag_states = agent_util.get_from_other_agents(self.agent, key="state", default=[])
        other_ag_action = agent_util.get_from_other_agents(self.agent, key="action", default=[])
        other_ag_rewards = agent_util.get_from_other_agents(self.agent, key="reward", default=[])
        other_ag_next_states = agent_util.get_from_other_agents(self.agent, key="next_state", default=[])
        other_ag_algorithms = agent_util.get_from_other_agents(self.agent, key="algorithm", default=[])




        for idx, (s, a, r, n_s, algo) in enumerate(zip(other_ag_states, other_ag_action,
                                                       other_ag_rewards, other_ag_next_states,
                                                       other_ag_algorithms)):

            coop_net_simul_opponent = self.punish_algo_idx + idx +1
            # TODO is not currently shared between agents because it is only computed in update (agent sequential)
            # Recompute welfare using the currently agent welfare function
            w = self.agent.welfare_function(algo.agent, r)
            if not self.is_fully_init:
                logger.info("LE algo finishing init by copying weight from opponent network")
                if isinstance(algo, meta_algorithm.MetaAlgorithm):
                    net_from = algo.algorithms[algo.coop_algo_idx].net
                else:
                    net_from = algo.net
                copy_weights_between_networks(copy_from_net = net_from,
                                                  copy_to_net= self.algorithms[coop_net_simul_opponent].net)

            self.algorithms[coop_net_simul_opponent].memory_update(
                s, a, w, n_s, done)

            diff = compare_two_identical_net(self.algorithms[coop_net_simul_opponent].net,
                                                  algo.net)
            self.defection_carac_tot += diff
            self.defection_carac_steps += 1
            self.algo_temp_info['log_defection_carac_tot'] += diff
            self.algo_temp_info['log_defection_carac_steps'] += 1

        if not self.is_fully_init:
            self.is_fully_init = True

    def defection_from_action_pd_kl_div(self, state, action, welfare, next_state, done):
        """Defection is determined by the average of the kullback leibler divergence over an episode"""

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


                # kl = ','.join([ "{:g}".format(el) for el in kl_div])
                # other = ','.join([ "{:g}".format(el) for el in other_ag_act_prob_distrib.probs.tolist()[0]])
                # current = ','.join([ "{:g}".format(el) for el in current_ag_act_prob_distrib.probs.tolist()[0]])
                # print("state", other_ag_s)
                # print('kl_div', kl, 'other', other, 'current', current)

                # assert kl_div < 50.0
                self.defection_carac_tot += kl_div
                self.defection_carac_steps += 1
                self.algo_temp_info['log_defection_carac_tot'] += kl_div
                self.algo_temp_info['log_defection_carac_steps'] += 1

    @lab_api
    def memory_update(self, state, action, welfare, next_state, done):

        # start at the last step before the end of min_coop_time
        if self.remeaning_punishing_time <= - (self.min_coop_time-1):
            self.detect_defection(state, action, welfare, next_state, done)

        assert (self.remeaning_punishing_time > 0) == (self.active_algo_idx == self.punish_algo_idx)
        assert (self.remeaning_punishing_time <= 0) == (self.active_algo_idx == self.coop_algo_idx)

        if np.isnan(self.algo_temp_info['log_punishement_time']):
            self.algo_temp_info['log_punishement_time'] = 0
            self.algo_temp_info['log_coop_time'] = 0

        if self.remeaning_punishing_time > 0:
            other_agents_rewards = agent_util.get_from_other_agents(self.agent, key="reward", default=[])
            welfare = - sum(other_agents_rewards)
            self.algo_temp_info['log_punishement_time'] += 1
        else:
            self.algo_temp_info['log_coop_time'] += 1

        outputs = self.algorithms[self.active_algo_idx].memory_update(state, action, welfare, next_state, done)

        if done:
            # print("self.defection_carac_tot", self.defection_carac_tot)
            if self.remeaning_punishing_time <= - (self.min_coop_time - 1):
                if self.defection_carac_tot / (self.defection_carac_steps + self.epsilon) > self.defection_carac_threshold:
                    self.detected_defection = True

            self.defection_carac_tot = np.nan
            self.defection_carac_steps = np.nan

            if self.remeaning_punishing_time > - self.min_coop_time:
                self.remeaning_punishing_time -= 1

            # Switch from coop to punishement only at the start of epi
            if self.detected_defection:
                self.active_algo_idx = self.punish_algo_idx
                self.remeaning_punishing_time = self.punishement_time
                self.detected_defection = False
                logger.debug("DEFECTION DETECTED")

            if self.remeaning_punishing_time <= 0:
                self.active_algo_idx = self.coop_algo_idx
        return outputs

    def get_log_values(self):
        self.to_log = {
            "d_carac": round(float((self.algo_temp_info['log_defection_carac_tot'] /
                            (self.algo_temp_info['log_defection_carac_steps'] + self.epsilon))),4)
        }

        if self.algo_temp_info['log_coop_time'] == 0 and self.algo_temp_info['log_punishement_time'] == 0:
            self.to_log["coop_frac"] = 0.5
        else:
            self.to_log["coop_frac"] = round(float((self.algo_temp_info['log_coop_time'] /
                                                    (self.algo_temp_info['log_punishement_time'] +
                                                                   self.algo_temp_info['log_coop_time']))), 2)

        # Log actions prod distrib
        if self.agent.body.action_space_is_discrete:
            action_pd_coop = list(copy.deepcopy(self.action_pd_coop))
            action_pd_punish = list(copy.deepcopy(self.action_pd_punish))
            for act_idx in range(self.agent.body.action_dim):
                n_action_i = sum([el[act_idx] for el in action_pd_coop])
                self.to_log[f'ca{act_idx}'] = round(float(n_action_i /
                                                          (len(action_pd_coop) + self.epsilon)), 2)
            for act_idx in range(self.agent.body.action_dim):
                n_action_i = sum([el[act_idx] for el in action_pd_punish])
                self.to_log[f'pa{act_idx}'] = round(float(n_action_i /
                                                          (len(action_pd_punish) + self.epsilon)), 2)
        else:
            raise NotImplementedError()


        return super().get_log_values()

        # for idx, algo in enumerate(self.algorithms):
        #     for k, v in algo.get_extra_training_log_info().items():
        #         k_meta = f'{k}_alg{idx}'
        #         assert k_meta not in self.to_log.keys()
        #         self.to_log[k_meta] = v
        #
        # self._reset_temp_info()
        # extra_training_info_to_log = self.to_log
        # self.to_log = {}
        # return extra_training_info_to_log
