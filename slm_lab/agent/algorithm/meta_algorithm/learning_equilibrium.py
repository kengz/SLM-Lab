import random
from collections import deque

import numpy as np
import torch
from torch.distributions.kl import kl_divergence

from slm_lab.agent.agent import agent_util
from slm_lab.agent.algorithm import meta_algorithm
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


def compare_two_identical_net(net1, net2, fn=lambda x: x.norm(p=2)):
    aggregated_value = 0
    for (n1, p1), (n2, p2) in zip(net1.named_parameters(), net2.named_parameters()):
        aggregated_value += fn(p1.data - p2.data)
    return aggregated_value


def copy_weights_between_networks(copy_from_net, copy_to_net):
    copy_to_net.load_state_dict(copy_from_net.state_dict())
    compare_two_identical_net(copy_to_net, copy_to_net)


def plot_hist(value_list, save_to):
    import matplotlib.pyplot as plt
    plt.hist(value_list, histtype='step', stacked=False, fill=False, bins=30)
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.savefig(save_to)
    plt.close()


class LE(meta_algorithm.OneOfNAlgoActived):
    """
    Env must returns symmetrical agent states (giving agent 2 state to agent 1 should be fine)
    """

    # TODO add docs
    # TODO make it work for epi (how to divide epi when defection in during an epi ? (=> wait for end ?)

    def __init__(self, agent, global_nets, algorithm_spec,
                 memory_spec, net_spec, algo_idx=0):

        super().__init__(agent, global_nets, algorithm_spec, memory_spec, net_spec, algo_idx)

        util.set_attr(self, dict(
            defection_detection_mode="action_pd_kullback_leibler_div",
            punishement_time=10,
            min_coop_time=0,
            defection_carac_threshold=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'defection_detection_mode',
            'punishement_time',
            'min_coop_time',
            'defection_carac_threshold',
            'opp_policy_from_history',
            'opp_policy_from_supervised_learning',
            'policy_approx_batch_size'
        ])

        self.all_defection_detection_modes = ["network_weights",
                                              "action_pd_kullback_leibler_div",
                                              "observed_actions",
                                              "spl_observed_actions"]

        log_len_in_steps = 100

        assert self.defection_detection_mode in self.all_defection_detection_modes
        if self.defection_detection_mode == self.all_defection_detection_modes[0]:
            assert len(self.algorithms) == 3
            self.is_fully_init = False
            self.action_pd_opp_coop = deque(maxlen=log_len_in_steps)
            self.action_pd_opp = deque(maxlen=log_len_in_steps)
        elif self.defection_detection_mode == self.all_defection_detection_modes[1]:
            assert len(self.algorithms) == 2, str(len(self.algorithms))
            self.is_fully_init = True
        elif self.defection_detection_mode == self.all_defection_detection_modes[2]:
            assert len(self.algorithms) == 3, str(len(self.algorithms))
            self.is_fully_init = False
            self.opp_policy_from_history=True
            self.opp_policy_from_supervised_learning=False
            self.action_pd_opp_coop = deque(maxlen=log_len_in_steps)
            self.action_pd_opp = deque(maxlen=log_len_in_steps)
        elif self.defection_detection_mode == self.all_defection_detection_modes[3]:
            assert len(self.algorithms) == 4, str(len(self.algorithms))
            self.is_fully_init = False
            self.opp_policy_from_history=False
            self.opp_policy_from_supervised_learning=True
            self.action_pd_opp_coop = deque(maxlen=log_len_in_steps)
            self.action_pd_opp = deque(maxlen=log_len_in_steps)
            self.action_pd_opp_approx = deque(maxlen=log_len_in_steps)

        self.coop_algo_idx = 0
        self.punish_algo_idx = 1

        self.remeaning_punishing_time = 0
        self.detected_defection = False

        self.defection_carac_tot = 0
        self.defection_carac_steps = 0

        self.action_pd_coop = deque(maxlen=log_len_in_steps)
        self.action_pd_punish = deque(maxlen=log_len_in_steps)


        self.algo_temp_info = {
            "log_coop_time": 0,
            "log_punishement_time": 0,
            "log_defection_carac_tot": 0,
            "log_defection_carac_steps": 0
        }
        self.epsilon = 1e-6

        self.debug = False

        self.new_improved_perf = True

    def act(self, state):
        action, action_pd = self.algorithms[self.active_algo_idx].act(state)

        # To log action prob distrib
        if self.active_algo_idx == self.coop_algo_idx:
            self.action_pd_coop.append(action_pd.probs[0])
        elif self.active_algo_idx == self.punish_algo_idx:
            self.action_pd_punish.append(action_pd.probs[0])

        self.last_used_algo = self.active_algo_idx

        if self.debug:
            logger.info(f"action_pd {self.agent.agent_idx} {action_pd.probs}")

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
            if self.remeaning_punishing_time <= - (self.min_coop_time - 1):
                return self.defection_from_network_weights(state, action, welfare, next_state, done)
            return True
        elif self.defection_detection_mode == self.all_defection_detection_modes[1]:
            if self.remeaning_punishing_time <= - (self.min_coop_time - 1):
                return self.defection_from_action_pd_kl_div(state, action, welfare, next_state, done)
            return True
        elif (self.defection_detection_mode == self.all_defection_detection_modes[2] or
            self.defection_detection_mode  == self.all_defection_detection_modes[3]):
            return self.defection_from_observed_actions(state, action, welfare, next_state, done)
        else:
            raise NotImplementedError()

    def hash_fn(self, object):
        if isinstance(object, np.ndarray):
            v = str(object.tolist())
        else:
            v = str(object)
        return v

    def approximate_policy_from_history(self, opp_idx):
        s_coop_prob_dict = {}
        opp_data = list(self.data_queue[opp_idx])

        for v in opp_data:
            _, state_hash, action_hash, _ = v

            if state_hash not in s_coop_prob_dict.keys():
                s_coop_prob_dict[state_hash] = {"n_occurences": 0}
            s_coop_prob_dict[state_hash]["n_occurences"] += 1

            if action_hash not in s_coop_prob_dict[state_hash].keys():
                s_coop_prob_dict[state_hash][action_hash] = {"n_occurences": 0,
                                                             "a_proba": 0,
                                                             }  # "a": a}
            s_coop_prob_dict[state_hash][action_hash]["n_occurences"] += 1

        results = {}
        for state_hash in s_coop_prob_dict.keys():
            for action_hash in s_coop_prob_dict[state_hash].keys():
                if action_hash == "n_occurences":
                    continue

                if not self.new_improved_perf:
                    s_coop_prob_dict[state_hash][action_hash]["a_proba"] = (
                            s_coop_prob_dict[state_hash][action_hash]["n_occurences"] /
                            s_coop_prob_dict[state_hash]["n_occurences"]) + self.epsilon
                else:
                    results[state_hash+action_hash] = (
                            s_coop_prob_dict[state_hash][action_hash]["n_occurences"] /
                            s_coop_prob_dict[state_hash]["n_occurences"]) + self.epsilon

        if self.debug:
            logger.info(f"histo_opp_action_pc {self.agent.agent_idx} {s_coop_prob_dict}")

        if not self.new_improved_perf:
            return (lambda state_action_hash:
                    s_coop_prob_dict[state_action_hash[1]][state_action_hash[2]]["a_proba"])
        else:
            return results

    # @profile
    def defection_from_observed_actions(self, state, action, welfare, next_state, done):
        # TODO prob: this only works with 2 agents and with a discrete action space
        train = True

        if done:
            self.defection_carac_tot = 0
            self.defection_carac_steps = 0
            self.algo_temp_info['log_defection_carac_tot'] = 0
            self.algo_temp_info['log_defection_carac_steps'] = 0

        if not self.is_fully_init:
            self.n_steps_since_start = 0
            self.percentile = 95
            self.data_queue = []

            self.length_of_history = 200
            self.warmup_length = 0 # 200
            self.n_steps_in_bstrap_replts = 20  # 20
            self.n_bootstrapped_replications = 50  # 50

            self.last_computed_w = None

        my_r = agent_util.get_from_current_agents(self.agent, key="reward", default=None)

        # Get the observed data
        other_ag_action = agent_util.get_from_other_agents(self.agent, key="action", default=[])
        other_ag_states = agent_util.get_from_other_agents(self.agent, key="state", default=[])
        other_ag_rewards = agent_util.get_from_other_agents(self.agent, key="reward", default=[])
        other_ag_welfares = agent_util.get_from_other_agents(self.agent, key="welfare", default=[])
        other_ag_next_states = agent_util.get_from_other_agents(self.agent, key="next_state", default=[])
        other_ag_algorithms = agent_util.get_from_other_agents(self.agent, key="algorithm", default=[])

        # For each opponents (only tested for 1 opponent)
        for opp_idx, (s, a, r, w, n_s, algo) in enumerate(zip(other_ag_states, other_ag_action,
                                                           other_ag_rewards, other_ag_welfares,
                                                           other_ag_next_states,
                                                           other_ag_algorithms)):

            if not self.is_fully_init:
                self.data_queue.append(deque(maxlen=self.length_of_history))

            if self.opp_policy_from_supervised_learning:
                coop_net_simul_opponent_idx = self.punish_algo_idx + 2* opp_idx + 1
                approx_net_opponent_policy_idx = self.punish_algo_idx + 2* opp_idx + 2

            else:
                coop_net_simul_opponent_idx = self.punish_algo_idx + opp_idx + 1

            if not (isinstance(algo, LE) and algo.last_used_algo == algo.punish_algo_idx):
                # The opponent agent not is currenlty in the punish "state"

                # Get the log_likelihood of the observed action under the simulated opponent coop policy
                with torch.no_grad():
                    _, opponent_action_prob_distrib = self.algorithms[coop_net_simul_opponent_idx].act(s)
                # print("opponent_action_prob_distrib.probs.shape", opponent_action_prob_distrib.probs.shape)
                opponent_action_prob_distrib = opponent_action_prob_distrib.probs[0, ...]
                self.action_pd_opp_coop.append(opponent_action_prob_distrib)
                opp_true_prob_distrib = algo.agent.action_pd.probs[0, ...]
                self.action_pd_opp.append(opp_true_prob_distrib)
                if self.debug:
                    logger.info(f"coop_opp_action_pc {self.agent.agent_idx} {opponent_action_prob_distrib}")
                opponent_observed_action_index = a
                log_likelihood_opponent_cooporating = np.log(np.array(opponent_action_prob_distrib[
                                                                          opponent_observed_action_index]+ self.epsilon
                                                                      , dtype=np.float32))

                if self.opp_policy_from_supervised_learning:
                    with torch.no_grad():
                        _, opponent_action_prob_distrib = self.algorithms[approx_net_opponent_policy_idx].act(s)
                    # print("opponent_action_prob_distrib.probs.shape",opponent_action_prob_distrib.probs.shape)
                    opponent_action_prob_distrib = opponent_action_prob_distrib.probs[0, ...]
                    self.action_pd_opp_approx.append(opponent_action_prob_distrib)
                    opponent_observed_action_index = a
                    log_likelihood_approximated_opponent = np.log(np.array(opponent_action_prob_distrib[
                                                                              opponent_observed_action_index]+ self.epsilon,
                                                                          dtype=np.float32))

                # Store for later
                if self.opp_policy_from_history:
                    self.data_queue[opp_idx].append([log_likelihood_opponent_cooporating,
                                                     self.hash_fn(s),
                                                     self.hash_fn(a),
                                                     self.hash_fn(s)+self.hash_fn(a)])
                elif self.opp_policy_from_supervised_learning:

                    self.data_queue[opp_idx].append([log_likelihood_opponent_cooporating,
                                                     self.hash_fn(s),
                                                     self.hash_fn(a),
                                                     self.hash_fn(s)+self.hash_fn(a),
                                                     log_likelihood_approximated_opponent])

                self.n_steps_since_start += 1
                if self.debug:
                    logger.info(f"update queues {self.agent.agent_idx}")

                # Update the coop networks simulating the opponents
                computed_w = self.agent.welfare_function(algo.agent, r)
                self.last_computed_w = computed_w
                self.algorithms[coop_net_simul_opponent_idx].memory_update(s, a, computed_w, n_s, done)

                if self.opp_policy_from_supervised_learning:
                    # Update the networks learning the actual opponents policy (with supervised learning)
                    self.algorithms[approx_net_opponent_policy_idx].memory_update(s, a, None, None, done)

            if done and self.n_steps_since_start >= self.length_of_history + self.warmup_length:
                if not self.new_improved_perf:
                    data_queue_list = list(self.data_queue[opp_idx])
                    bstrap_replts_data = [random.choices(population=data_queue_list,
                                                         k=self.n_steps_in_bstrap_replts)
                                          for i in range(self.n_bootstrapped_replications)]
                else:
                    data_array = np.array(list(self.data_queue[opp_idx]), dtype=np.object)
                    bstrap_idx = np.random.random_integers(0, high=data_array.shape[0]-1,
                                                           size=(self.n_bootstrapped_replications,
                                                                 self.n_steps_in_bstrap_replts))
                    bstrap_replts_data = data_array[bstrap_idx]
                # Sum log_likelihood over u steps
                if not self.new_improved_perf:
                    log_lik_cooperate = np.array([[el[0] for el in one_replicat]
                                                  for one_replicat in bstrap_replts_data]).sum(axis=1)
                else:
                    log_lik_cooperate = bstrap_replts_data[:, :, 0].sum(axis=1)

                # Get the log_likelihood of the observed actions under the computed opponent policy
                if self.opp_policy_from_history:
                    opp_historical_policy = self.approximate_policy_from_history(opp_idx)
                    if not self.new_improved_perf:
                        bstrap_replts_log_lik_defect = np.log(np.array([[opp_historical_policy(data)
                                                                         for data in bs_data]
                                                                        for bs_data in bstrap_replts_data]))
                        # Sum log_likelihood over u steps
                        log_lik_defect = [sum(e) for e in bstrap_replts_log_lik_defect]
                    else:
                        bstrap_replts_log_lik_defect = np.log(np.array([[opp_historical_policy[data[3]]
                                                                         for data in bs_data]
                                                                        for bs_data in bstrap_replts_data]))
                        # Sum log_likelihood over u steps
                        log_lik_defect = bstrap_replts_log_lik_defect.sum(axis=1)

                elif self.opp_policy_from_supervised_learning:
                    # Sum log_likelihood over u steps
                    if not self.new_improved_perf:
                        log_lik_defect = np.array([[el[4] for el in one_replicat]
                                                      for one_replicat in bstrap_replts_data]).sum(axis=1)
                    else:
                        log_lik_defect = bstrap_replts_data[:, :, 4].sum(axis=1)

                else:
                    raise NotImplementedError()

                # Defect if in more than 0.95 of the replicates, the actual policy is more likely than the simulated coop policy
                if not self.new_improved_perf:
                    log_lik_check_coop = np.array(log_lik_cooperate) - np.array(log_lik_defect)
                else:
                    log_lik_check_coop = log_lik_cooperate - log_lik_defect
                assert len(log_lik_check_coop) == self.n_bootstrapped_replications
                percentile_value = np.percentile(log_lik_check_coop, self.percentile, interpolation="linear")

                self.to_log.update({
                    "log_lik_check_coop_std":log_lik_check_coop.std(),
                    "log_lik_check_coop_mean":log_lik_check_coop.mean()
                })


                if self.debug:
                    logger.info(log_lik_check_coop.shape)
                    logger.info(f"log_lik_cooperate {self.agent.agent_idx} {log_lik_cooperate[0:5]}")
                    logger.info(f"log_lik_defect { self.agent.agent_idx} {log_lik_defect[0:5]}")
                    logger.info(f"log_lik_check_coop {self.agent.agent_idx} {log_lik_check_coop[0:5]}")
                    logger.info(
                        f"percentile_value {self.agent.agent_idx} {percentile_value} {log_lik_check_coop.min()} {log_lik_check_coop.mean()} {log_lik_check_coop.max()}")
                    if self.n_steps_since_start % 500 == 0:
                        if not hasattr(self, "to_plot"):
                            self.to_plot = np.expand_dims(log_lik_check_coop, axis=1)
                        else:
                            self.to_plot = np.concatenate([self.to_plot, np.expand_dims(log_lik_check_coop, axis=1)],
                                                          axis=1)
                        plot_hist(value_list=self.to_plot,
                                  save_to=f"./diff_distrib_{self.agent.agent_idx}__{self.n_steps_since_start}.png")
                self.defection_carac_tot += -percentile_value
                self.defection_carac_steps += 1
                self.algo_temp_info['log_defection_carac_tot'] += -percentile_value
                self.algo_temp_info['log_defection_carac_steps'] += 1

        if not self.is_fully_init:
            self.is_fully_init = True

        return train

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

            coop_net_simul_opponent_idx = self.punish_algo_idx + idx + 1
            # TODO is not currently shared between agents because it is only computed in update (agent sequential)
            if not self.is_fully_init:
                logger.info("LE algo finishing init by copying weight from opponent network")
                if isinstance(algo, meta_algorithm.LE):
                    net_from = algo.algorithms[algo.coop_algo_idx].net
                else:
                    net_from = algo.net
                copy_weights_between_networks(copy_from_net=net_from,
                                              copy_to_net=self.algorithms[coop_net_simul_opponent_idx].net)

            # Recompute welfare using the currently agent welfare function
            w = self.agent.welfare_function(algo.agent, r)
            self.algorithms[coop_net_simul_opponent_idx].memory_update(
                s, a, w, n_s, done)

            diff = compare_two_identical_net(self.algorithms[coop_net_simul_opponent_idx].net,
                                             algo.net)
            self.defection_carac_tot += diff
            self.defection_carac_steps += 1
            self.algo_temp_info['log_defection_carac_tot'] += diff
            self.algo_temp_info['log_defection_carac_steps'] += 1

        if not self.is_fully_init:
            self.is_fully_init = True

        return True

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

                self.defection_carac_tot += kl_div
                self.defection_carac_steps += 1
                self.algo_temp_info['log_defection_carac_tot'] += kl_div
                self.algo_temp_info['log_defection_carac_steps'] += 1
        return True

    @lab_api
    def memory_update(self, state, action, welfare, next_state, done):

        # start at the last step before the end of min_coop_time
        train = self.detect_defection(state, action, welfare, next_state, done)

        assert (self.remeaning_punishing_time > 0) == (self.active_algo_idx == self.punish_algo_idx)
        assert (self.remeaning_punishing_time <= 0) == (self.active_algo_idx == self.coop_algo_idx)

        if np.isnan(self.algo_temp_info['log_punishement_time']):
            self.algo_temp_info['log_punishement_time'] = 0
            self.algo_temp_info['log_coop_time'] = 0

        if self.remeaning_punishing_time > 0:
            other_agents_rewards = agent_util.get_from_other_agents(self.agent, key="reward", default=[])
            welfare = 1 - sum(other_agents_rewards)
            self.algo_temp_info['log_punishement_time'] += 1
        else:
            self.algo_temp_info['log_coop_time'] += 1

        outputs = None
        if train:
            outputs = self.algorithms[self.active_algo_idx].memory_update(state, action, welfare,
                                                                                      next_state, done)

        if done:

            if self.remeaning_punishing_time <= - (self.min_coop_time - 1):
                if self.defection_carac_tot / (
                        self.defection_carac_steps + self.epsilon) > self.defection_carac_threshold:
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
                                    (self.algo_temp_info['log_defection_carac_steps'] + self.epsilon))), 4)
        }

        if self.algo_temp_info['log_coop_time'] == 0 and self.algo_temp_info['log_punishement_time'] == 0:
            self.to_log["coop_frac"] = 0.5
        else:
            self.to_log["coop_frac"] = round(float((self.algo_temp_info['log_coop_time'] /
                                                    (self.algo_temp_info['log_punishement_time'] +
                                                     self.algo_temp_info['log_coop_time']))), 2)

        # Log actions prod distrib
        if self.agent.body.action_space_is_discrete:
            if self.defection_detection_mode == self.all_defection_detection_modes[1]:
                action_pd_coop = list(self.action_pd_coop)
                action_pd_punish = list(self.action_pd_punish)
                for act_idx in range(self.agent.body.action_dim):
                    n_action_i = sum([el[act_idx] for el in action_pd_coop])
                    self.to_log[f'ca{act_idx}'] = round(float(n_action_i /
                                                              (len(action_pd_coop) + self.epsilon)), 2)
                for act_idx in range(self.agent.body.action_dim):
                    n_action_i = sum([el[act_idx] for el in action_pd_punish])
                    self.to_log[f'pa{act_idx}'] = round(float(n_action_i /
                                                              (len(action_pd_punish) + self.epsilon)), 2)
            else:
                action_pd_opp = list(self.action_pd_opp)
                for act_idx in range(self.agent.body.action_dim):
                    n_action_i = sum([el[act_idx] for el in action_pd_opp])
                    self.to_log[f'oppa{act_idx}'] = round(float(n_action_i /
                                                              (len(action_pd_opp) + self.epsilon)), 2)

            # if (self.defection_detection_mode == self.all_defection_detection_modes[0] or
            #         self.defection_detection_mode == self.all_defection_detection_modes[2]):
                action_pd_opp_coop = list(self.action_pd_opp_coop)
                for act_idx in range(self.agent.body.action_dim):
                    n_action_i = sum([el[act_idx] for el in action_pd_opp_coop])
                    self.to_log[f'sim_ca{act_idx}'] = round(float(n_action_i /
                                                              (len(action_pd_opp_coop) + self.epsilon)), 2)

            if self.defection_detection_mode == self.all_defection_detection_modes[3]:
                action_pd_approx_opp = list(self.action_pd_opp_approx)
                for act_idx in range(self.agent.body.action_dim):
                    n_action_i = sum([el[act_idx] for el in action_pd_approx_opp])
                    self.to_log[f'spl_ca{act_idx}'] = round(float(n_action_i /
                                                               (len(action_pd_approx_opp) + self.epsilon)), 2)

        else:
            raise NotImplementedError()

        return super().get_log_values()
