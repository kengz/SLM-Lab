import random
from collections import deque
import copy
import numpy as np
import pydash as ps
import torch
from torch.distributions.kl import kl_divergence

from slm_lab.agent import memory
from slm_lab.agent.agent import agent_util
from slm_lab.agent.algorithm import meta_algorithm
from slm_lab.agent.algorithm import policy_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from slm_lab.lib.util import PID
from slm_lab.lib import math_util, util
from slm_lab.agent.net import net_util

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
    The base logging behavior is averaging over one episode.
    """

    # TODO add docs
    # TODO make it work for epi (how to divide epi when defection in during an epi ? (=> wait for end ?)

    def __init__(self, agent, global_nets, algorithm_spec,
                 memory_spec, net_spec, algo_idx=0):

        self.algorithms = []
        super().__init__(agent, global_nets, algorithm_spec, memory_spec, net_spec, algo_idx)

        util.set_attr(self, dict(
            defection_detection_mode="action_pd_kullback_leibler_div",
            punishement_time=10,
            min_cooperative_epi_after_punishement=0,
            defection_carac_threshold=1.0,
            average_d_carac=True,
            average_d_carac_len=20,
            coop_net_auxloss_ent_diff=False,
            coop_net_auxloss_ent_diff_coeff=1.0,
            coop_net_ent_diff_as_lr=False,
            coop_net_ent_diff_as_lr_coeff=1 / 0.7 * 30,
            spl_net_ent_diff_as_lr=False,
            spl_net_ent_diff_as_lr_coeff=1.0,
            use_sl_for_simu_coop=False,
            use_strat_4=False,
            use_strat_5=False,
            strat_5_coeff=10,
            use_strat_2=False,
            use_gene_algo=True,
            n_gene_algo=10,
            same_init_weights=False,
            block_len=None
        ))
        util.set_attr(self, self.algorithm_spec, [
            'defection_detection_mode',
            'punishement_time',
            'min_cooperative_epi_after_punishement',
            'defection_carac_threshold',
            'opp_policy_from_history',
            'opp_policy_from_supervised_learning',
            'policy_approx_batch_size',
            'average_d_carac',
            'average_d_carac_len',
            'coop_net_auxloss_ent_diff',
            'coop_net_auxloss_ent_diff_coeff',
            'coop_net_ent_diff_as_lr',
            'coop_net_ent_diff_as_lr_coeff',
            'spl_net_ent_diff_as_lr',
            'spl_net_ent_diff_as_lr_coeff',
            "use_sl_for_simu_coop",
            "use_strat_4",
            "use_strat_5",
            "strat_5_coeff",
            "use_strat_2",
            "use_gene_algo",
            "n_gene_algo",
            "same_init_weights",
            "block_len"
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
            self.opp_policy_from_history = True
            self.opp_policy_from_supervised_learning = False
            self.action_pd_opp_coop = deque(maxlen=log_len_in_steps)
            self.action_pd_opp = deque(maxlen=log_len_in_steps)
        elif self.defection_detection_mode == self.all_defection_detection_modes[3]:
            assert len(self.algorithms) == 4, str(len(self.algorithms))
            self.is_fully_init = False
            self.opp_policy_from_history = False
            self.opp_policy_from_supervised_learning = True
            self.action_pd_opp_coop = deque(maxlen=log_len_in_steps)
            self.action_pd_opp = deque(maxlen=log_len_in_steps)
            self.action_pd_opp_approx = deque(maxlen=log_len_in_steps)

            # if not self.is_fully_init:
            self.n_steps_since_start = 0
            self.percentile = 95
            self.data_queue = []

            self.length_of_history = 200
            self.warmup_length = 0  # 200
            self.n_steps_in_bstrap_replts = 20  # 20
            self.n_bootstrapped_replications = 50  # 50
            # if self.use_strat_4:
            #     self.n_steps_in_bstrap_replts = 60  # 20
            #     self.n_bootstrapped_replications = 200  # 50

            self.last_computed_w = None

        self.coop_algo_idx = 0
        self.punish_algo_idx = 1

        self.remeaning_punishing_time = 0
        self.detected_defection = False

        # Defection metric
        # self.defection_metric_total = 0
        # self.defection_metric_steps = 0
        self.defection_metric = 0
        if not self.average_d_carac:
            self.average_d_carac_len = 1
        self.defection_carac_queue = deque(maxlen=self.average_d_carac_len)

        self.action_pd_coop = deque(maxlen=log_len_in_steps)
        self.action_pd_punish = deque(maxlen=log_len_in_steps)

        self.epsilon = 1e-12
        self.debug = False
        # Modif/Improvements
        self.new_improved_perf = True

        self.use_historical_policy_as_target = False

        if self.coop_net_ent_diff_as_lr:
            self.entropy_diff_queue_len = 5
            self.entropy_diff_queue = deque(maxlen=self.entropy_diff_queue_len)
            self.base_lr_value_coop_net = self.algorithms[self.coop_algo_idx].lr_scheduler.get_lr()

            self.negative_warmup_len = 100
            self.pid_time = 0
            self.pid = PID(P=-100.0, I=-0.05, D=-20.0, current_time=self.pid_time + 1, verbose=True, )
            self.pid.setWindup(10)
            self.correction_len = 20
            self.correction = deque(maxlen=self.correction_len)

            self.max_lr = 0.005

        if self.use_strat_4:
            self.counter = 0

        if self.use_strat_2:
            self.memory_spec = self.meta_algorithm_spec["meta_algo_memory"]
            if self.memory_spec is not None:
                MemoryClass = getattr(memory, ps.get(self.memory_spec, 'name'))
                self.memory = MemoryClass(self.memory_spec, self)
            else:
                self.memory = None
            self.training_start_step = 0
            self.to_train = 0

            self.spl_loss_fn = net_util.get_loss_fn(self, self.meta_algorithm_spec["meta_algo_loss"])

            self.best_lr = 0

            # self.lr_perturbation = 2.0
            self.lr_perturbation = 1.5
            # self.lr_perturbation = 1.25
            assert self.block_len is not None
            # self.block_len = 72
            self.block_pos = 0

            self.best_lr = -1

            self.n_algo_in_strat_2 = 10
            self.strat_2_data = [{"lr":-1, "steps_seen":0,
                                 # "spl_losses":deque(maxlen=20),
                                  "spl_losses": [],
                                  "age":0,
                                  "algo":None,
                                  "log_likelihood":deque(maxlen=self.length_of_history)}
                                 for i in range(self.n_algo_in_strat_2)]
            # self.spl_list_list = [None]*self.n_algo_in_strat_2
            # self.algo_list = [None]*self.n_algo_in_strat_2
            # self.log_likelihood_list = [deque(maxlen=self.length_of_history)]*self.n_algo_in_strat_2
            self.best_lr_idx = 0
            self.algo_to_kill_idx = 0

            self.kill_oldest_every_n = 10
            self.strat_2_counter = 0
            self.strat_2_is_init = False
            self.strat_2_epsilon = 1e-3


        # Logging
        self.n_cooperation_steps = 0
        self.n_punishement_steps = 0


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
        # if not isinstance(self.defection_metric, torch.Tensor) and np.isnan(self.defection_metric):
        #     self.defection_metric = 0

        if self.defection_detection_mode == self.all_defection_detection_modes[0]:
            return self.defection_from_network_weights(state, action, welfare, next_state, done)
        elif (self.defection_detection_mode == self.all_defection_detection_modes[2] or
              self.defection_detection_mode == self.all_defection_detection_modes[3]):
            return self.defection_from_observed_actions(state, action, welfare, next_state, done)
        else:
            raise NotImplementedError()

    def hash_fn(self, object):
        if isinstance(object, np.ndarray):
            v = str(object.tolist())
        else:
            v = str(object)
        return v

    def approximate_policy_from_history(self, opp_idx, separate_actions=True):
        s_coop_prob_dict = {}
        opp_data = list(self.data_queue[opp_idx])

        for v in opp_data:
            if self.opp_policy_from_supervised_learning:
                _, state_hash, action_hash, _, _ = v
            else:
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

                v = (s_coop_prob_dict[state_hash][action_hash]["n_occurences"] /
                     s_coop_prob_dict[state_hash]["n_occurences"])
                if v == 0.0:
                    v = self.epsilon
                if not separate_actions:
                    if state_hash not in results:
                        if not self.agent.body.action_space_is_discrete:
                            raise NotImplementedError()
                        results[state_hash] = np.zeros(shape=self.agent.body.action_dim)

                    a = int(action_hash)
                    results[state_hash][a] = v
                else:
                    results[state_hash + action_hash] = v

        if self.debug:
            logger.info(f"histo_opp_action_pc {self.agent.agent_idx} {s_coop_prob_dict}")

        return results

    def defection_from_observed_actions(self, state, action, welfare, next_state, done):
        # TODO prob: this currently only works with 2 agents and with a discrete action space
        train = True

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

            if self.opp_policy_from_supervised_learning:
                self.coop_net_simul_opponent_idx = self.punish_algo_idx + 2 * opp_idx + 1
                self.approx_net_opponent_policy_idx = self.punish_algo_idx + 2 * opp_idx + 2
            else:
                self.coop_net_simul_opponent_idx = self.punish_algo_idx + opp_idx + 1

            if not self.is_fully_init:
                self.data_queue.append(deque(maxlen=self.length_of_history))
                if self.same_init_weights:
                    logger.info("LE algo finishing init by copying weight between simu coop and opp approx")
                    copy_weights_between_networks(copy_from_net=self.algorithms[self.approx_net_opponent_policy_idx].net,
                                                  copy_to_net=self.algorithms[self.coop_net_simul_opponent_idx].net)

            if not (isinstance(algo, LE) and algo.last_used_algo == algo.punish_algo_idx):
                # The opponent agent not is currenlty in the punish "state"

                (self.log_likelihood_opponent_cooporating,
                 self.opp_coop_a_prob_distrib) = (self.compute_s_a_log_likelihood(s, a,
                                                          algo=self.algorithms[self.coop_net_simul_opponent_idx],
                                                          no_grad = self.use_strat_5))

                opp_true_prob_distrib = algo.agent.action_pd.probs[0, ...].detach()
                self.action_pd_opp.append(opp_true_prob_distrib)

                if self.opp_policy_from_supervised_learning:
                    (self.log_likelihood_approximated_opponent,
                     self.opp_spl_a_prob_distrib) = self.compute_s_a_log_likelihood(s, a,
                                                            algo=self.algorithms[self.approx_net_opponent_policy_idx])


                self.ipm_store_temp_data(opp_idx, s, a)
                self.ipm_memory_update(opp_idx, algo, s, a, r, n_s, done)
                self.apply_ipm_options_every_steps()

            if done and self.use_strat_2:
                self.block_pos += 1
                if self.block_pos % self.block_len == 0:
                    self.train_simu_coop_from_scratch(opp_idx)

            if done and self.n_steps_since_start >= self.length_of_history + self.warmup_length:
                percentile_value = self.compare_log_likelihood_on_sequence(opp_idx)
                self._update_defection_metric(epi_defection_metric=-percentile_value)

        if not self.is_fully_init:
            self.is_fully_init = True

        return train

    def compute_s_a_log_likelihood(self, s, a, algo, no_grad=True):
        """ compute log_likelihood(s,a)_under_algo_policy """
        # Get the log_likelihood of the observed action under the algo policy
        if no_grad:
            with torch.no_grad():
                _, a_prob_distrib = algo.act(s)
        else:
            _, a_prob_distrib = algo.act(s)
        opp_coop_a_probs = a_prob_distrib.probs[0, ...].detach()
        self.action_pd_opp_coop.append(opp_coop_a_probs)

        if self.debug:
            logger.info(f"coop_opp_action_pc {self.agent.agent_idx} {opp_coop_a_probs}")
        opponent_observed_action_index = a
        s_a_log_likelihood_under_algo_policy = np.log(np.array(opp_coop_a_probs[
                                                                  opponent_observed_action_index] + self.epsilon
                                                              , dtype=np.float32))
        return s_a_log_likelihood_under_algo_policy, a_prob_distrib

    def ipm_store_temp_data(self, opp_idx, s, a):
        self.n_steps_since_start += 1
        if self.opp_policy_from_history:
            self.data_queue[opp_idx].append([self.log_likelihood_opponent_cooporating,
                                             self.hash_fn(s),
                                             self.hash_fn(a),
                                             self.hash_fn(s) + self.hash_fn(a)])
        elif self.opp_policy_from_supervised_learning:
            # if self.use_strat_2:
            #     self.log_likelihood_opponent_cooporating = None
            self.data_queue[opp_idx].append([self.log_likelihood_opponent_cooporating,
                                             self.hash_fn(s),
                                             self.hash_fn(a),
                                             self.hash_fn(s) + self.hash_fn(a),
                                             self.log_likelihood_approximated_opponent])

    def ipm_memory_update(self, opp_idx, algo, s, a, r, n_s, done):
        # print("ipm_memory_update s, a, w, n_s, done", s, a, r, n_s, done)
        # print("s, a, w, n_s, done", type(s), type(a), type(r), type(n_s), type(done))

        # Update the coop networks simulating the opponents
        computed_w = self.agent.welfare_function(algo.agent, r)
        self.last_computed_w = computed_w
        # if self.use_sl_for_simu_coop:
        #     if hasattr(self, "simu_coop_w_n_minus_1"):
        #         current_net_params = dict(self.algorithms[coop_net_simul_opponent_idx].net.named_parameters())
        #         penalty = sum((current_net_params[k] - (v + )).abs().sum()
        #                       for k, v in self.simu_coop_w_n_minus_1.items())
        #         penalty *= self.penalty_coeff
        #     else:
        #         penalty = 0
        #     self.algorithms[coop_net_simul_opponent_idx].memory_update(s, a, penalty, None, done)
        #     if self.algorithms[coop_net_simul_opponent_idx].to_train == 1:
        #         self.simu_coop_w_n_minus_1 = { k:v.clone() for k,v in self.algorithms[
        #             coop_net_simul_opponent_idx].net.named_parameters()}
        # else:
        if self.opp_policy_from_supervised_learning:
            if self.use_historical_policy_as_target:
                self.opp_historical_policy = self.approximate_policy_from_history(opp_idx, separate_actions=False)

        if self.use_sl_for_simu_coop:
            self.algorithms[self.coop_net_simul_opponent_idx].memory_update(s, a, computed_w, n_s, done, idx=1)
        elif self.use_strat_2:
            self.memory.update(s, a, computed_w, n_s, done)
        else:
            self.algorithms[self.coop_net_simul_opponent_idx].memory_update(s, a, computed_w, n_s, done)

        if self.opp_policy_from_supervised_learning:

            # Update the networks learning the actual opponents policy (with supervised learning)
            if self.use_historical_policy_as_target:
                one_hot_target = self.opp_historical_policy[self.hash_fn(s)]
                # print("target",one_hot_target)
                self.algorithms[self.approx_net_opponent_policy_idx].memory_update(s, one_hot_target, None, None,
                                                                              done)
                if self.use_sl_for_simu_coop:
                    self.algorithms[self.coop_net_simul_opponent_idx].memory_update(s, one_hot_target, None, None,
                                                                               done, idx=0)
            else:
                self.algorithms[self.approx_net_opponent_policy_idx].memory_update(s, a, None, None, done)

    def apply_ipm_options_every_steps(self):
        if self.opp_policy_from_supervised_learning:

            # if self.spl_net_ent_diff_as_lr:
            #     entropy_diff = opp_spl_a_prob_distrib.entropy() - .entropy().detach()
            #     self.algorithms[approx_net_opponent_policy_idx].lr_overwritter = entropy_diff * \
            #                                                                  self.spl_net_ent_diff_as_lr_coeff

            # Options to modify the simulated coop net training
            # if self.coop_net_auxloss_ent_diff:
            #     if not hasattr(self.algorithms[coop_net_simul_opponent_idx], "auxilary_loss"):
            #         self.algorithms[coop_net_simul_opponent_idx].auxilary_loss = 0
            #     entropy_diff = opp_coop_a_prob_distrib.entropy() - opp_spl_a_prob_distrib.entropy().detach()
            #     self.algorithms[coop_net_simul_opponent_idx].auxilary_loss += (
            #             entropy_diff * self.coop_net_auxloss_ent_diff_coeff /
            #             self.algorithms[coop_net_simul_opponent_idx].training_frequency)

            if self.use_strat_4 and self.algorithms[self.coop_net_simul_opponent_idx].to_train == 1:
                self.counter += 1
                if self.counter == 10:
                    copy_weights_between_networks(copy_from_net=self.algorithms[self.approx_net_opponent_policy_idx].net,
                                                  copy_to_net=self.algorithms[self.coop_net_simul_opponent_idx].net)
                    self.counter = 0

            if self.use_strat_5 and self.algorithms[self.coop_net_simul_opponent_idx].to_train == 1:
                # ent_diff = (opp_coop_a_prob_distrib.entropy().mean() -
                #             opp_spl_a_prob_distrib.entropy().detach().mean() * 0.95 + 0.01) ** 2
                # self.algorithms[coop_net_simul_opponent_idx].auxilary_loss = ent_diff * self.strat_5_coeff

                # batch = self.algorithms[coop_net_simul_opponent_idx].sample(reset=False)
                # pd_param = self.algorithms[coop_net_simul_opponent_idx].calc_pdparam_batch(batch)
                # action_pd = policy_util.init_action_pd(self.algorithms[coop_net_simul_opponent_idx].body.ActionPD, pd_param)
                # coop_entropy = action_pd.entropy().mean()

                batch = self.algorithms[self.approx_net_opponent_policy_idx].sample(reset=False)
                pd_param = self.algorithms[self.approx_net_opponent_policy_idx].calc_pdparam_batch(batch)
                action_pd = policy_util.init_action_pd(self.algorithms[self.approx_net_opponent_policy_idx].body.ActionPD, pd_param)
                opp_approx_entropy = action_pd.entropy().mean()

                # ent_diff = (coop_entropy -
                #             (opp_approx_entropy.detach() * 0.95) + 0.01) ** 2
                # self.to_log["entropy_diff"] = ent_diff * self.strat_5_coeff
                # self.algorithms[coop_net_simul_opponent_idx].auxilary_loss = ent_diff * self.strat_5_coeff

                # def entop_diff():
                #     ent_diff = (coop_entropy -
                #             (opp_approx_entropy.detach() *0.95) + 0.01)**2
                #     return ent_diff * self.strat_5_coeff
                # self.algorithms[coop_net_simul_opponent_idx].auxilary_loss = entop_diff

                self.to_log['entropy_opp_approx'] = opp_approx_entropy
                self.algorithms[self.coop_net_simul_opponent_idx].auxilary_loss = opp_approx_entropy.detach()
                self.algorithms[self.coop_net_simul_opponent_idx].strat_5_coeff = self.strat_5_coeff

            if self.coop_net_ent_diff_as_lr:

                if self.algorithms[self.coop_net_simul_opponent_idx].to_train == 1:
                    entropy_diff = (self.opp_coop_a_prob_distrib.entropy().detach().mean() -
                                    self.opp_spl_a_prob_distrib.entropy().detach().mean() * 0.95 + 0.01)
                    self.to_log['entr_diff'] = entropy_diff.mean()
                    self.base_lr_value_coop_net = np.mean(self.algorithms[
                                                              self.coop_algo_idx].lr_scheduler.get_lr())
                    self.entropy_diff_queue.append(entropy_diff)
                    entropy_diff = sum(list(self.entropy_diff_queue)) / len(self.entropy_diff_queue)
                    self.pid_time += 1
                    self.pid.update(entropy_diff, current_time=self.pid_time)
                    lr_multiplier = (1 + self.pid.output)
                    lr_multiplier = lr_multiplier * abs(lr_multiplier)
                    self.lr_overwritter = ((self.base_lr_value_coop_net * lr_multiplier).item()) * (
                            min(self.pid_time, self.negative_warmup_len) / self.negative_warmup_len
                    )
                    self.lr_overwritter = min(self.lr_overwritter, self.max_lr)
                    self.lr_overwritter = max(self.lr_overwritter, -self.max_lr)
                    self.algorithms[self.coop_net_simul_opponent_idx].lr_overwritter = self.lr_overwritter

                if self.algorithms[
                    self.approx_net_opponent_policy_idx].to_train == 1 and self.active_algo_idx != self.punish_algo_idx:
                    if hasattr(self, "lr_overwritter"):
                        self.correction.append(self.lr_overwritter)
                        if len(self.correction) == self.correction_len:
                            lr = ((sum(list(self.correction)) / len(list(self.correction))) /
                                  np.mean(self.algorithms[self.coop_net_simul_opponent_idx].lr_scheduler.get_lr()))
                            lr = max(lr, 1)
                            lr = np.mean(self.algorithms[self.approx_net_opponent_policy_idx].lr_scheduler.get_lr(
                            )) * lr
                            self.algorithms[self.approx_net_opponent_policy_idx].lr_overwritter = lr

    def train_simu_coop_from_scratch(self, opp_idx):

        if not self.strat_2_is_init:
            self.save_init_weights = copy.deepcopy(self.algorithms[self.coop_net_simul_opponent_idx].net)

        # Next LR to test
        best_lr = self.strat_2_data[self.best_lr_idx]['lr']
        if best_lr != -1:
            new_lr = best_lr * np.random.uniform(low=1/self.lr_perturbation, high=self.lr_perturbation)
            # new_lr = best_lr
            # lrs = [data['lr'] for data in self.strat_2_data if data['algo'] is not None]
            # # print("torch.tensor(lrs).median()",torch.tensor(lrs).median())
            # if len(lrs) == self.n_algo_in_strat_2:
            #     new_lr = torch.tensor(lrs).median() * np.random.uniform(low=1/self.lr_perturbation,
            #                                                          high=self.lr_perturbation)
            # else:
            #     new_lr = best_lr * np.random.uniform(low=1/self.lr_perturbation, high=self.lr_perturbation)

            self.to_log['new_lr']= new_lr
            spec_to_modify = copy.deepcopy(self.meta_algorithm_spec['contained_algorithms'][
                                               self.coop_net_simul_opponent_idx]['net'])
            if spec_to_modify['lr_scheduler_spec'] is None or spec_to_modify['lr_scheduler_spec']["name"] == "LinearToZero":
                if 'optim_spec' in spec_to_modify.keys():
                    spec_to_modify['optim_spec']['lr'] = new_lr
                elif 'actor_optim_spec' in spec_to_modify.keys():
                    spec_to_modify['actor_optim_spec']['lr'] = new_lr
            elif spec_to_modify['lr_scheduler_spec']["name"] == "CyclicLR":
                current_base_lr = spec_to_modify['lr_scheduler_spec']['base_lr']
                current_max_lr = spec_to_modify['lr_scheduler_spec']['max_lr']
                spec_to_modify['lr_scheduler_spec'].update({"base_lr": new_lr * current_base_lr/current_max_lr,
                                                                   "max_lr": new_lr})
                # print("spec_to_modify['lr_scheduler_spec']",spec_to_modify['lr_scheduler_spec'])
            else:
                raise NotImplementedError()
            self.meta_algorithm_spec['contained_algorithms'][
                                               self.coop_net_simul_opponent_idx]['net'] = spec_to_modify

        else:
            spec = self.meta_algorithm_spec['contained_algorithms'][self.coop_net_simul_opponent_idx]['net']
            if spec['lr_scheduler_spec'] is None or spec['lr_scheduler_spec']["name"] == "LinearToZero":
                if 'optim_spec' in spec.keys():
                    lr = spec['optim_spec']['lr']
                elif 'actor_optim_spec' in spec.keys():
                    lr = spec['actor_optim_spec']['lr']
            elif spec['lr_scheduler_spec']["name"] == "CyclicLR":
                lr = spec['lr_scheduler_spec']['max_lr']
            else:
                raise NotImplementedError()

            new_lr = lr

        # Re-init algo
        # TODO improve this by using the global_nets to always init with the weights
        algo = self.deploy_contained_algo(global_nets=None,
                                   idx_selector=[self.coop_net_simul_opponent_idx])[0]
        copy_weights_between_networks(copy_from_net=self.save_init_weights,
                                      copy_to_net=algo.net)
        # print("init param", sum([ p.sum() for p in self.save_init_weights.parameters()]))
        # print("init param", sum([ p.sum() for p in algo.net.parameters()]))

        # if not self.strat_2_is_init or self.strat_2_counter < 5:
        self.strat_2_data[self.algo_to_kill_idx]['steps_seen'] = 0
        self.strat_2_data[self.algo_to_kill_idx]['algo'] = algo
        self.strat_2_data[self.algo_to_kill_idx]['lr'] = new_lr
        self.strat_2_data[self.algo_to_kill_idx]['spl_losses'] = []
        self.strat_2_data[self.algo_to_kill_idx]['log_likelihood'].clear()
        self.strat_2_data[self.algo_to_kill_idx]['age'] = 0

        # Train from strach
        # stored_data_len = len(self.data_queue[opp_idx])
        print("agent_idx", self.agent.agent_idx)
        print("New tested LR", new_lr)
        # print("self.memory.states", len(self.memory.states))
        for step_idx, data in enumerate(self.memory.replay_all_history()):
            s = data["states"]
            a = data["actions"]
            w = float(data["rewards"])
            n_s = data["next_states"]
            done = bool(data["dones"])

            for algo_idx, algo_data in enumerate(self.strat_2_data):
                if algo_data["algo"] is not None:
                    if step_idx >= algo_data['steps_seen']:
                        self.strat_2_data[algo_idx]['steps_seen'] += 1
                        self.strat_2_data[algo_idx]['algo'].memory_update(s, a, w, n_s, done)
                        if self.strat_2_data[algo_idx]['algo'].to_train == 1:
                            with torch.no_grad():
                                batch = self.strat_2_data[algo_idx]['algo'].sample(reset=False)
                                pdparams = self.strat_2_data[algo_idx]['algo'].calc_pdparam_batch(batch)
                                spl_loss = self.calc_supervised_learn_loss(batch, pdparams)
                                self.strat_2_data[algo_idx]['spl_losses'].append(spl_loss)
                                self.to_log[f"loss_spl_strat4_{algo_idx}"] = spl_loss
                            print("Train algo",algo_idx)

                        self.strat_2_data[algo_idx]['algo'].train()
                            # w = sum([ p.sum() for p in self.strat_2_data[algo_idx]['algo'].net.parameters()])
                            # print("Train algo",algo_idx, "weights", w)

                        self.strat_2_data[algo_idx]['algo'].update()

                        # Update data in the data queue to compare the log likelihood
                        if len(self.memory.states) - step_idx <= self.length_of_history:
                            (log_likelihood_opponent_cooporating,
                             prob_distrib) = (self.compute_s_a_log_likelihood(s, a,
                                                                   algo=self.strat_2_data[algo_idx]['algo'],
                                                                   no_grad=True))
                            self.to_log[f'entropy_strat_2_{algo_idx}'] = prob_distrib.entropy()
                            self.strat_2_data[algo_idx]['log_likelihood'].append(log_likelihood_opponent_cooporating)

        # Best
        log_likelihoods = [ len(data['log_likelihood']) for data in self.strat_2_data]
        steps_seen = [data['steps_seen'] for data in self.strat_2_data]
        lrs = [data['lr'] for data in self.strat_2_data]
        spl_losses = torch.tensor([torch.tensor(data['spl_losses']).mean() for data in self.strat_2_data if
                                        data['algo'] is not None])
        # spl_losses = torch.tensor([torch.tensor(data['spl_losses']).median() for data in self.strat_2_data if
        #                                 data['algo'] is not None])
        # spl_losses = torch.tensor([(torch.tensor(data['spl_losses'])+self.strat_2_epsilon).log2().mean() for data in
        #                            self.strat_2_data if
        #                            data['algo'] is not None])
        _, new_best_lr_idx = torch.min(spl_losses, dim=0)
        age = [data['age'] for data in self.strat_2_data]
        # To kill
        for algo_idx, algo_data in enumerate(self.strat_2_data):
            self.strat_2_data[algo_idx]['age'] += 1
        full = True
        for idx, lr in enumerate(lrs):
            if lr == -1:
                self.algo_to_kill_idx = idx
                full = False
                break
        self.strat_2_counter += 1
        if full:
            if self.strat_2_counter % self.kill_oldest_every_n == 0 and self.kill_oldest_every_n > 0:

                _, self.algo_to_kill_idx = torch.max(torch.tensor(age), dim=0)
            else:
                 _, self.algo_to_kill_idx = torch.max(spl_losses, dim=0)


        self.to_log['best_lr_spl_mean'] = spl_losses[new_best_lr_idx]
        change_best_algo = new_best_lr_idx != self.best_lr_idx
        self.to_log['found_better_lr'] = change_best_algo

        print("new", new_best_lr_idx, "best", self.best_lr_idx, "kill", self.algo_to_kill_idx)
        print("lrs", lrs)
        print("mean_spl_losses",spl_losses)
        print('steps_seen',steps_seen)
        print("log_likelihoods", log_likelihoods)
        print("age", age)
        if change_best_algo:
            self.best_lr_idx = new_best_lr_idx
            print("New best LR", self.strat_2_data[self.best_lr_idx]['lr'])

            # Update stored data
            for data_idx, (data, log_lik) in enumerate(zip(self.data_queue[opp_idx],
                                               list(self.strat_2_data[self.best_lr_idx]["log_likelihood"]))):
                stored_data = self.data_queue[opp_idx][data_idx]
                self.data_queue[opp_idx][data_idx] = [log_lik] + stored_data[1:]

        self.algorithms[self.coop_net_simul_opponent_idx] = self.strat_2_data[self.best_lr_idx]['algo']

        self.strat_2_is_init = True

    def calc_supervised_learn_loss(self, batch, pdparams):
        '''Calculate the actor's policy loss'''
        action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)
        targets = batch['actions']
        if self.body.env.is_venv:
            targets = math_util.venv_unpack(targets)
        preds = action_pd.probs
        if targets.dim() == 1:
            targets = self.one_hot_embedding(targets.long(), self.agent.body.action_space[self.agent.agent_idx].n)

        # print("targets", targets)
        if isinstance(self.spl_loss_fn, torch.nn.SmoothL1Loss):
            # Used with the SmoothL1Loss loss (Huber loss)  where err < 1 => MSE and err > 1 => MAE
            scaling = 2
            supervised_learning_loss = self.spl_loss_fn(preds * scaling, targets * scaling) / scaling
            supervised_learning_loss = supervised_learning_loss.mean()
        else:
            supervised_learning_loss = self.spl_loss_fn(preds, targets).mean()
        return supervised_learning_loss

    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes)
        return y[labels]

    def compare_log_likelihood_on_sequence(self, opp_idx):
        # if not self.new_improved_perf:
        #     data_queue_list = list(self.data_queue[opp_idx])
        #     bstrap_replts_data = [random.choices(population=data_queue_list,
        #                                          k=self.n_steps_in_bstrap_replts)
        #                           for i in range(self.n_bootstrapped_replications)]
        # else:
        data_array = np.array(list(self.data_queue[opp_idx]), dtype=np.object)
        bstrap_idx = np.random.random_integers(0, high=data_array.shape[0] - 1,
                                               size=(self.n_bootstrapped_replications,
                                                     self.n_steps_in_bstrap_replts))
        bstrap_replts_data = data_array[bstrap_idx]
        # Sum log_likelihood over u steps
        # if not self.new_improved_perf:
        #     log_lik_cooperate = np.array([[el[0] for el in one_replicat]
        #                                   for one_replicat in bstrap_replts_data]).sum(axis=1)
        # else:
        log_lik_cooperate = bstrap_replts_data[:, :, 0].sum(axis=1)

        # Get the log_likelihood of the observed actions under the computed opponent policy
        if self.opp_policy_from_history:
            # if "opp_historical_policy" not in locals():
            opp_historical_policy = self.approximate_policy_from_history(opp_idx)
            # else:
            #     print("reuse opp_historical_policy")
            # if not self.new_improved_perf:
            #     bstrap_replts_log_lik_defect = np.log(np.array([[opp_historical_policy(data)
            #                                                      for data in bs_data]
            #                                                     for bs_data in bstrap_replts_data]))
            #     # Sum log_likelihood over u steps
            #     log_lik_defect = [sum(e) for e in bstrap_replts_log_lik_defect]
            # else:
            bstrap_replts_log_lik_defect = np.log(np.array([[opp_historical_policy[data[3]]
                                                             for data in bs_data]
                                                            for bs_data in bstrap_replts_data]))
            # Sum log_likelihood over u steps
            log_lik_defect = bstrap_replts_log_lik_defect.sum(axis=1)

        elif self.opp_policy_from_supervised_learning:
            # Sum log_likelihood over u steps
            # if not self.new_improved_perf:
            #     log_lik_defect = np.array([[el[4] for el in one_replicat]
            #                                for one_replicat in bstrap_replts_data]).sum(axis=1)
            # else:
            log_lik_defect = bstrap_replts_data[:, :, 4].sum(axis=1)

        else:
            raise NotImplementedError()

        # Defect if in more than 0.95 of the replicates, the actual policy is more likely than the simulated coop policy
        # if not self.new_improved_perf:
        #     log_lik_check_coop = np.array(log_lik_cooperate) - np.array(log_lik_defect)
        # else:
        log_lik_check_coop = log_lik_cooperate - log_lik_defect
        assert len(log_lik_check_coop) == self.n_bootstrapped_replications
        percentile_value = np.percentile(log_lik_check_coop, self.percentile, interpolation="linear")
        percentile_0_5_value = np.percentile(log_lik_check_coop, 50, interpolation="linear")

        self.to_log.update({
            "percentile_value": percentile_value,
            "percentile_0_5_value": percentile_0_5_value,
            "log_lik_check_coop_std": log_lik_check_coop.std(),
            "log_lik_check_coop_mean": log_lik_check_coop.mean()
        })
        return percentile_value

    def defection_from_network_weights(self, state, action, welfare, next_state, done):

        if done:
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
                    net_from = algo.algorithms[algo.coop_algo_idx].net if isinstance(algo, meta_algorithm.LE) else algo.net
                    copy_weights_between_networks(copy_from_net=net_from,
                                                  copy_to_net=self.algorithms[coop_net_simul_opponent_idx].net)

                # Recompute welfare using the currently agent welfare function
                # if self.remeaning_punishing_time <= 0:
                w = self.agent.welfare_function(algo.agent, r)
                self.algorithms[coop_net_simul_opponent_idx].memory_update(
                    s, a, w, n_s, done)

                diff = compare_two_identical_net(self.algorithms[coop_net_simul_opponent_idx].net,
                                                 algo.net)

                self._update_defection_metric(epi_defection_metric=diff)

            if not self.is_fully_init:
                self.is_fully_init = True

        return True

    def _update_defection_metric(self, epi_defection_metric):
        self.defection_carac_queue.append(epi_defection_metric)
        self.defection_metric = (sum(self.defection_carac_queue) / (len(self.defection_carac_queue) + self.epsilon))

        self.to_log["defection_metric"] = round(float(self.defection_metric), 4)

    @lab_api
    def memory_update(self, state, action, welfare, next_state, done):

        # start at the last step before the end of min_cooperative_epi_after_punishement
        # TODO remove train var always to True
        train_current_active_algo = self.detect_defection(state, action, welfare, next_state, done)

        assert (self.remeaning_punishing_time > 0) == (self.active_algo_idx == self.punish_algo_idx)
        assert (self.remeaning_punishing_time <= 0) == (self.active_algo_idx == self.coop_algo_idx)
        assert (self.active_algo_idx == self.punish_algo_idx) or (self.active_algo_idx == self.coop_algo_idx)

        # Reset after a log
        if self.remeaning_punishing_time > 0:
            other_agents_rewards = agent_util.get_from_other_agents(self.agent, key="reward", default=[])
            welfare = 1 - sum(other_agents_rewards)
            self.n_punishement_steps += 1
        else:
            self.n_cooperation_steps += 1

        outputs = None
        if train_current_active_algo: # This is currently always True
            outputs = self.algorithms[self.active_algo_idx].memory_update(state, action, welfare,
                                                                          next_state, done)

        if done:

            if self.remeaning_punishing_time <= - (self.min_cooperative_epi_after_punishement - 1):
                if self.defection_metric > self.defection_carac_threshold:
                    self.detected_defection = True

            # Averaged by episode
            self.to_log["coop_frac"] = round(self.n_cooperation_steps /
                                            (self.n_punishement_steps + self.n_cooperation_steps), 2)
            self.n_cooperation_steps = 0
            self.n_punishement_steps = 0

            if self.remeaning_punishing_time > - self.min_cooperative_epi_after_punishement:
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

    def _log_actions_proba(self, action_pd, prefix):
        for act_idx in range(self.agent.body.action_dim):
            n_action_i = sum([el[act_idx] for el in action_pd])
            self.to_log[f'{prefix}{act_idx}'] = round(float(n_action_i /
                                                      (len(action_pd) + self.epsilon)), 2)

    def get_log_values(self):
        if self.agent.body.action_space_is_discrete:
            # Log action actual proba under the cooperative policy
            self._log_actions_proba(action_pd=list(self.action_pd_coop), prefix="coop_a")
            # Log action actual proba under the punishement policy
            self._log_actions_proba(action_pd=list(self.action_pd_punish), prefix="punish_a")
            # Log action actual proba under the opponent policy
            self._log_actions_proba(action_pd=list(self.action_pd_opp), prefix="opp_a")
            # Log action actual proba under the opponent simulated policy
            self._log_actions_proba(action_pd=list(self.action_pd_opp_coop), prefix="simu_opp_a")
            # Log action actual proba under the opponent approximated policy
            if hasattr(self, "action_pd_opp_approx"):
                self._log_actions_proba(action_pd=list(self.action_pd_opp_approx), prefix="approx_opp_a")
        else:
            logger.error("Logging of action proba is not implemented for no discrete action")
            raise NotImplementedError()

        to_log = super().get_log_values()
        self.to_log = {}
        return to_log
