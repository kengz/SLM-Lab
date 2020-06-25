from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib import math_util, util
from slm_lab.agent.net import net_util
from slm_lab.agent.agent import agent_util
from slm_lab.lib.decorator import lab_api
import numpy as np
from collections import deque
import torch
from collections import Iterable

from slm_lab.lib import logger
logger = logger.get_logger(__name__)


class Planning_Agent(Algorithm):
    # TODO make it work not only on matrix games


    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            max_reward_strength=3,
            cost_param=0.0002,
            with_redistribution=False,
            gamma=0.95,
            value_fn_variant='exact',
            n_players=2,
            action_flip_prob=0,
            n_planning_eps=-1,
            training_frequency=1,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'max_reward_strength',
            'cost_param',
            'with_redistribution',
            'gamma',  # the discount factor
            'value_fn_variant',
            'n_players,',
            'action_flip_prob'
            'n_planning_eps',
            'training_frequency'
        ])
        self.to_train = 0

        self.cummulated_planning_r = [0] * self.n_players

        self.underlying_agents = [agent for agent in self.agent.observed_agents if agent.playing_agent]

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPNet'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''


        in_dim = list(self.body.observation_dim)
        # N x state + 1 action per player
        in_dim[-1] = in_dim[-1] * self.n_players + self.n_players
        out_dim = self.n_players
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim, self.internal_clock)
        self.net_names = ['net']

        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        self.post_init_nets()


        # if self.max_reward_strength is not None:
        #     self.net = torch.sigmoid(self.net)

    @lab_api
    def act(self, state):
        return None, None

    @lab_api
    def act_after_env_step(self):
        self.agent.observe_other_agents()
        self.other_ag_states = agent_util.get_from_other_agents(self.agent, key="state", default=[])
        other_ag_actions = agent_util.get_from_other_agents(self.agent, key="action", default=[])
        self.other_ag_reward = agent_util.get_from_other_agents(self.agent, key="reward", default=[])
        self.perturbed_actions = [(1 - a if np.random.binomial(1, self.action_flip_prob) else a) for a in
                                  other_ag_actions]

        # print("self.other_ag_states", self.other_ag_states)
        # print("self.other_ag_reward", self.other_ag_reward)
        # print("self.perturbed_actions", self.perturbed_actions)

        if self.agent.body.env.clock.get("epi") < self.n_planning_eps or self.n_planning_eps < 0:
            planned_r_nudge = self.choose_action(self.other_ag_states, self.perturbed_actions)
            if self.with_redistribution:
                sum_planning_r = sum(planned_r_nudge)
                mean_planning_r = sum_planning_r / self.n_players
                planned_r_nudge = [r - mean_planning_r for r in planned_r_nudge]
            modified_rewards = [sum(r) for r in zip(self.other_ag_reward, planned_r_nudge)]
            self.cummulated_planning_r = [sum(r) for r in zip(self.cummulated_planning_r, planned_r_nudge)]

            # # Training planning agent
            # self.learn(state, self.perturbed_actions)
            # logger.info('Actions:' + str(other_ag_actions))
            # logger.info('Rewards: ' + str(modified_rewards))
            for agent_idx in range(self.n_players):
                if self.with_redistribution:
                    self.to_log[f'sum_planning_r_{agent_idx}'] = sum_planning_r[agent_idx]
                    self.to_log[f'mean_planning_r_{agent_idx}'] = mean_planning_r[agent_idx]
                self.to_log[f'cum_planning_rs_{agent_idx}'] = self.cummulated_planning_r[agent_idx]
                self.to_log[f'planned_r_nudge_{agent_idx}'] = planned_r_nudge[agent_idx]

            return {"reward":modified_rewards}
        else:
            return {}


    def choose_action(self, s, a_players):

        # if not a_players[0] and not a_players[1]:
        #     return (3,3)
        # if not a_players[0] and  a_players[1]:
        #     return (3,-3)
        # if  a_players[0] and not a_players[1]:
        #     return (-3,3)
        # if  a_players[0] and  a_players[1]:
        #     return (-3,-3)

        # logger.info('Player actions: ' + str(a_players))
        # s = s[np.newaxis, :]
        s = np.concatenate(s,axis=0)[np.newaxis, :]
        a_players = np.asarray(a_players)[np.newaxis, :]
        # a_plan = self.sess.run(self.net, {self.s: s, self.a_players: a_players[np.newaxis, :]})[0, :]

        # net_input = torch.cat([torch.tensor(s).float(), torch.tensor(a_players).float()], dim=1).to(self.net.device)
        # a_plan = self.net(net_input)[0, :]
        a_plan = self.net(self.format_input_to_net(s, a_players))
        if self.max_reward_strength is not None:
            a_plan = torch.sigmoid(a_plan)
            a_plan = 2 * self.max_reward_strength * (a_plan - 0.5)
        # logger.info('Planning action: ' + str(a_plan))
        # logger.info(f"{self.calc_conditional_planning_actions(s)}")

        return a_plan[0] # with batch size 1

    def format_input_to_net(self, s, a_players):
        return torch.cat([torch.as_tensor(s).float(), torch.as_tensor(a_players).float()], dim=1).to(self.net.device)

    def calc_conditional_planning_actions(self,s):
        # Planning actions in each of the 4 cases: DD, CD, DC, CC
        batch_size = len(s)
        a_plan_DD = self.net(self.format_input_to_net(s, np.array([[1, 1]]*batch_size)))
        a_plan_CD = self.net(self.format_input_to_net(s, np.array([[0, 1]]*batch_size)))
        a_plan_DC = self.net(self.format_input_to_net(s, np.array([[1, 0]]*batch_size)))
        a_plan_CC = self.net(self.format_input_to_net(s, np.array([[0, 0]]*batch_size)))

        a_plan_DD_mean = a_plan_DD.mean(dim=0)
        a_plan_CD_mean = a_plan_CD.mean(dim=0)
        a_plan_DC_mean = a_plan_DC.mean(dim=0)
        a_plan_CC_mean = a_plan_CC.mean(dim=0)
        for agent_idx in range(len(a_plan_DD_mean)):
            self.to_log[f'r_nudge_dd_plan_ag_{agent_idx}'] = a_plan_DD_mean[agent_idx]
            self.to_log[f'r_nudge_cd_plan_ag_{agent_idx}'] = a_plan_CD_mean[agent_idx]
            self.to_log[f'r_nudge_dc_plan_ag_{agent_idx}'] = a_plan_DC_mean[agent_idx]
            self.to_log[f'r_nudge_cc_plan_ag_{agent_idx}'] = a_plan_CC_mean[agent_idx]

        l_temp = [a_plan_DD,a_plan_CD,a_plan_DC,a_plan_CC]
        if self.max_reward_strength is not None:
            l_temp = [ torch.sigmoid(a_plan_X) for a_plan_X in l_temp]
            # l = [2 * self.max_reward_strength * (a_plan_X[:,0]-0.5) for a_plan_X in l_temp]
            l = [2 * self.max_reward_strength * (a_plan_X[:1]-0.5) for a_plan_X in l_temp]
        else:
            l = [a_plan_X.mean() for a_plan_X in l_temp]
        if self.with_redistribution:
            if self.max_reward_strength is not None:
                # l2 = [2 * self.max_reward_strength * (a_plan_X[:,1]-0.5) for a_plan_X in l_temp]
                l2 = [2 * self.max_reward_strength * (a_plan_X[:,0]-0.5) for a_plan_X in l_temp]
            else:
                # l2 = [a_plan_X[:,1] for a_plan_X in l_temp]
                l2 = [a_plan_X[:, 0] for a_plan_X in l_temp]
            # l = [0.5 * (elt[0]-elt[1]) for elt in zip(l,l2)]
            l = [0.5 * (elt[1] - elt[2]) for elt in zip(l, l2)]
        plan = np.transpose(np.reshape(np.asarray(l),[2,2]))
        # print("l",l)
        # print("plan",plan.shape)
        return plan

    def memory_update(self, state, action, welfare, next_state, done):
        # TODO Support vectorized env += n vectorized, done ?
        self.internal_clock.tick(unit='t')
        return self.memory.update(self.other_ag_states, self.perturbed_actions, self.other_ag_reward, None, done)

    @lab_api
    def update(self):
        return None

    @lab_api
    def sample(self, reset=True):
        '''Samples a batch from memory'''
        batch = self.memory.sample(reset=reset)
        batch = util.to_torch_batch(batch, self.net.device, self.memory.is_episodic)
        return batch

    @lab_api
    def train(self):
        if util.in_eval_lab_modes():
            return np.nan
        if self.to_train == 1:
            batch = self.sample()

            # s = s[np.newaxis, :]
            # r_players = np.asarray(self.env.calculate_payoffs(a_players))
            # a_players = np.asarray(a_players)
            # feed_dict = {self.s: s,
            #              self.a_players: a_players[np.newaxis, :],
            #              self.r_players: r_players[np.newaxis, :]}
            s = batch["states"]
            s_batch = s.reshape((len(s), -1))
            r_players = batch["rewards"]
            a_players = batch["actions"]
            if self.value_fn_variant == 'estimated':
                grad_log_pi_list = []
                for underlying_idx, underlying_agent in enumerate(self.underlying_agents):
                    # TODO calc_grad_log_pi
                    # self.log_prob = tf.log(self.actions_prob[0, self.a])
                    # self.theta = tf.Variable(tf.random_normal([1], mean=-2, stddev=0.5))
                    # self.g_log_pi = tf.gradients(self.log_prob, self.theta)
                    # grad_log_pi_list.append(underlying_agent.calc_grad_log_pi(s, a_players[idx]))
                    _, a_prob_distrib = underlying_agent.act(s[:,underlying_idx,:])
                    a_proba = a_prob_distrib.probs[:, ...]
                    log_a_proba = torch.log(a_proba)
                    # # TODO change this (not working currently)
                    print("underlying_agent.algorithm.net.parameters()",
                          type(torch.tensor(list(underlying_agent.algorithm.net.parameters())[0])))
                    print("log_a_proba",log_a_proba.shape)
                    grad_log_pi = torch.autograd.grad(log_a_proba,underlying_agent.algorithm.net.parameters())
                    grad_log_pi_list.append(grad_log_pi)


                grad_log_pi = np.reshape(np.asarray(grad_log_pi_list), [len(s), -1])
                a_proba_players = None
                # feed_dict[self.g_log_pi] = g_log_pi_arr
            elif self.value_fn_variant == 'exact':
                a_proba_players_list = []
                # print("s[:, idx]", s.shape)
                # print("s[:, idx]", s[:,0,:].shape)

                for underlying_ag_idx, underlying_agent in enumerate(self.underlying_agents):
                    _, a_prob_distrib = underlying_agent.act(s[:,underlying_ag_idx,:])
                    a_proba = a_prob_distrib.probs[:, ...]
                    # p_players_list.append(a_proba[0, -1])
                    a_proba_players_list.append(a_proba)
                # feed_dict[self.p_players] = p_players_arr
                # feed_dict[self.a_plan] = self.calc_conditional_planning_actions(s)
                grad_log_pi = None
                # a_proba_players_list = np.stack(a_proba_players_list, axis=-1)
                # a_proba_players = np.reshape(np.asarray(a_proba_players_list), [len(s), -1])
                a_proba_players = np.stack(a_proba_players_list, axis=1)
                a_plan = self.calc_conditional_planning_actions(s_batch)

            # self.sess.run([self.train_op], feed_dict)

            # action = self.net()
            loss = self.calc_loss(s_batch, r_players, a_players, a_proba_players, grad_log_pi) #* 1e4
            print("loss", loss)
            # action, loss, g_Vp, g_V = self.sess.run([self.net, self.loss,
            #                                          self.g_Vp, self.g_V], feed_dict)
            # logger.info('Learning step')
            # logger.info('Planning_action: ' + str(action))
            # if self.value_fn_variant == 'estimated':
            #     # vp,v = self.sess.run([self.vp,self.v],feed_dict)
            #     vp, v = self.compute_V_player(), self.compute_V_tot()
            #     logger.info('Vp: ' + str(vp))
            #     logger.info('V: ' + str(v))
            # logger.info('Gradient of V_p: ' + str(g_Vp))
            # logger.info('Gradient of V: ' + str(g_V))
            # logger.info('Loss: ' + str(loss))




            lr_source = self.lr_scheduler
            self.net.train_step(loss.mean(), self.optim, lr_source, clock=self.internal_clock,
                global_net=self.global_net)
            if hasattr(self, "lr_overwritter"):
                del self.lr_overwritter
            else:
                self.to_log['lr'] = np.mean(self.lr_scheduler.get_lr())

            # reset
            self.to_train = 0
            clock = self.body.env.clock
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.env.total_reward}, loss: {loss:g}')
            self.to_log["loss_tot"] = loss.item()
            return loss.item()
        else:
            return np.nan

    def calc_grad_log_pi(self, algo, s):
        # TODO Will act work on batch of s ?
        _, a_prob_distrib = algo.act(s)
        action_probs = a_prob_distrib.probs[0, ...]
        log_action_probs = torch.log(action_probs)

        log_action_probs.backward()
        grad_log_action_probs = s.grad
        return grad_log_action_probs

    def compute_V_player(self, s, a_players):

        # Vp is trivial to calculate in this special case
        input = torch.cat([s, a_players], dim=1)
        if self.max_reward_strength is not None:
            output = torch.sigmoid(self.net(input))
            vp = 2 * self.max_reward_strength * ( output - 0.5)
        else:
            vp = self.net(input)
        return vp

    def compute_V_tot(self, r_players, a_players):
        if self.value_fn_variant == 'proxy':
            # TODO replace hardcoded, what is it (look at paper)?
            v = 2 * a_players - 1
        elif self.value_fn_variant == 'estimated':
            # TODO replace hardcoded, what is it (look at paper)?
            v = r_players.sum() - 1.9
        return v

    def calc_loss(self, s, r_players, a_players_orig, a_proba_players_orig, grad_log_pi):

        a_players_orig = torch.autograd.Variable(a_players_orig, requires_grad=True)
        a_proba_players_orig = torch.as_tensor(a_proba_players_orig)


        print("a_proba_players",a_proba_players_orig.shape)
        print("a_proba_players",a_proba_players_orig.mean(dim=0))

        cost_list = []
        for underlying_ag_idx, underlying_agent in enumerate(self.underlying_agents):
            a_proba_players = a_proba_players_orig  # to account for the CC = 11 for Thobias and = 00 here
            a_players = a_players_orig

            vp = self.compute_V_player(s, a_players_orig)

            # to account for the CC = 11 for Thobias and = 00 here
            # vp = torch.flip(vp, [0, 1])

            # policy gradient theorem
            # idx = underlying_agent.agent_idx

            if self.value_fn_variant == 'estimated':
                v = self.compute_V_tot(r_players, a_players)

                # g_Vp = grad_log_pi[0, idx] * vp[0, idx]
                # g_V = grad_log_pi[0, idx] * (v[0, idx] if self.value_fn_variant == 'proxy' else v)
                grad_Vp = grad_log_pi[:, underlying_ag_idx] * vp[:, underlying_ag_idx]
                # TODO next line has a bug?
                grad_V = grad_log_pi[:, underlying_ag_idx] * (v[:, underlying_ag_idx] if self.value_fn_variant == 'proxy' else v)
            elif self.value_fn_variant == 'exact':

                # to account for the CC = 11 for Thobias and = 00 here
                # a_proba_players = 1 - a_proba_players_orig  # to account for the CC = 11 for Thobias and = 00 here
                # a_players = 1 - a_players_orig


                # TODO make this work
                # grad_a_player = a_proba_players[0, idx] * (1 - a_proba_players[0, idx])
                # player_opp_a_prob = a_proba_players[0, 1 - idx]
                # self.g_Vp = self.g_p * tf.gradients(ys = vp[0,idx],xs = self.a_players)[0][0,idx]
                # vp[0, idx].backward()
                # g_Vp = grad_a_player * a_players.grad[0][0,idx]
                grad_a_player = a_proba_players[:, underlying_ag_idx,:] * (1 - a_proba_players[:, underlying_ag_idx,:])
                player_opp_a_prob = a_proba_players[:, 1 - underlying_ag_idx,:]

                # vp[:, idx].mean().backward()
                # print("a_players.grad",a_players.grad)
                # grad_Vp = grad_a_player * a_players.grad[0][:,idx]
                # self.g_Vp = self.g_p * tf.gradients(ys=self.vp[0, idx], xs=self.a_players)[0][0, idx]

                grad = torch.autograd.grad(vp[:, underlying_ag_idx].mean(), a_players,
                                           retain_graph=True, create_graph=True)
                grad_Vp = grad_a_player * grad[0]

                R = 1
                T = 4
                S = 0
                P = 1
                # grad_V = grad_a_player * (player_opp_a_prob * (2 * R - T - S)
                #                        + (1 - player_opp_a_prob) * (T + S - 2 * P))
                # grad_a_player = torch.flip(grad_a_player, [0, 1])
                grad_V = grad_a_player * ((1-player_opp_a_prob) * (2 * R - T - S)
                                       + player_opp_a_prob * (T + S - 2 * P))
                grad_V = torch.as_tensor(grad_V)

            # underlying_agent.learning_rate * tf.tensordot(self.g_Vp,self.g_V,1))
            lr = underlying_agent.algorithm.lr_scheduler.get_lr()
            if isinstance(lr, Iterable):
                lr = lr[0]
            # print("lr * grad_Vp * grad_V", lr, grad_Vp, grad_V)
            # print("lr * grad_Vp * grad_V", lr, grad_Vp.shape, grad_V.shape)
            cost_list.append(- lr * grad_Vp * grad_V)

        # print("torch.stack(cost_list).sum()", torch.stack(cost_list).sum())
        if self.with_redistribution:
            extra_loss = self.cost_param * (vp-vp.mean()).norm()
        else:
            extra_loss = self.cost_param * vp.norm()
        loss = torch.stack(cost_list).sum() + extra_loss
        return loss

    # def learn(self, s, a_players):
    #     s = s[np.newaxis,:]
    #     r_players = np.asarray(self.env.calculate_payoffs(a_players))
    #     a_players = np.asarray(a_players)
    #     feed_dict = {self.s: s, self.a_players: a_players[np.newaxis,:],
    #                 self.r_players: r_players[np.newaxis,:]}
    #     if self.value_fn_variant == 'estimated':
    #         g_log_pi_list = []
    #         for underlying_agent in self.underlying_agents:
    #             idx = underlying_agent.agent_idx
    #             g_log_pi_list.append(underlying_agent.calc_g_log_pi(s,a_players[idx]))
    #         g_log_pi_arr = np.reshape(np.asarray(g_log_pi_list),[1,-1])
    #         feed_dict[self.g_log_pi] = g_log_pi_arr
    #     if self.value_fn_variant == 'exact':
    #         p_players_list = []
    #         for underlying_agent in self.underlying_agents:
    #             idx = underlying_agent.agent_idx
    #             p_players_list.append(underlying_agent.calc_action_probs(s)[0,-1])
    #         p_players_arr = np.reshape(np.asarray(p_players_list),[1,-1])
    #         feed_dict[self.p_players] = p_players_arr
    #         feed_dict[self.a_plan] = self.calc_conditional_planning_actions(s)
    #
    #     self.sess.run([self.train_op], feed_dict)
    #
    #     action,loss,g_Vp,g_V = self.sess.run([self.net, self.loss,
    #                                           self.g_Vp, self.g_V], feed_dict)
    #     logger.info('Learning step')
    #     logger.info('Planning_action: ' + str(action))
    #     if self.value_fn_variant == 'estimated':
    #         # vp,v = self.sess.run([self.vp,self.v],feed_dict)
    #         vp,v = self.compute_V_player(), self.compute_V_tot()
    #         logger.info('Vp: ' + str(vp))
    #         logger.info('V: ' + str(v))
    #     logger.info('Gradient of V_p: ' + str(g_Vp))
    #     logger.info('Gradient of V: ' + str(g_V))
    #     logger.info('Loss: ' + str(loss))

    # def get_log(self):
    #     return self.log







