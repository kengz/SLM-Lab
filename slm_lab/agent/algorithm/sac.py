import numpy as np
import torch

from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


class SoftActorCritic(ActorCritic):
    '''
    Soft Actor-Critic (SAC) for discrete and continuous actions.
    Discrete: Exact expectation (Christodoulou 2019)
    Continuous: Reparameterization trick (Haarnoja et al. 2018)
    '''
    @lab_api
    def init_algorithm_params(self):
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            training_iter=self.agent.env.num_envs,
            training_start_step=self.agent.memory.batch_size,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            'gamma',  # the discount factor
            'training_iter',
            'training_frequency',
            'training_start_step',
        ])
        if self.agent.is_discrete:
            assert self.action_pdtype == 'Categorical', f'Discrete SAC requires Categorical, got {self.action_pdtype}'
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)

    @lab_api
    def init_nets(self, global_nets=None):
        self.shared = False  # SAC does not share networks
        NetClass = getattr(net, self.net_spec['type'])
        # steps_per_schedule: frames processed per scheduler.step() call
        steps_per_schedule = self.training_frequency * self.agent.env.num_envs

        # Actor network
        self.net = NetClass(self.net_spec, self.agent.state_dim, net_util.get_out_dim(self.agent))
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec, steps_per_schedule)

        # Q-networks: use standard MLPNet with [state, action] concatenated input
        if self.agent.is_discrete:
            q_in_dim, q_out_dim = self.agent.state_dim, self.agent.action_dim
        else:
            q_in_dim = self.agent.state_dim + self.agent.action_dim
            q_out_dim = 1

        self.q1_net = NetClass(self.net_spec, q_in_dim, q_out_dim)
        self.target_q1_net = NetClass(self.net_spec, q_in_dim, q_out_dim)
        net_util.copy(self.q1_net, self.target_q1_net)
        self.q1_optim = net_util.get_optim(self.q1_net, self.q1_net.optim_spec)
        self.q1_lr_scheduler = net_util.get_lr_scheduler(self.q1_optim, self.q1_net.lr_scheduler_spec, steps_per_schedule)

        self.q2_net = NetClass(self.net_spec, q_in_dim, q_out_dim)
        self.target_q2_net = NetClass(self.net_spec, q_in_dim, q_out_dim)
        net_util.copy(self.q2_net, self.target_q2_net)
        self.q2_optim = net_util.get_optim(self.q2_net, self.q2_net.optim_spec)
        self.q2_lr_scheduler = net_util.get_lr_scheduler(self.q2_optim, self.q2_net.lr_scheduler_spec, steps_per_schedule)

        self.net_names = ['net', 'q1_net', 'target_q1_net', 'q2_net', 'target_q2_net']

        # Automatic entropy temperature tuning
        # Use 'auto' (default) or specify explicit target_entropy value
        target_entropy_config = self.algorithm_spec.get('target_entropy', 'auto')
        if target_entropy_config == 'auto':
            # Discrete: H_target = 0.98 * log(|A|) per Christodoulou 2019
            # Continuous: H_target = -dim(A) per Haarnoja 2018
            if self.agent.is_discrete:
                self.target_entropy = 0.98 * np.log(self.agent.action_dim)
            else:
                action_dim = np.prod(self.agent.action_space.shape)
                self.target_entropy = -action_dim
        else:
            self.target_entropy = float(target_entropy_config)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.net.device)
        self.alpha = self.log_alpha.detach().exp()
        self.alpha_optim = net_util.get_optim(self.log_alpha, self.net.optim_spec)
        self.alpha_lr_scheduler = net_util.get_lr_scheduler(self.alpha_optim, self.net.lr_scheduler_spec, steps_per_schedule)
        self.agent.mt.register_algo_var('alpha', self)

        net_util.set_global_nets(self, global_nets)
        self.end_init_nets()

    @lab_api
    def act(self, state):
        if self.agent.env.get('frame') < self.training_start_step:
            action = policy_util.random(state, self)
        else:
            action = self.action_policy(state, self)
            if not self.agent.is_discrete:
                # Squash to [-1, 1] with tanh; RescaleAction wrapper scales to env bounds
                action = torch.tanh(action)
        return self.to_action(action)

    def calc_log_prob_action(self, action_pd, reparam=False):
        actions = action_pd.rsample() if reparam else action_pd.sample()
        if self.agent.is_discrete:
            log_probs = action_pd.log_prob(actions)
        else:
            raw_log_probs = action_pd.log_prob(actions).sum(-1)  # Sum across action dimensions
            # Tanh squash, change of variables: log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
            actions = torch.tanh(actions)
            squash_correction = torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
            log_probs = raw_log_probs - squash_correction
            # Actions are in [-1, 1]; RescaleAction wrapper handles scaling to env bounds
        return log_probs, actions

    def calc_q_discrete(self, states, actions, q_net):
        q_all = q_net(states)
        q_preds = q_all.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        return q_preds, q_all

    def calc_q_cont(self, states, actions, q_net):
        # Concatenate states and actions for Q-network input
        s_a = torch.cat((states, actions), dim=-1)
        q_preds = q_net(s_a).view(-1)
        return q_preds, None

    def calc_q(self, states, actions, q_net):
        '''Returns (q_preds, q_all) where q_all is only populated for discrete'''
        fn = self.calc_q_discrete if self.agent.is_discrete else self.calc_q_cont
        return fn(states, actions, q_net)

    def calc_v_next(self, next_states, action_pd):
        '''
        Value function V(s').
        Discrete: V(s') = Σ_a π(a|s')[min(Q1,Q2) - α·log(π)]
        Continuous: V(s') = min(Q1,Q2) - α·log(π) where a ~ π
        '''
        if self.agent.is_discrete:
            next_probs = action_pd.probs
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1_all = self.target_q1_net(next_states)
            next_q2_all = self.target_q2_net(next_states)
            min_q = torch.min(next_q1_all, next_q2_all)
            return (next_probs * (min_q - self.alpha * next_log_probs)).sum(dim=-1)
        else:
            next_log_probs, next_actions = self.calc_log_prob_action(action_pd)
            next_q1, _ = self.calc_q_cont(next_states, next_actions, self.target_q1_net)
            next_q2, _ = self.calc_q_cont(next_states, next_actions, self.target_q2_net)
            min_q = torch.min(next_q1, next_q2)
            return min_q - self.alpha * next_log_probs

    def calc_q_targets(self, batch):
        with torch.no_grad():
            pdparams = self.calc_pdparam(batch['next_states'])
            action_pd = policy_util.init_action_pd(self.agent.ActionPD, pdparams)
            v_next = self.calc_v_next(batch['next_states'], action_pd)
            q_targets = batch['rewards'] + self.gamma * (1 - batch['terminateds']) * v_next
        return q_targets

    def calc_policy_loss_discrete(self, states, action_pd, q1_all, q2_all):
        '''J_π = E[Σ_a π(a|s)[α·log(π) - min(Q1,Q2)]]'''
        action_probs = action_pd.probs
        action_log_probs = torch.log(action_probs + 1e-8)
        with torch.no_grad():
            min_q_all = torch.min(q1_all, q2_all)
        return (action_probs * (self.alpha.detach() * action_log_probs - min_q_all)).sum(dim=1).mean()

    def calc_policy_loss_cont(self, states, action_pd, q1_all=None, q2_all=None):
        '''J_π = E[α·log(π) - Q(s,a)] where a ~ π'''
        log_probs, reparam_actions = self.calc_log_prob_action(action_pd, reparam=True)
        q1_preds, _ = self.calc_q_cont(states, reparam_actions, self.q1_net)
        q2_preds, _ = self.calc_q_cont(states, reparam_actions, self.q2_net)
        q_preds = torch.min(q1_preds, q2_preds)
        policy_loss = (self.alpha * log_probs - q_preds).mean()
        # Cache log_probs for alpha loss to avoid resampling
        self._cached_log_probs = log_probs
        return policy_loss

    def calc_policy_loss(self, states, action_pd, q1_all=None, q2_all=None):
        '''Dispatcher for policy loss calculation'''
        fn = self.calc_policy_loss_discrete if self.agent.is_discrete else self.calc_policy_loss_cont
        return fn(states, action_pd, q1_all, q2_all)

    def calc_alpha_loss_discrete(self, action_pd):
        '''J_α = log_α * (H_target - H)'''
        action_probs = action_pd.probs
        action_log_probs = torch.log(action_probs + 1e-8)
        with torch.no_grad():
            entropy_current = -(action_probs * action_log_probs).sum(dim=-1).mean()
        return self.log_alpha * (self.target_entropy - entropy_current)

    def calc_alpha_loss_cont(self, action_pd):
        '''J_α = -α * (log π + H_target)'''
        # Reuse cached log_probs from policy loss to avoid resampling
        log_probs = self._cached_log_probs
        # Use log_alpha.exp() (= alpha) as the coefficient, matching CleanRL
        return -(self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)).mean()

    def calc_alpha_loss(self, action_pd):
        '''Dispatcher for alpha loss calculation'''
        fn = self.calc_alpha_loss_discrete if self.agent.is_discrete else self.calc_alpha_loss_cont
        return fn(action_pd)

    def try_update_per(self, q_preds, q_targets):
        if 'Prioritized' not in util.get_class_name(self.agent.memory):
            return
        with torch.no_grad():
            errors = (q_preds - q_targets).abs().cpu().numpy()
        self.agent.memory.update_priorities(errors)

    def train_alpha(self, alpha_loss):
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # NOTE: alpha_lr_scheduler.step() is called once per training iteration in train(),
        # not here per gradient step
        # No clamping - let automatic entropy tuning work naturally (CleanRL approach)
        self.alpha = self.log_alpha.detach().exp()

    def train(self):
        if self.to_train == 1:
            for _ in range(self.training_iter):
                batch = self.sample()
                self.agent.env.set_batch_size(len(batch))

                states = batch['states']
                actions = batch['actions']
                q_targets = self.calc_q_targets(batch)

                # Q-value loss for both Q nets
                q1_preds, q1_all = self.calc_q(states, actions, self.q1_net)
                q2_preds, q2_all = self.calc_q(states, actions, self.q2_net)

                q1_loss = self.net.loss_fn(q1_preds, q_targets)
                self.q1_net.train_step(q1_loss, self.q1_optim, self.q1_lr_scheduler, global_net=self.global_q1_net)

                q2_loss = self.net.loss_fn(q2_preds, q_targets)
                self.q2_net.train_step(q2_loss, self.q2_optim, self.q2_lr_scheduler, global_net=self.global_q2_net)

                # policy loss
                action_pd = policy_util.init_action_pd(self.agent.ActionPD, self.calc_pdparam(states))
                policy_loss = self.calc_policy_loss(states, action_pd, q1_all, q2_all)
                self.net.train_step(policy_loss, self.optim, self.lr_scheduler, global_net=self.global_net)

                # alpha loss
                alpha_loss = self.calc_alpha_loss(action_pd)
                self.train_alpha(alpha_loss)

                loss = q1_loss + q2_loss + policy_loss + alpha_loss
                self.agent.env.tick_opt_step()
                # update target networks
                self.update_nets()
                # update PER priorities if available
                self.try_update_per(torch.min(q1_preds, q2_preds), q_targets)

            # Step LR schedulers once per training iteration (after all gradient updates)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.q1_lr_scheduler is not None:
                self.q1_lr_scheduler.step()
            if self.q2_lr_scheduler is not None:
                self.q2_lr_scheduler.step()
            if self.alpha_lr_scheduler is not None:
                self.alpha_lr_scheduler.step()
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {self.agent.env.get("epi")}, frame: {self.agent.env.get("frame")}, t: {self.agent.env.get("t")}, total_reward so far: {self.agent.env.total_reward}, loss: {loss.item():g}')
            return loss.item()
        else:
            return np.nan

    def update_nets(self):
        net_util.update_target_net(self.q1_net, self.target_q1_net, self.agent.env.get('frame'), self.agent.env.num_envs)
        net_util.update_target_net(self.q2_net, self.target_q2_net, self.agent.env.get('frame'), self.agent.env.num_envs)

    @lab_api
    def update(self):
        return self.agent.explore_var
