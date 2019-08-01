from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch

logger = logger.get_logger(__name__)


class SoftActorCritic(ActorCritic):
    '''
    Implementation of Soft Actor-Critic (SAC)
    Original paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    https://arxiv.org/abs/1801.01290

    e.g. algorithm_spec
    "algorithm": {
        "name": "SoftActorCritic",
        "action_pdtype": "default",
        "action_policy": "default",
        "gamma": 0.99,
        "training_frequency": 1,
    }
    '''
    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            training_iter=self.body.env.num_envs,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            'gamma',  # the discount factor
            'training_iter',
            'training_frequency',
        ])
        self.to_train = 0
        self.training_start_step = self.training_frequency
        self.action_policy = getattr(policy_util, self.action_policy)

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Networks: net(actor/policy), critic (value), target_critic, q1_net, q1_net
        All networks are separate, and have the same hidden layer architectures and optim specs, so tuning is minimal
        '''
        self.shared = False  # SAC does not share networks
        in_dim = self.body.state_dim
        out_dim = net_util.get_out_dim(self.body)
        NetClass = getattr(net, self.net_spec['type'])
        # main actor network
        # NOTE continuous action bound
        policy_net_spec = self.net_spec.copy()
        policy_net_spec['out_layer_activation'] = 'tanh'
        self.net = NetClass(policy_net_spec, in_dim, out_dim)
        self.net_names = ['net']
        # critic network and its target network
        val_out_dim = 1
        self.critic_net = NetClass(self.net_spec, in_dim, val_out_dim)
        self.target_critic_net = NetClass(self.net_spec, in_dim, val_out_dim)
        self.net_names += ['critic_net', 'target_critic_net']
        # two Q-networks to mitigate positive bias in q_loss and speed up training
        q_in_dim = in_dim + self.body.action_dim  # NOTE concat s, a for now
        self.q1_net = NetClass(self.net_spec, q_in_dim, val_out_dim)
        self.q2_net = NetClass(self.net_spec, q_in_dim, val_out_dim)
        self.net_names += ['q1_net', 'q2_net']

        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        self.critic_optim = net_util.get_optim(self.critic_net, self.critic_net.optim_spec)
        self.critic_lr_scheduler = net_util.get_lr_scheduler(self.critic_optim, self.critic_net.lr_scheduler_spec)
        self.q1_optim = net_util.get_optim(self.q1_net, self.q1_net.optim_spec)
        self.q1_lr_scheduler = net_util.get_lr_scheduler(self.q1_optim, self.q1_net.lr_scheduler_spec)
        self.q2_optim = net_util.get_optim(self.q2_net, self.q2_net.optim_spec)
        self.q2_lr_scheduler = net_util.get_lr_scheduler(self.q2_optim, self.q2_net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        self.post_init_nets()

    @lab_api
    def act(self, state):
        # NOTE continuous action bound
        action = super().act(state)
        return np.tanh(action)

    def calc_q(self, state, action, net=None):
        '''Forward-pass to calculate the predicted state-action-value from q1_net.'''
        x = torch.cat((state, action), dim=-1)
        net = self.q1_net if net is None else net
        q_pred = net(x).view(-1)
        return q_pred

    def calc_v_targets(self, batch, action_pd, q1_preds, q2_preds):
        '''V_tar = Q(s, a) - log pi(a|s), Q(s, a) = min(Q1(s, a), Q2(s, a))'''
        with torch.no_grad():
            q_preds = torch.min(q1_preds, q2_preds)
            log_probs = action_pd.log_prob(batch['actions'])
            v_targets = q_preds - log_probs
        return v_targets

    def calc_q_targets(self, batch):
        '''Q_tar = r + gamma * V_pred(s'; target_critic)'''
        with torch.no_grad():
            target_next_v_preds = self.calc_v(batch['next_states'], net=self.target_critic_net)
            q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * target_next_v_preds
        return q_targets

    def calc_reg_loss(self, preds, targets):
        '''Calculate the regression loss for V and Q values, using the same loss function from net_spec'''
        assert preds.shape == targets.shape, f'{preds.shape} != {targets.shape}'
        reg_loss = self.net.loss_fn(preds, targets)
        return reg_loss

    def calc_policy_loss(self, batch, action_pd):
        '''policy_loss = log pi(f(a)|s) - Q1(s, f(a)), where f(a) = reparametrized action'''
        reparam_actions = action_pd.rsample()
        log_probs = action_pd.log_prob(reparam_actions)
        q1_preds = self.calc_q(batch['states'], reparam_actions, self.q1_net)
        policy_loss = (log_probs - q1_preds).mean()
        return policy_loss

    def try_update_per(self, q_preds, q_targets):
        if 'Prioritized' in util.get_class_name(self.body.memory):  # PER
            with torch.no_grad():
                errors = (q_preds - q_targets).abs().cpu().numpy()
            self.body.memory.update_priorities(errors)

    def train(self):
        '''Train actor critic by computing the loss in batch efficiently'''
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            for _ in range(self.training_iter):
                batch = self.sample()
                clock.set_batch_size(len(batch))

                # forward passes for losses
                states = batch['states']
                actions = batch['actions']
                v_preds = self.calc_v(states, net=self.critic_net)
                q1_preds = self.calc_q(states, actions, self.q1_net)
                q2_preds = self.calc_q(states, actions, self.q2_net)
                pdparams = self.calc_pdparam(states)
                action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)

                # V-value loss
                v_targets = self.calc_v_targets(batch, action_pd, q1_preds, q2_preds)
                val_loss = self.calc_reg_loss(v_preds, v_targets)
                self.critic_net.train_step(val_loss, self.critic_optim, self.critic_lr_scheduler, clock=clock, global_net=self.global_critic_net)

                # Q-value loss for both Q nets
                q_targets = self.calc_q_targets(batch)
                q1_loss = self.calc_reg_loss(q1_preds, q_targets)
                self.q1_net.train_step(q1_loss, self.q1_optim, self.q1_lr_scheduler, clock=clock, global_net=self.global_q1_net)
                q2_loss = self.calc_reg_loss(q2_preds, q_targets)
                self.q2_net.train_step(q2_loss, self.q2_optim, self.q2_lr_scheduler, clock=clock, global_net=self.global_q2_net)

                # policy loss
                policy_loss = self.calc_policy_loss(batch, action_pd)
                self.net.train_step(policy_loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)

                loss = policy_loss + val_loss + q1_loss + q2_loss

                # update target_critic_net
                self.update_nets()
                # update PER priorities if availalbe
                self.try_update_per(q1_preds, q_targets)

            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.env.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    def update_nets(self):
        '''Update target critic net'''
        if util.frame_mod(self.body.env.clock.frame, self.critic_net.update_frequency, self.body.env.num_envs):
            if self.critic_net.update_type == 'replace':
                net_util.copy(self.critic_net, self.target_critic_net)
            elif self.critic_net.update_type == 'polyak':
                net_util.polyak_update(self.critic_net, self.target_critic_net, self.critic_net.polyak_coef)
            else:
                raise ValueError('Unknown critic_net.update_type. Should be "replace" or "polyak". Exiting.')

    @lab_api
    def update(self):
        '''Updates self.target_critic_net and the explore variables'''
        return self.body.explore_var
