from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch
import torch.nn.functional as F

logger = logger.get_logger(__name__)


class SoftActorCritic(ActorCritic):
    '''
    Implementation of Soft Actor-Critic (SAC)
    Original paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    https://arxiv.org/abs/1801.01290
    Improvement of SAC paper: "Soft Actor-Critic Algorithms and Applications"
    https://arxiv.org/abs/1812.05905

    e.g. algorithm_spec
    "algorithm": {
        "name": "SoftActorCritic",
        "action_pdtype": "default",
        "action_policy": "default",
        "gamma": 0.99,
        "training_frequency": 1,
        "target_entropy_epsilon": 0.1,
    }
    '''
    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
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
            assert self.action_pdtype == 'GumbelSoftmax'
        else:
            # Cache action scaling tensors for continuous envs
            space = self.agent.action_space
            self._action_low = torch.from_numpy(space.low).float()
            self._action_high = torch.from_numpy(space.high).float()
            self._action_scale = (self._action_high - self._action_low) / 2
            self._action_bias = (self._action_low + self._action_high) / 2
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Networks: net(actor/policy), q1_net, target_q1_net, q2_net, target_q2_net
        All networks are separate, and have the same hidden layer architectures and optim specs, so tuning is minimal
        '''
        self.shared = False  # SAC does not share networks
        NetClass = getattr(net, self.net_spec['type'])
        # main actor network
        self.net = NetClass(self.net_spec, self.agent.state_dim, net_util.get_out_dim(self.agent))
        self.net_names = ['net']
        # two critic Q-networks to mitigate positive bias in q_loss and speed up training, uses q_net.py with prefix Q
        QNetClass = getattr(net, 'Q' + self.net_spec['type'])
        q_in_dim = [self.agent.state_dim, self.agent.action_dim]
        self.q1_net = QNetClass(self.net_spec, q_in_dim, 1)
        self.target_q1_net = QNetClass(self.net_spec, q_in_dim, 1)
        self.q2_net = QNetClass(self.net_spec, q_in_dim, 1)
        self.target_q2_net = QNetClass(self.net_spec, q_in_dim, 1)
        self.net_names += ['q1_net', 'target_q1_net', 'q2_net', 'target_q2_net']
        net_util.copy(self.q1_net, self.target_q1_net)
        net_util.copy(self.q2_net, self.target_q2_net)
        # temperature variable to be learned, and its target entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.net.device)
        self.alpha = self.log_alpha.detach().exp()
        # Register alpha for logging
        self.agent.mt.register_algo_var('alpha', self)
        
        # Target entropy using epsilon-greedy policy bound
        # Epsilon = exploration probability for baseline policy (default 0.1 = 10% exploration)
        # Based on: https://discuss.ray.io/t/target-entropy-in-discrete-sac-implementation/12182
        epsilon = self.algorithm_spec.get('target_entropy_epsilon', 0.1)
        
        if self.agent.is_discrete:
            # Discrete: entropy of epsilon-greedy policy over discrete actions
            greedy_term = (1 - epsilon) * np.log((1 - epsilon) / (self.agent.action_dim - 1))
            self.target_entropy = -(epsilon * np.log(epsilon) + greedy_term)
        else:
            # Continuous: epsilon-greedy bound applied to standard -action_dim target
            action_dim = np.prod(self.agent.action_space.shape)
            self.target_entropy = -(1 - epsilon) * action_dim

        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        self.q1_optim = net_util.get_optim(self.q1_net, self.q1_net.optim_spec)
        self.q1_lr_scheduler = net_util.get_lr_scheduler(self.q1_optim, self.q1_net.lr_scheduler_spec)
        self.q2_optim = net_util.get_optim(self.q2_net, self.q2_net.optim_spec)
        self.q2_lr_scheduler = net_util.get_lr_scheduler(self.q2_optim, self.q2_net.lr_scheduler_spec)
        self.alpha_optim = net_util.get_optim(self.log_alpha, self.net.optim_spec)
        self.alpha_lr_scheduler = net_util.get_lr_scheduler(self.alpha_optim, self.net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        
        # Move cached tensors to network device
        if not self.agent.is_discrete:
            device = self.net.device
            self._action_low = self._action_low.to(device)
            self._action_high = self._action_high.to(device)
            self._action_scale = self._action_scale.to(device)
            self._action_bias = self._action_bias.to(device)
            
        self.end_init_nets()

    @lab_api
    def act(self, state):
        if self.agent.env.get('frame') < self.training_start_step:
            action = policy_util.random(state, self)
        else:
            action = self.action_policy(state, self)
            if not self.agent.is_discrete:
                action = self.scale_action(torch.tanh(action))
        return self.to_action(action)

    def scale_action(self, action):
        '''Scale continuous actions from tanh range using cached tensors'''
        return action * self._action_scale + self._action_bias

    def guard_q_actions(self, actions):
        '''Guard to convert actions to one-hot for input to Q-network'''
        if self.agent.is_discrete and actions.shape[-1] != self.agent.action_dim:
            # Convert discrete action indices to one-hot encoding
            actions = F.one_hot(actions.long(), self.agent.action_dim).float()
        return actions

    def calc_log_prob_action(self, action_pd, reparam=False):
        '''Calculate log_probs and actions with option to reparametrize from paper eq. 11'''
        samples = action_pd.rsample() if reparam else action_pd.sample()
        if self.agent.is_discrete:  # this is straightforward using GumbelSoftmax
            actions = samples
            log_probs = action_pd.log_prob(actions)
        else:
            mus = samples
            actions = self.scale_action(torch.tanh(mus))
            if actions.dim() == 1:  # handle shape consistency for single actions
                actions = actions.unsqueeze(dim=-1)
            # paper Appendix C. Enforcing Action Bounds for continuous actions
            log_probs = (action_pd.log_prob(mus) - torch.log(1 - actions.pow(2) + 1e-6).sum(1))
        return log_probs, actions

    def calc_q(self, state, action, net):
        '''Forward-pass to calculate the predicted state-action-value from q1_net.'''
        if not self.agent.is_discrete and action.dim() == 1:  # handle shape consistency for single continuous action
            action = action.unsqueeze(dim=-1)
        q_pred = net(state, action).view(-1)
        return q_pred

    def calc_q_targets(self, batch):
        '''Q_tar = r + gamma * (target_Q(s', a') - alpha * log pi(a'|s'))'''
        next_states = batch['next_states']
        with torch.no_grad():
            pdparams = self.calc_pdparam(next_states)
            action_pd = policy_util.init_action_pd(self.agent.ActionPD, pdparams)
            next_log_probs, next_actions = self.calc_log_prob_action(action_pd)
            next_actions = self.guard_q_actions(next_actions)  # non-reparam discrete actions need to be converted into one-hot

            next_target_q1_preds = self.calc_q(next_states, next_actions, self.target_q1_net)
            next_target_q2_preds = self.calc_q(next_states, next_actions, self.target_q2_net)
            next_target_q_preds = torch.min(next_target_q1_preds, next_target_q2_preds)
            q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * (next_target_q_preds - self.alpha * next_log_probs)
        return q_targets

    def calc_reg_loss(self, preds, targets):
        '''Calculate the regression loss for V and Q values, using the same loss function from net_spec'''
        assert preds.shape == targets.shape, f'{preds.shape} != {targets.shape}'
        reg_loss = self.net.loss_fn(preds, targets)
        return reg_loss

    def calc_policy_loss(self, batch, log_probs, reparam_actions):
        '''policy_loss = alpha * log pi(f(a)|s) - Q1(s, f(a)), where f(a) = reparametrized action'''
        states = batch['states']
        q1_preds = self.calc_q(states, reparam_actions, self.q1_net)
        q2_preds = self.calc_q(states, reparam_actions, self.q2_net)
        q_preds = torch.min(q1_preds, q2_preds)
        policy_loss = (self.alpha * log_probs - q_preds).mean()
        return policy_loss

    def calc_alpha_loss(self, log_probs):
        alpha_loss = - (self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        return alpha_loss

    def try_update_per(self, q_preds, q_targets):
        if 'Prioritized' in util.get_class_name(self.agent.memory):  # PER
            with torch.no_grad():
                errors = (q_preds - q_targets).abs().cpu().numpy()
            self.agent.memory.update_priorities(errors)

    def train_alpha(self, alpha_loss):
        '''Custom method to train the alpha variable'''
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha_lr_scheduler.step()
        self.alpha = self.log_alpha.detach().exp()

    def train(self):
        '''Train actor critic by computing the loss in batch efficiently'''
        if self.to_train == 1:
            for _ in range(self.training_iter):
                batch = self.sample()
                self.agent.env.set_batch_size(len(batch))

                states = batch['states']
                actions = self.guard_q_actions(batch['actions'])
                q_targets = self.calc_q_targets(batch)
                # Q-value loss for both Q nets
                q1_preds = self.calc_q(states, actions, self.q1_net)
                q1_loss = self.calc_reg_loss(q1_preds, q_targets)
                self.q1_net.train_step(q1_loss, self.q1_optim, self.q1_lr_scheduler, global_net=self.global_q1_net)
                self.agent.env.tick_opt_step()

                q2_preds = self.calc_q(states, actions, self.q2_net)
                q2_loss = self.calc_reg_loss(q2_preds, q_targets)
                self.q2_net.train_step(q2_loss, self.q2_optim, self.q2_lr_scheduler, global_net=self.global_q2_net)
                self.agent.env.tick_opt_step()

                # policy loss
                action_pd = policy_util.init_action_pd(self.agent.ActionPD, self.calc_pdparam(states))
                log_probs, reparam_actions = self.calc_log_prob_action(action_pd, reparam=True)
                policy_loss = self.calc_policy_loss(batch, log_probs, reparam_actions)
                self.net.train_step(policy_loss, self.optim, self.lr_scheduler, global_net=self.global_net)
                self.agent.env.tick_opt_step()

                # alpha loss
                alpha_loss = self.calc_alpha_loss(log_probs)
                self.train_alpha(alpha_loss)

                loss = q1_loss + q2_loss + policy_loss + alpha_loss
                # update target networks
                self.update_nets()
                # update PER priorities if availalbe
                self.try_update_per(torch.min(q1_preds, q2_preds), q_targets)

            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {self.agent.env.get("epi")}, frame: {self.agent.env.get("frame")}, t: {self.agent.env.get("t")}, total_reward so far: {self.agent.env.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    def update_nets(self):
        '''Update target networks'''
        if util.frame_mod(self.agent.env.get('frame'), self.q1_net.update_frequency, self.agent.env.num_envs):
            if self.q1_net.update_type == 'replace':
                net_util.copy(self.q1_net, self.target_q1_net)
                net_util.copy(self.q2_net, self.target_q2_net)
            elif self.q1_net.update_type == 'polyak':
                net_util.polyak_update(self.q1_net, self.target_q1_net, self.q1_net.polyak_coef)
                net_util.polyak_update(self.q2_net, self.target_q2_net, self.q2_net.polyak_coef)
            else:
                raise ValueError('Unknown q1_net.update_type. Should be "replace" or "polyak". Exiting.')

    @lab_api
    def update(self):
        '''Override parent method to do nothing'''
        return self.agent.explore_var
