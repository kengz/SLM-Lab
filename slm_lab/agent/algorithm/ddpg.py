from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
import torch

logger = logger.get_logger(__name__)


class OUNoise:

    def __init__(self, action_dim, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


class DDPG(ActorCritic):
    '''
    Implementation of Deep Deterministic Policy Gradient (DDPG)
    Original paper: "Continuous Control with Deep Reinforcement Learning"
    https://arxiv.org/pdf/1509.02971.pdf
    This is similar to ActorCritic, but instead the critic is a Q function, policy is direct, and add noise
    e.g. algorithm_spec
    "algorithm": {
        "name": "DDPG",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "policy_loss_coef": 1.0,
        "val_loss_coef": 0.01,
        "training_frequency": 1,
        "training_start_step": 10,
        "normalize_state": true
    }

    e.g. special net_spec param "shared" to share/separate Actor/Critic
    "net": {
        "type": "MLPNet",
        "shared": true,
        ...
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            explore_var_spec=None,
            policy_loss_coef=1.0,
            val_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, AC does not have policy update; but in this implementation we have such option
            'explore_var_spec',
            'gamma',  # the discount factor
            'policy_loss_coef',
            'val_loss_coef',
            'training_frequency',
            'training_start_step',  # how long before starting training
            'normalize_state',
        ])
        self.to_train = 0
        # self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.body.explore_var = self.explore_var_scheduler.start_val

        # initialize noise process for action
        self.noise_pd = OUNoise(self.body.action_dim)

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize actor(s) = a, and critic(s, a) = q,
        and their target networks
        '''
        # create actor/critic specific specs
        actor_net_spec = self.net_spec.copy()
        critic_net_spec = self.net_spec.copy()
        for k in self.net_spec:
            if 'actor_' in k:
                actor_net_spec[k.replace('actor_', '')] = actor_net_spec.pop(k)
                critic_net_spec.pop(k)
            if 'critic_' in k:
                critic_net_spec[k.replace('critic_', '')] = critic_net_spec.pop(k)
                actor_net_spec.pop(k)
        if critic_net_spec['use_same_optim']:
            critic_net_spec = actor_net_spec
        critic_net_spec.pop('out_layer_activation', None)  # dont use activation for critic output

        if global_nets is None:
            # actor(s) = a
            state_dim = self.body.state_dim
            action_dim = self.body.action_dim
            NetClass = getattr(net, actor_net_spec['type'])
            self.net = NetClass(actor_net_spec, state_dim, action_dim)
            self.target_net = NetClass(actor_net_spec, state_dim, action_dim)
            self.net_names = ['net', 'target_net']

            # critic(s, a) = q
            critic_in_dim = state_dim + action_dim
            # fuck architecture
            critic_out_dim = 1
            CriticNetClass = getattr(net, critic_net_spec['type'])
            self.critic = CriticNetClass(critic_net_spec, critic_in_dim, critic_out_dim)
            self.target_critic = CriticNetClass(critic_net_spec, critic_in_dim, critic_out_dim)
            self.net_names += ['critic', 'target_critic']
        else:
            util.set_attr(self, global_nets)
            self.net_names = list(global_nets.keys())
        self.post_init_nets()

    @lab_api
    def calc_pdparam(self, x, net=None):
        raise NotImplementedError('DDPG does not output pdparam')

    @lab_api
    def act(self, state):
        body = self.body
        if self.normalize_state:
            state = policy_util.update_online_stats_and_normalize_state(body, state)
        # TODO also clamp actor output
        state = policy_util.try_preprocess(state, self, body)
        noise = self.body.explore_var * self.noise_pd.sample()[0]
        action = self.net(state)
        return action.cpu().squeeze().numpy() + noise  # squeeze to handle scalar

    def calc_policy_loss(self, batch):
        '''Calculate the actor's policy loss'''
        states = batch['states']
        actions = self.net(states)
        q_preds = self.critic(torch.cat([states, actions], dim=-1))
        policy_loss = -q_preds.mean()
        logger.debug(f'Actor policy loss: {policy_loss:g}')
        return policy_loss

    def calc_val_loss(self, batch):
        '''Calculate the critic's value loss'''
        # TODO generate next_actions
        states = batch['states']
        actions = batch['actions'].unsqueeze(-1)
        next_states = batch['next_states']
        # TODO venv unpack
        q_preds = self.critic(torch.cat([states, actions], dim=-1))
        with torch.no_grad():
            next_actions = self.target_net(next_states)
            next_q_preds = self.target_critic(torch.cat([next_states, next_actions], dim=-1))
        q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * next_q_preds
        logger.debug(f'q_targets: {q_targets}')
        val_loss = self.net.loss_fn(q_preds, q_targets)
        logger.debug(f'Critic val loss: {val_loss:g}')
        return val_loss

    def train(self):
        '''Train actor critic by computing the loss in batch efficiently'''
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            batch = self.sample()
            policy_loss = self.calc_policy_loss(batch)  # from actor
            val_loss = self.calc_val_loss(batch)  # from critic
            self.net.training_step(loss=policy_loss, lr_clock=clock)
            self.critic.training_step(loss=val_loss, lr_clock=clock)
            loss = policy_loss + val_loss
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, total_t: {clock.total_t}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    def update_nets(self):
        total_t = self.body.env.clock.total_t
        if total_t % self.net.update_frequency == 0:
            if self.net.update_type == 'replace':
                net_util.copy(self.net, self.target_net)
            elif self.net.update_type == 'polyak':
                net_util.polyak_update(self.net, self.target_net, self.net.polyak_coef)
            else:
                raise ValueError('Unknown net.update_type. Should be "replace" or "polyak". Exiting.')
        if total_t % self.critic.update_frequency == 0:
            if self.critic.update_type == 'replace':
                net_util.copy(self.critic, self.target_critic)
            elif self.critic.update_type == 'polyak':
                net_util.polyak_update(self.critic, self.target_critic, self.critic.polyak_coef)
            else:
                raise ValueError('Unknown critic.update_type. Should be "replace" or "polyak". Exiting.')

    @lab_api
    def update(self):
        self.update_nets()
        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var
