from copy import deepcopy
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


class PPO(ActorCritic):
    '''
    Implementation of PPO
    This is actually just ActorCritic with a custom loss function
    Original paper: "Proximal Policy Optimization Algorithms"
    https://arxiv.org/pdf/1707.06347.pdf

    Adapted from OpenAI baselines, CPU version https://github.com/openai/baselines/tree/master/baselines/ppo1
    Algorithm:
    for iteration = 1, 2, 3, ... do
        for actor = 1, 2, 3, ..., N do
            run policy pi_old in env for T timesteps
            compute advantage A_1, ..., A_T
        end for
        optimize surrogate L wrt theta, with K epochs and minibatch size M <= NT
    end for

    e.g. algorithm_spec
    "algorithm": {
        "name": "PPO",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "minibatch_size": 256,
        "time_horizon": 32,
        "training_epoch": 8,
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
            entropy_coef_spec=None,
            minibatch_size=4,
            val_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, PPO does not have policy update; but in this implementation we have such option
            'explore_var_spec',
            'gamma',
            'lam',
            'clip_eps_spec',
            'entropy_coef_spec',
            'val_loss_coef',
            'minibatch_size',
            'time_horizon',  # training_frequency = actor * horizon
            'training_epoch',
        ])
        self.to_train = 0
        self.training_frequency = self.time_horizon * self.body.env.num_envs
        assert self.memory_spec['name'] == 'OnPolicyBatchReplay', f'PPO only works with OnPolicyBatchReplay, but got {self.memory_spec["name"]}'
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.body.explore_var = self.explore_var_scheduler.start_val
        # extra variable decays for PPO
        self.clip_eps_scheduler = policy_util.VarScheduler(self.clip_eps_spec)
        self.body.clip_eps = self.clip_eps_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.body.entropy_coef = self.entropy_coef_scheduler.start_val
        # PPO uses GAE
        self.calc_advs_v_targets = self.calc_gae_advs_v_targets

    @lab_api
    def init_nets(self, global_nets=None):
        '''PPO uses old and new to calculate ratio for loss'''
        super().init_nets(global_nets)
        # create old net to calculate ratio
        self.old_net = deepcopy(self.net)
        assert id(self.old_net) != id(self.net)

    def calc_policy_loss(self, batch, pdparams, advs):
        '''
        The PPO loss function (subscript t is omitted)
        L^{CLIP+VF+S} = E[ L^CLIP - c1 * L^VF + c2 * H[pi](s) ]

        Breakdown piecewise,
        1. L^CLIP = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
        where ratio = pi(a|s) / pi_old(a|s)

        2. L^VF = E[ mse(V(s_t), V^target) ]

        3. H = E[ entropy ]
        '''
        clip_eps = self.body.clip_eps
        action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)
        states = batch['states']
        actions = batch['actions']
        if self.body.env.is_venv:
            states = math_util.venv_unpack(states)
            actions = math_util.venv_unpack(actions)

        # L^CLIP
        log_probs = action_pd.log_prob(actions)
        with torch.no_grad():
            old_pdparams = self.calc_pdparam(states, net=self.old_net)
            old_action_pd = policy_util.init_action_pd(self.body.ActionPD, old_pdparams)
            old_log_probs = old_action_pd.log_prob(actions)
        assert log_probs.shape == old_log_probs.shape
        ratios = torch.exp(log_probs - old_log_probs)
        logger.debug(f'ratios: {ratios}')
        sur_1 = ratios * advs
        sur_2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advs
        # flip sign because need to maximize
        clip_loss = -torch.min(sur_1, sur_2).mean()
        logger.debug(f'clip_loss: {clip_loss}')

        # L^VF (inherit from ActorCritic)

        # H entropy regularization
        entropy = action_pd.entropy().mean()
        self.body.mean_entropy = entropy  # update logging variable
        ent_penalty = -self.body.entropy_coef * entropy
        logger.debug(f'ent_penalty: {ent_penalty}')

        policy_loss = clip_loss + ent_penalty
        logger.debug(f'PPO Actor policy loss: {policy_loss:g}')
        return policy_loss

    def train(self):
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            net_util.copy(self.net, self.old_net)  # update old net
            batch = self.sample()
            clock.set_batch_size(len(batch))
            _pdparams, v_preds = self.calc_pdparam_v(batch)
            advs, v_targets = self.calc_advs_v_targets(batch, v_preds)
            # piggy back on batch, but remember to not pack or unpack
            batch['advs'], batch['v_targets'] = advs, v_targets
            if self.body.env.is_venv:  # unpack if venv for minibatch sampling
                for k, v in batch.items():
                    if k not in ('advs', 'v_targets'):
                        batch[k] = math_util.venv_unpack(v)
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                minibatches = util.split_minibatch(batch, self.minibatch_size)
                for minibatch in minibatches:
                    if self.body.env.is_venv:  # re-pack to restore proper shape
                        for k, v in minibatch.items():
                            if k not in ('advs', 'v_targets'):
                                minibatch[k] = math_util.venv_pack(v, self.body.env.num_envs)
                    advs, v_targets = minibatch['advs'], minibatch['v_targets']
                    pdparams, v_preds = self.calc_pdparam_v(minibatch)
                    policy_loss = self.calc_policy_loss(minibatch, pdparams, advs)  # from actor
                    val_loss = self.calc_val_loss(v_preds, v_targets)  # from critic
                    if self.shared:  # shared network
                        loss = policy_loss + val_loss
                        self.net.train_step(loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                    else:
                        self.net.train_step(policy_loss, self.optim, self.lr_scheduler, clock=clock, global_net=self.global_net)
                        self.critic_net.train_step(val_loss, self.critic_optim, self.critic_lr_scheduler, clock=clock, global_net=self.global_critic_net)
                        loss = policy_loss + val_loss
                    total_loss += loss
            loss = total_loss / self.training_epoch / len(minibatches)
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        if self.entropy_coef_spec is not None:
            self.body.entropy_coef = self.entropy_coef_scheduler.update(self, self.body.env.clock)
        self.body.clip_eps = self.clip_eps_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var
