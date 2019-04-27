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
        "lam": 1.0,
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
        "training_frequency": 1,
        "training_epoch": 8,
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
            entropy_coef_spec=None,
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
            'training_frequency',  # horizon
            'training_epoch',
            'normalize_state',
        ])
        self.to_train = 0
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
        super(PPO, self).init_nets(global_nets)
        # create old net to calculate ratio
        self.old_net = deepcopy(self.net)
        assert id(self.old_net) != id(self.net)

    def calc_policy_loss(self, batch, advs):
        '''
        The PPO loss function (subscript t is omitted)
        L^{CLIP+VF+S} = E[ L^CLIP - c1 * L^VF + c2 * S[pi](s) ]

        Breakdown piecewise,
        1. L^CLIP = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
        where ratio = pi(a|s) / pi_old(a|s)

        2. L^VF = E[ mse(V(s_t), V^target) ]

        3. S = E[ entropy ]
        '''
        clip_eps = self.body.clip_eps

        # L^CLIP
        log_probs = policy_util.calc_log_probs(self, self.net, self.body, batch)
        old_log_probs = policy_util.calc_log_probs(self, self.old_net, self.body, batch).detach()
        assert log_probs.shape == old_log_probs.shape
        assert advs.shape[0] == log_probs.shape[0]  # batch size
        ratios = torch.exp(log_probs - old_log_probs)  # clip to prevent overflow
        logger.debug(f'ratios: {ratios}')
        sur_1 = ratios * advs
        sur_2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advs
        # flip sign because need to maximize
        clip_loss = -torch.mean(torch.min(sur_1, sur_2))
        logger.debug(f'clip_loss: {clip_loss}')

        # L^VF (inherit from ActorCritic)

        # S entropy bonus
        entropies = torch.stack(self.body.entropies)
        ent_penalty = torch.mean(-self.body.entropy_coef * entropies)
        logger.debug(f'ent_penalty: {ent_penalty}')

        policy_loss = clip_loss + ent_penalty
        logger.debug(f'PPO Actor policy loss: {policy_loss:g}')
        return policy_loss

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        clock = self.body.env.clock
        if self.to_train == 1:
            # update old net
            torch.cuda.empty_cache()
            net_util.copy(self.net, self.old_net)
            batch = self.sample()
            total_loss = torch.tensor(0.0, device=self.net.device)
            for _ in range(self.training_epoch):
                with torch.no_grad():
                    advs, v_targets = self.calc_advs_v_targets(batch)
                policy_loss = self.calc_policy_loss(batch, advs)  # from actor
                val_loss = self.calc_val_loss(batch, v_targets)  # from critic
                loss = policy_loss + val_loss
                # retain for entropies etc.
                self.net.training_step(loss=loss, lr_clock=clock, retain_graph=True)
                total_loss += loss
            loss = total_loss / self.training_epoch
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, total_t: {clock.total_t}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    def train_separate(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        clock = self.body.env.clock
        if self.to_train == 1:
            torch.cuda.empty_cache()
            net_util.copy(self.net, self.old_net)
            batch = self.sample()
            policy_loss = self.train_actor(batch)
            val_loss = self.train_critic(batch)
            loss = val_loss + policy_loss
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, total_t: {clock.total_t}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    def train_actor(self, batch):
        '''Trains the actor when the actor and critic are separate networks'''
        total_policy_loss = torch.tensor(0.0, device=self.net.device)
        for _ in range(self.training_epoch):
            with torch.no_grad():
                advs, _v_targets = self.calc_advs_v_targets(batch)
            policy_loss = self.calc_policy_loss(batch, advs)
            # retain for entropies etc.
            self.net.training_step(loss=policy_loss, lr_clock=self.body.env.clock, retain_graph=True)
        val_loss = total_policy_loss / self.training_epoch
        return policy_loss

    @lab_api
    def update(self):
        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        if self.entropy_coef_spec is not None:
            self.body.entropy_coef = self.entropy_coef_scheduler.update(self, self.body.env.clock)
        self.body.clip_eps = self.clip_eps_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var
