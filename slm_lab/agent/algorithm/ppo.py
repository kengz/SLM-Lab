from copy import deepcopy
from slm_lab.agent import net
from slm_lab.agent.algorithm import math_util, policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch
import pydash as ps

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
    '''

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.body = self.agent.nanflat_body_a[0]  # single-body algo
        self.init_algorithm_params()
        self.init_nets()
        logger.info(util.self_desc(self))

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            action_policy_update='no_update',
            explore_var_start=np.nan,
            explore_var_end=np.nan,
            explore_anneal_epi=np.nan,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_policy',
            # theoretically, PPO does not have policy update; but in this implementation we have such option
            'action_policy_update',
            'explore_var_start', 'explore_var_end', 'explore_anneal_epi',
            'gamma',
            'lam',
            'clip_eps',
            'entropy_coef',
            'training_frequency',  # horizon
            'training_epoch',  # epoch
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.action_policy_update = getattr(policy_util, self.action_policy_update)
        for body in self.agent.nanflat_body_a:
            body.explore_var = self.explore_var_start

    @lab_api
    def init_nets(self):
        '''PPO uses old and new to calculate ratio for loss'''
        super(PPO, self).init_nets()
        if self.share_architecture:
            self.old_net = deepcopy(self.net)
        else:
            self.old_net = deepcopy(self.net)
            self.old_critic = deepcopy(self.critic)

    def calc_log_probs(self, batch, use_old_net=False):
        '''Helper method to calculate log_probs with the option to swith net'''
        if use_old_net:
            # temporarily swap to do calc
            self.tmp_net = self.net
            self.net = self.old_net
        states, actions = batch['states'], batch['actions']
        # get ActionPD
        ActionPD, _pdparam, _body = policy_util.init_action_pd(states[0].numpy(), self, self.body)
        # construct log_probs for each state-action
        ActionPD, _pdparam, _body = policy_util.init_action_pd(states[0].numpy(), self, self.body)
        pdparams = self.calc_pdparam(states)
        log_probs = []
        for idx, pdparam in enumerate(pdparams):
            _action, action_pd = policy_util.sample_action_pd(ActionPD, pdparam, self.body)
            log_prob = action_pd.log_prob(actions[idx])
            log_probs.append(log_prob)
        log_probs = torch.tensor(log_probs)
        if use_old_net:
            # swap back
            self.old_net = self.net
            self.net = self.tmp_net
        return log_probs

    def calc_loss(self, batch):
        '''
        The PPO loss function (subscript t is omitted)
        L^{CLIP+VF+S} = E[ L^CLIP - c1 * L^VF + c2 * S[pi](s) ]

        Breakdown piecewise,
        1. L^CLIP = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
        where ratio = pi(a|s) / pi_old(a|s)

        2. L^VF = E[ (V(s_t) - V^target)^2 ]

        3. S = E[ entropy ]
        '''
        with torch.no_grad():
            adv_targets, v_targets = self.calc_gae_advs_v_targets(batch)
        # TODO decay by some annealing param. use the policy_util method and put in update
        clip_eps = self.clip_eps

        # L^CLIP
        # body already keeps track of it. so just reuse
        log_probs = torch.tensor(self.body.log_probs)
        old_log_probs = self.calc_log_probs(batch, use_old_net=True)
        assert adv_targets.shape == old_log_probs.shape
        assert log_probs.shape == old_log_probs.shape
        ratios = torch.exp(log_probs - old_log_probs)
        sur_1 = ratios * adv_targets
        sur_2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * adv_targets
        # flip sign because need to maximize
        clip_loss = -torch.mean(torch.min(sur_1, sur_2))

        # L^VF
        val_loss = self.calc_val_loss(batch, v_targets)  # from critic

        # S entropy bonusdiagnosis
        ent_mean = torch.mean(torch.tensor(self.body.entropies))
        ent_penalty = -self.entropy_coef * ent_mean
        loss = clip_loss + val_loss + ent_penalty
        return loss

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        if self.to_train == 1:
            batch = self.sample()
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                loss = self.calc_loss(batch)
                self.net.training_step(loss=loss)
                total_loss += loss
            loss = total_loss.mean()
            net_util.copy(self.net, self.old_net)
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Loss: {loss:.2f}')
            return loss.item()
        else:
            return np.nan

    def train_separate(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        if self.to_train == 1:
            batch = self.sample()
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                # TODO double pass inefficient
                loss = self.calc_loss(batch)
                self.net.training_step(loss=loss)
                loss = self.calc_loss(batch)
                self.critic.training_step(loss=loss)
                total_loss += loss
            loss = total_loss.mean()
            net_util.copy(self.net, self.old_net)
            net_util.copy(self.critic, self.old_critic)
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Loss: {loss:.2f}')
            return loss.item()
        else:
            return np.nan
