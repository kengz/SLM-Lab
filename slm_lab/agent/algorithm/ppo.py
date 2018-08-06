from copy import deepcopy
from functools import partial
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

    e.g. algorithm_spec
    "algorithm": {
        "name": "PPO",
        "action_pdtype": "default",
        "action_policy": "default",
        "action_policy_update": "no_update",
        "explore_var_start": null,
        "explore_var_end": null,
        "explore_anneal_epi": null,
        "gamma": 0.99,
        "lam": 1.0,
        "clip_eps": 0.10,
        "entropy_coef": 0.02,
        "training_frequency": 1,
        "training_epoch": 8
    }
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
            val_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, PPO does not have policy update; but in this implementation we have such option
            'action_policy_update',
            'explore_var_start',
            'explore_var_end',
            'explore_anneal_epi',
            'gamma',
            'lam',
            'clip_eps',
            'entropy_coef',
            'val_loss_coef',
            'training_frequency',  # horizon
            'training_epoch',
        ])
        # use the same annealing epi as lr
        self.clip_eps_anneal_epi = self.net_spec['lr_decay_min_timestep'] + self.net_spec['lr_decay_frequency'] * 20
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.action_policy_update = getattr(policy_util, self.action_policy_update)
        for body in self.agent.nanflat_body_a:
            body.explore_var = self.explore_var_start
        # PPO uses GAE
        self.calc_advs_v_targets = self.calc_gae_advs_v_targets

    @lab_api
    def init_nets(self):
        '''PPO uses old and new to calculate ratio for loss'''
        super(PPO, self).init_nets()
        # create old net to calculate ratio
        self.old_net = deepcopy(self.net)
        assert id(self.old_net) != id(self.net)

    def calc_log_probs(self, batch, use_old_net=False):
        '''Helper method to calculate log_probs with the option to swith net'''
        if use_old_net:  # temporarily swap to do calc
            self.tmp_net = self.net
            self.net = self.old_net
        log_probs = policy_util.calc_log_probs(self, self.body, batch)
        if use_old_net:  # swap back
            self.old_net = self.net
            self.net = self.tmp_net
        return log_probs

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
        # decay clip_eps by episode
        clip_eps = policy_util._linear_decay(self.clip_eps, 0.1 * self.clip_eps, self.clip_eps_anneal_epi, self.body.env.clock.get('epi'))

        # L^CLIP
        log_probs = self.calc_log_probs(batch)
        old_log_probs = self.calc_log_probs(batch, use_old_net=True)
        assert log_probs.shape == old_log_probs.shape
        assert advs.shape[0] == log_probs.shape[0]  # batch size
        ratios = torch.exp(log_probs - old_log_probs)
        logger.debug(f'ratios: {ratios}')
        sur_1 = ratios * advs
        sur_2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advs
        # flip sign because need to maximize
        clip_loss = -torch.mean(torch.min(sur_1, sur_2))
        logger.debug(f'clip_loss: {clip_loss}')

        # L^VF (inherit from ActorCritic)

        # S entropy bonus
        entropies = torch.stack(self.body.entropies)
        ent_penalty = torch.mean(-self.entropy_coef * entropies)
        logger.debug(f'ent_penalty: {ent_penalty}')

        policy_loss = clip_loss + ent_penalty
        if torch.cuda.is_available() and self.net.gpu:
            policy_loss = policy_loss.cuda()
        logger.debug(f'Actor policy loss: {policy_loss:.4f}')
        return policy_loss

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        if self.to_train == 1:
            # update old net
            net_util.copy(self.net, self.old_net)
            batch = self.sample()
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                with torch.no_grad():
                    advs, v_targets = self.calc_advs_v_targets(batch)
                policy_loss = self.calc_policy_loss(batch, advs)  # from actor
                val_loss = self.calc_val_loss(batch, v_targets)  # from critic
                loss = policy_loss + val_loss
                # retain for entropies etc.
                self.net.training_step(loss=loss, retain_graph=True)
                total_loss += loss.cpu()
            loss = total_loss / self.training_epoch
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Loss: {loss:.4f}')
            self.last_loss = loss.item()
        return self.last_loss

    def train_separate(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        if self.to_train == 1:
            net_util.copy(self.net, self.old_net)
            batch = self.sample()
            policy_loss = self.train_actor(batch)
            val_loss = self.train_critic(batch)
            loss = val_loss + abs(policy_loss)
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Loss: {loss:.4f}')
            self.last_loss = loss.item()
        return self.last_loss

    def train_actor(self, batch):
        '''Trains the actor when the actor and critic are separate networks'''
        total_policy_loss = torch.tensor(0.0)
        for _ in range(self.training_epoch):
            with torch.no_grad():
                advs, _v_targets = self.calc_advs_v_targets(batch)
            policy_loss = self.calc_policy_loss(batch, advs)
            # retain for entropies etc.
            self.net.training_step(loss=policy_loss, retain_graph=True)
        val_loss = total_policy_loss / self.training_epoch
        return policy_loss
