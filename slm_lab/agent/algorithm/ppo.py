from copy import deepcopy
from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch
import pydash as ps

logger = logger.get_logger(__name__)


class PPO(ActorCritic):
    '''
    Implementation of PPO
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
            'clip_eps',
            'ent_coef',
            'epoch',
            'gamma',
            'horizon',
            'lam',
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
        # target advantage function
        adv_target = None # computed advs, then standardized
        # empirical return
        v_target = None # tdlamrets
        # learning rate multiplier for schedule
        # lr_mult = tf.placeholder(name='lr_mult', dtype=tf.float32, shape=[])
        # annealed clipping param epsilon for L^CLIP
        # do linear decay with same annealing as lr
        clip_eps = self.clip_eps * lr_mult

        # L^CLIP
        # the log probs are saved in body. though make another one for old log_prob
        # probably good to recalculate per batch action due to shuffling or replay data
        # recalculate per batch, or save during body_act
        ratio = tf.exp(pi.pd.logp(ac) - pi_old.pd.logp(ac))
        sur_1 = ratio * adv_target
        sur_2 = tf.clip_by_value(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_target
        # flip sign because need to maximize
        loss_clip = -tf.reduce_mean(tf.minimum(sur_1, sur_2))

        # L^VF
        loss_vf = tf.losses.mean_squared_error(v_target, pi.v_pred)

        # S entropy bonus, some variables for diagnosis
        kl_mean = tf.reduce_mean(pi_old.pd.kl(pi.pd))
        # same per batch
        ent_mean = tf.reduce_mean(pi.pd.entropy())
        ent_penalty = -self.ent_coef * ent_mean

        loss = loss_clip + loss_vf + ent_penalty
        return loss

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        # TODO if just want raw rewards, skip conversion to torch batch
        batch = util.concat_batches(batches)
        batch = util.to_torch_nested_batch_ex_rewards(batch, self.net.gpu)
        return batch

    @lab_api
    def train(self):
        if self.to_train == 1:
            logger.debug2(f'Training...')
            # We only care about the rewards from the batch
            rewards = self.sample()['rewards']
            loss = self.calc_policy_loss(rewards)
            self.net.training_step(loss=loss)

            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Policy loss: {loss}')
            return loss.item()
        else:
            return np.nan

    def calc_policy_loss(self, batch):
        '''
        Returns the policy loss for a batch of data.
        For REINFORCE just rewards are passed in as the batch
        '''
        advantage = self.calc_advantage(batch)
        advantage = self.check_sizes(advantage)
        policy_loss = torch.tensor(0.0)
        for log_prob, a, e in zip(self.body.log_probs, advantage, self.body.entropies):
            logger.debug3(f'log prob: {log_prob.item()}, advantage: {a}, entropy: {e.item()}')
            if self.add_entropy:
                policy_loss += (-log_prob * a - self.entropy_weight * e)
            else:
                policy_loss += (-log_prob * a)
        return policy_loss

    def check_sizes(self, advantage):
        '''
        Checks that log probs, advantage, and entropy all have the same size
        Occassionally they do not, this is caused by first reward of an episode being nan. If they are not the same size, the function removes the elements of the log probs and entropy that correspond to nan rewards.
        '''
        body = self.body
        nan_idxs = body.memory.last_nan_idxs
        num_nans = sum(nan_idxs)
        assert len(nan_idxs) == len(body.log_probs)
        assert len(nan_idxs) == len(body.entropies)
        assert len(nan_idxs) - num_nans == advantage.size(0)
        logger.debug2(f'{num_nans} nans encountered when gathering data')
        if num_nans != 0:
            idxs = [x for x in range(len(nan_idxs)) if nan_idxs[x] == 1]
            logger.debug3(f'Nan indexes: {idxs}')
            for idx in idxs[::-1]:
                del body.log_probs[idx]
                del body.entropies[idx]
        assert len(body.log_probs) == advantage.size(0)
        assert len(body.entropies) == advantage.size(0)
        return advantage

    def calc_advantage(self, raw_rewards):
        '''Returns the advantage for each action'''
        advantage = []
        logger.debug3(f'Raw rewards: {raw_rewards}')
        for epi_rewards in raw_rewards:
            big_r = 0
            T = len(epi_rewards)
            returns = np.empty(T, 'float32')
            for t in reversed(range(T)):
                big_r = epi_rewards[t] + self.gamma * big_r
                returns[t] = big_r
            logger.debug3(f'Rewards: {returns}')
            returns = (returns - returns.mean()) / (returns.std() + 1e-08)
            returns = torch.from_numpy(returns)
            logger.debug3(f'Normalized returns: {returns}')
            advantage.append(returns)
        advantage = torch.cat(advantage)
        return advantage

    @lab_api
    def update(self):
        for net in [self.net]:
            net.update_lr()
        explore_vars = [self.action_policy_update(self, body) for body in self.agent.nanflat_body_a]
        explore_var_a = self.nanflat_to_data_a('explore_var', explore_vars)
        return explore_var_a
