from copy import deepcopy
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic, ReturnNormalizer
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import math
import numpy as np
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
            normalize_v_targets=False,  # Normalize value targets to prevent gradient explosion
            clip_vloss=False,  # CleanRL-style value loss clipping (uses clip_eps)
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
            'normalize_v_targets',
            'clip_vloss',
        ])
        self.to_train = 0
        # guard
        num_envs = self.agent.env.num_envs
        if self.minibatch_size % num_envs != 0 or self.time_horizon % num_envs != 0:
            self.minibatch_size = math.ceil(self.minibatch_size / num_envs) * num_envs
            self.time_horizon = math.ceil(self.time_horizon / num_envs) * num_envs
            logger.info(f'minibatch_size and time_horizon needs to be multiples of num_envs; autocorrected values: minibatch_size: {self.minibatch_size}  time_horizon {self.time_horizon}')
        # Ensure minibatch_size doesn't exceed batch_size
        batch_size = self.time_horizon * num_envs
        if self.minibatch_size > batch_size:
            self.minibatch_size = batch_size
            logger.info(f'minibatch_size cannot exceed batch_size ({batch_size}); autocorrected to: {self.minibatch_size}')
        self.training_frequency = self.time_horizon  # since all memories stores num_envs by batch in list
        assert self.memory_spec['name'] == 'OnPolicyBatchReplay', f'PPO only works with OnPolicyBatchReplay, but got {self.memory_spec["name"]}'
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.agent.explore_var = self.explore_var_scheduler.start_val
        # extra variable decays for PPO
        self.clip_eps_scheduler = policy_util.VarScheduler(self.clip_eps_spec)
        self.clip_eps = self.clip_eps_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.agent.entropy_coef = self.entropy_coef_scheduler.start_val
        # Initialize return normalizer for value target scaling (VecNormalize-style)
        if self.normalize_v_targets:
            self.return_normalizer = ReturnNormalizer()
        else:
            self.return_normalizer = None
        # PPO uses GAE
        self.calc_advs_v_targets = self.calc_gae_advs_v_targets
        # Register PPO-specific variables for logging
        self.agent.mt.register_algo_var('clip_eps', self)
        if self.entropy_coef_spec is not None:
            self.agent.mt.register_algo_var('entropy', self.agent)

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
        clip_eps = self.clip_eps
        action_pd = policy_util.init_action_pd(self.agent.ActionPD, pdparams)
        states = batch['states']
        actions = batch['actions']
        if self.agent.env.is_venv:
            states = math_util.venv_unpack(states)
            actions = math_util.venv_unpack(actions)

        # Ensure advs is always 1D regardless of venv to match log_probs shape
        advs = advs.view(-1)

        # Normalize advantages per minibatch (like SB3)
        if len(advs) > 1:
            advs = math_util.standardize(advs)

        # L^CLIP
        log_probs = policy_util.reduce_multi_action(action_pd.log_prob(actions))
        with torch.no_grad():
            old_pdparams = self.calc_pdparam(states, net=self.old_net)
            old_action_pd = policy_util.init_action_pd(self.agent.ActionPD, old_pdparams)
            old_log_probs = policy_util.reduce_multi_action(old_action_pd.log_prob(actions))
        assert log_probs.shape == old_log_probs.shape, f'log_probs shape {log_probs.shape} != old_log_probs shape {old_log_probs.shape}'
        # Clip log ratio to prevent numerical instability (exp overflow)
        log_ratio = torch.clamp(log_probs - old_log_probs, -20.0, 20.0)
        ratios = torch.exp(log_ratio)
        sur_1 = ratios * advs
        sur_2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advs
        # flip sign because need to maximize
        clip_loss = -torch.min(sur_1, sur_2).mean()

        # L^VF (inherit from ActorCritic)

        # H entropy regularization
        entropy = policy_util.reduce_multi_action(action_pd.entropy()).mean()
        self.agent.entropy = entropy.detach()  # Update value for logging
        ent_penalty = -self.agent.entropy_coef * entropy

        policy_loss = clip_loss + ent_penalty
        logger.debug(f'PPO policy loss: {policy_loss:g}')
        return policy_loss

    def calc_val_loss(self, v_preds, v_targets, old_v_preds=None):
        '''Calculate PPO value loss with optional CleanRL-style value clipping.

        When clip_vloss=True, clips value predictions relative to old predictions
        similar to policy clipping. This can improve stability for some environments.

        Args:
            v_preds: Current value predictions
            v_targets: GAE-computed value targets
            old_v_preds: Value predictions from before network update (for clipping)
        '''
        if self.clip_vloss and old_v_preds is not None:
            # CleanRL-style value clipping
            v_loss_unclipped = (v_preds - v_targets) ** 2
            v_clipped = old_v_preds + torch.clamp(
                v_preds - old_v_preds,
                -self.clip_eps,
                self.clip_eps,
            )
            v_loss_clipped = (v_clipped - v_targets) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            val_loss = 0.5 * self.val_loss_coef * v_loss_max.mean()
            logger.debug(f'PPO clipped value loss: {val_loss:g}')
            return val_loss
        else:
            # Standard value loss (inherited from ActorCritic)
            return super().calc_val_loss(v_preds, v_targets)

    def train(self):
        if self.to_train == 1:
            net_util.copy(self.net, self.old_net)  # update old net
            batch = self.sample()
            self.agent.env.set_batch_size(len(batch))
            with torch.no_grad():
                states = batch['states']
                if self.agent.env.is_venv:
                    states = math_util.venv_unpack(states)
                # NOTE states is massive with batch_size = time_horizon * num_envs. Chunk up so forward pass can fit into device esp. GPU
                num_chunks = max(1, int(len(states) / self.minibatch_size))
                v_preds_chunks = [self.calc_v(states_chunk, use_cache=False) for states_chunk in torch.chunk(states, num_chunks)]
                v_preds = torch.cat(v_preds_chunks)
                advs, v_targets = self.calc_advs_v_targets(batch, v_preds)
            # piggy back on batch, but remember to not pack or unpack
            # Store old v_preds for value clipping (CleanRL-style)
            batch['advs'], batch['v_targets'], batch['old_v_preds'] = advs, v_targets, v_preds
            if self.agent.env.is_venv:  # unpack if venv for minibatch sampling
                for k, v in batch.items():
                    if k not in ('advs', 'v_targets', 'old_v_preds'):
                        batch[k] = math_util.venv_unpack(v)
            total_loss = torch.tensor(0.0, device=self.net.device)
            for _ in range(self.training_epoch):
                minibatches = util.split_minibatch(batch, self.minibatch_size)
                for minibatch in minibatches:
                    if self.agent.env.is_venv:  # re-pack to restore proper shape
                        for k, v in minibatch.items():
                            if k not in ('advs', 'v_targets', 'old_v_preds'):
                                minibatch[k] = math_util.venv_pack(v, self.agent.env.num_envs)
                    advs, v_targets, old_v_preds = minibatch['advs'], minibatch['v_targets'], minibatch['old_v_preds']
                    pdparams, v_preds = self.calc_pdparam_v(minibatch)
                    policy_loss = self.calc_policy_loss(minibatch, pdparams, advs)  # from actor
                    val_loss = self.calc_val_loss(v_preds, v_targets, old_v_preds)  # from critic
                    if self.shared:  # shared network
                        loss = policy_loss + val_loss
                        self.net.train_step(loss, self.optim, self.lr_scheduler, global_net=self.global_net)
                        self.agent.env.tick_opt_step()
                    else:
                        self.net.train_step(policy_loss, self.optim, self.lr_scheduler, global_net=self.global_net)
                        self.critic_net.train_step(val_loss, self.critic_optim, self.critic_lr_scheduler, global_net=self.global_critic_net)
                        self.agent.env.tick_opt_step()
                        self.agent.env.tick_opt_step()
                        loss = policy_loss + val_loss
                    total_loss += loss
            # Step LR scheduler once per training iteration (per batch of collected experience)
            # This ensures proper LR decay matching CleanRL's approach
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if not self.shared and hasattr(self, 'critic_lr_scheduler') and self.critic_lr_scheduler is not None:
                self.critic_lr_scheduler.step()
            loss = total_loss / self.training_epoch / len(minibatches)
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {self.agent.env.get("epi")}, frame: {self.agent.env.get("frame")}, t: {self.agent.env.get("t")}, total_reward so far: {self.agent.env.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        self.agent.explore_var = self.explore_var_scheduler.update(self, self.agent.env)
        if self.entropy_coef_spec is not None:
            self.agent.entropy_coef = self.entropy_coef_scheduler.update(self, self.agent.env)
        self.clip_eps = self.clip_eps_scheduler.update(self, self.agent.env)
        return self.agent.explore_var
