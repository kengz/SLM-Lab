from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.reinforce import Reinforce
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
import torch

logger = logger.get_logger(__name__)


class ReturnNormalizer:
    """Running statistics normalizer for value targets (like SB3's RunningMeanStd).

    Tracks running mean/var of returns using Welford's algorithm and provides
    normalize/denormalize methods for consistent scaling across training.

    This is the key fix for environments with high returns (InvertedPendulum, Humanoid, etc.)
    where value targets can be 1000x larger than initial predictions.
    """

    def __init__(self, epsilon: float = 1e-8, clip: float = 10.0):
        self.epsilon = epsilon
        self.clip = clip
        # Running statistics using Welford's algorithm
        self.mean = 0.0
        self.var = 1.0  # Start at 1 to avoid division issues early
        self.count = 0
        self.m2 = 0.0  # Sum of squared differences
        self._warmup = 1000  # Number of samples before trusting variance

    def update(self, values: torch.Tensor) -> None:
        """Update running statistics with new values (batched Welford's)"""
        values_np = values.detach().cpu().numpy().flatten()
        for v in values_np:
            self.count += 1
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2
        # Update variance after enough samples
        if self.count > 1:
            self.var = self.m2 / self.count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize values by running std (not mean-centered for value function)"""
        # Use running statistics from the start, with minimum std to avoid explosion
        std = float(np.sqrt(self.var + self.epsilon))
        std = max(std, 1.0)  # Ensure minimum std of 1.0 to prevent explosion
        normalized = values / std
        return torch.clamp(normalized, -self.clip, self.clip)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize values back to original scale"""
        std = float(np.sqrt(self.var + self.epsilon))
        return values * std


class ActorCritic(Reinforce):
    '''
    Implementation of single threaded Advantage Actor Critic
    Original paper: "Asynchronous Methods for Deep Reinforcement Learning"
    https://arxiv.org/abs/1602.01783
    Algorithm specific spec param:
    memory.name: batch (through OnPolicyBatchReplay memory class) or episodic through (OnPolicyReplay memory class)
    lam: if not null, used as the lambda value of generalized advantage estimation (GAE) introduced in "High-Dimensional Continuous Control Using Generalized Advantage Estimation https://arxiv.org/abs/1506.02438. This lambda controls the bias variance tradeoff for GAE. Floating point value between 0 and 1. Lower values correspond to more bias, less variance. Higher values to more variance, less bias. Algorithm becomes A2C(GAE).
    num_step_returns: if lam is null and this is not null, specifies the number of steps for N-step returns from "Asynchronous Methods for Deep Reinforcement Learning". The algorithm becomes A2C(Nstep).
    If both lam and num_step_returns are null, use the default TD error. Then the algorithm stays as AC.
    net.type: whether the actor and critic should share params (e.g. through 'MLPNetShared') or have separate params (e.g. through 'MLPNetSeparate'). If param sharing is used then there is also the option to control the weight given to the policy and value components of the loss function through 'policy_loss_coef' and 'val_loss_coef'
    Algorithm - separate actor and critic:
        Repeat:
            1. Collect k examples
            2. Train the critic network using these examples
            3. Calculate the advantage of each example using the critic
            4. Multiply the advantage by the negative of log probability of the action taken, and sum all the values. This is the policy loss.
            5. Calculate the gradient the parameters of the actor network with respect to the policy loss
            6. Update the actor network parameters using the gradient
    Algorithm - shared parameters:
        Repeat:
            1. Collect k examples
            2. Calculate the target for each example for the critic
            3. Compute current estimate of state-value for each example using the critic
            4. Calculate the critic loss using a regression loss (e.g. square loss) between the target and estimate of the state-value for each example
            5. Calculate the advantage of each example using the rewards and critic
            6. Multiply the advantage by the negative of log probability of the action taken, and sum all the values. This is the policy loss.
            7. Compute the total loss by summing the value and policy lossses
            8. Calculate the gradient of the parameters of shared network with respect to the total loss
            9. Update the shared network parameters using the gradient

    e.g. algorithm_spec
    "algorithm": {
        "name": "ActorCritic",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "lam": 0.95,
        "num_step_returns": 100,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "policy_loss_coef": 1.0,
        "val_loss_coef": 0.01,
        "training_frequency": 1,
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
            policy_loss_coef=1.0,
            val_loss_coef=1.0,
            normalize_v_targets=False,  # Normalize value targets to prevent gradient explosion
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, AC does not have policy update; but in this implementation we have such option
            'explore_var_spec',
            'gamma',  # the discount factor
            'lam',
            'num_step_returns',
            'entropy_coef_spec',
            'policy_loss_coef',
            'val_loss_coef',
            'training_frequency',
            'normalize_v_targets',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.agent.explore_var = self.explore_var_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.agent.entropy_coef = self.entropy_coef_scheduler.start_val
            self.agent.mt.register_algo_var('entropy_coef', self.agent)
        # Initialize return normalizer for value target scaling (VecNormalize-style)
        if self.normalize_v_targets:
            self.return_normalizer = ReturnNormalizer()
        else:
            self.return_normalizer = None
        # Select appropriate methods to calculate advs and v_targets for training
        if self.lam is not None:
            self.calc_advs_v_targets = self.calc_gae_advs_v_targets
        elif self.num_step_returns is not None:
            # need to override training_frequency for nstep to be the same
            self.training_frequency = self.num_step_returns
            self.calc_advs_v_targets = self.calc_nstep_advs_v_targets
        else:
            self.calc_advs_v_targets = self.calc_ret_advs_v_targets

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural networks used to learn the actor and critic from the spec
        Below we automatically select an appropriate net based on two different conditions
        1. If the action space is discrete or continuous action
            - Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
            - Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        2. If the actor and critic are separate or share weights
            - If the networks share weights then the single network returns a list.
            - Continuous action spaces: The return list contains 3 elements: The first element contains the mean output for the actor (policy), the second element the std dev of the policy, and the third element is the state-value estimated by the network.
            - Discrete action spaces: The return list contains 2 element. The first element is a tensor containing the logits for a categorical probability distribution over the actions. The second element contains the state-value estimated by the network.
        3. If the network type is feedforward, convolutional, or recurrent
            - Feedforward and convolutional networks take a single state as input and require an OnPolicyReplay or OnPolicyBatchReplay memory
            - Recurrent networks take n states as input and use gymnasium's FrameStackObservation wrapper for sequence handling
        '''
        assert 'shared' in self.net_spec, 'Specify "shared" for ActorCritic network in net_spec'
        self.shared = self.net_spec['shared']

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

        in_dim = self.agent.state_dim
        out_dim = net_util.get_out_dim(self.agent, add_critic=self.shared)
        # main actor network, may contain out_dim self.shared == True
        NetClass = getattr(net, actor_net_spec['type'])
        self.net = NetClass(actor_net_spec, in_dim, out_dim)
        self.net_names = ['net']
        if not self.shared:  # add separate network for critic
            critic_out_dim = 1
            CriticNetClass = getattr(net, critic_net_spec['type'])
            self.critic_net = CriticNetClass(critic_net_spec, in_dim, critic_out_dim)
            self.net_names.append('critic_net')
        # init net optimizer and its lr scheduler
        # steps_per_schedule: frames processed per scheduler.step() call
        steps_per_schedule = self.training_frequency * self.agent.env.num_envs
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec, steps_per_schedule)
        if not self.shared:
            self.critic_optim = net_util.get_optim(self.critic_net, self.critic_net.optim_spec)
            self.critic_lr_scheduler = net_util.get_lr_scheduler(self.critic_optim, self.critic_net.lr_scheduler_spec, steps_per_schedule)
        net_util.set_global_nets(self, global_nets)
        self.end_init_nets()

    @lab_api
    def calc_pdparam(self, x, net=None):
        '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        out = super().calc_pdparam(x, net=net)
        if self.shared:
            assert ps.is_list(out), 'Shared output should be a list [pdparam, v]'
            if len(out) == 2:  # single policy
                pdparam = out[0]
            else:  # multiple-task policies, still assumes 1 value
                pdparam = out[:-1]
            self.v_pred = out[-1].view(-1)  # cache for loss calc to prevent double-pass
        else:  # out is pdparam
            pdparam = out
        return pdparam

    def calc_v(self, x, net=None, use_cache=True):
        '''
        Forward-pass to calculate the predicted state-value from critic_net.
        '''
        if self.shared:  # output: policy, value
            if use_cache:  # uses cache from calc_pdparam to prevent double-pass
                v_pred = self.v_pred
            else:
                net = self.net if net is None else net
                v_pred = net(x)[-1].view(-1)
        else:
            net = self.critic_net if net is None else net
            v_pred = net(x).view(-1)
        return v_pred

    def calc_pdparam_v(self, batch):
        '''Efficiently forward to get pdparam and v by batch for loss computation'''
        states = batch['states']
        if self.agent.env.is_venv:
            states = math_util.venv_unpack(states)
        pdparam = self.calc_pdparam(states)
        v_pred = self.calc_v(states)  # uses self.v_pred from calc_pdparam if self.shared
        return pdparam, v_pred

    def calc_ret_advs_v_targets(self, batch, v_preds):
        '''Calculate plain returns, and advs = rets - v_preds, v_targets = rets'''
        v_preds = v_preds.detach()  # adv does not accumulate grad
        if self.agent.env.is_venv:
            v_preds = math_util.venv_pack(v_preds, self.agent.env.num_envs)
        rets = math_util.calc_returns(batch['rewards'], batch['terminateds'], self.gamma)
        advs = rets - v_preds
        v_targets = rets
        if self.agent.env.is_venv:
            advs = math_util.venv_unpack(advs)
            v_targets = math_util.venv_unpack(v_targets)
        logger.debug(f'advs: {advs}\nv_targets: {v_targets}')
        return advs, v_targets

    def calc_nstep_advs_v_targets(self, batch, v_preds):
        '''
        Calculate N-step returns, and advs = nstep_rets - v_preds, v_targets = nstep_rets
        See n-step advantage under http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf
        '''
        next_states = batch['next_states'][-1]
        if not self.agent.env.is_venv:
            next_states = next_states.unsqueeze(dim=0)
        with torch.no_grad():
            next_v_pred = self.calc_v(next_states, use_cache=False)
        v_preds = v_preds.detach()  # adv does not accumulate grad
        if self.agent.env.is_venv:
            v_preds = math_util.venv_pack(v_preds, self.agent.env.num_envs)
        nstep_rets = math_util.calc_nstep_returns(batch['rewards'], batch['terminateds'], next_v_pred, self.gamma, self.num_step_returns)
        advs = nstep_rets - v_preds
        v_targets = nstep_rets
        if self.agent.env.is_venv:
            advs = math_util.venv_unpack(advs)
            v_targets = math_util.venv_unpack(v_targets)
        logger.debug(f'advs: {advs}\nv_targets: {v_targets}')
        return advs, v_targets

    def calc_gae_advs_v_targets(self, batch, v_preds):
        '''
        Calculate GAE, and advs = GAE, v_targets = advs + v_preds
        See GAE from Schulman et al. https://arxiv.org/pdf/1506.02438.pdf
        '''
        next_states = batch['next_states'][-1]
        if not self.agent.env.is_venv:
            next_states = next_states.unsqueeze(dim=0)
        with torch.no_grad():
            next_v_pred = self.calc_v(next_states, use_cache=False)
        v_preds = v_preds.detach()  # adv does not accumulate grad
        if self.agent.env.is_venv:
            v_preds = math_util.venv_pack(v_preds, self.agent.env.num_envs)
            next_v_pred = next_v_pred.unsqueeze(dim=0)
        v_preds_all = torch.cat((v_preds, next_v_pred), dim=0)
        advs = math_util.calc_gaes(batch['rewards'], batch['terminateds'], v_preds_all, self.gamma, self.lam)
        v_targets = advs + v_preds
        # NOTE: Advantage normalization moved to per-minibatch in training loop (like SB3)
        if self.agent.env.is_venv:
            advs = math_util.venv_unpack(advs)
            v_targets = math_util.venv_unpack(v_targets)
        logger.debug(f'advs: {advs}\nv_targets: {v_targets}')
        return advs, v_targets

    def calc_policy_loss(self, batch, pdparams, advs):
        '''Calculate the actor's policy loss'''
        return super().calc_policy_loss(batch, pdparams, advs)

    def calc_val_loss(self, v_preds, v_targets):
        '''Calculate the critic's value loss.

        If normalize_v_targets is enabled with return_normalizer, uses running statistics
        to normalize targets consistently across training (like SB3's VecNormalize).
        This enables the critic to learn values in a stable range regardless of
        the environment's actual return scale.
        '''
        assert v_preds.shape == v_targets.shape, f'{v_preds.shape} != {v_targets.shape}'

        if self.return_normalizer is not None:
            # Update running statistics with new targets
            self.return_normalizer.update(v_targets)
            # Normalize targets to ~N(0,1) for stable critic learning
            v_targets_norm = self.return_normalizer.normalize(v_targets)
            # Normalize predictions using same statistics
            v_preds_norm = self.return_normalizer.normalize(v_preds)
            val_loss = self.val_loss_coef * self.net.loss_fn(v_preds_norm, v_targets_norm)
        elif self.normalize_v_targets:
            # Fallback: batch normalization (less stable but prevents explosion)
            v_max = v_targets.abs().max() * 2 + 1e-8
            v_preds_clipped = torch.clamp(v_preds, -v_max, v_max)
            v_std = v_targets.std() + 1e-8
            v_preds_norm = v_preds_clipped / v_std
            v_targets_norm = v_targets / v_std
            val_loss = self.val_loss_coef * self.net.loss_fn(v_preds_norm, v_targets_norm)
        else:
            val_loss = self.val_loss_coef * self.net.loss_fn(v_preds, v_targets)

        logger.debug(f'Critic value loss: {val_loss:g}')
        return val_loss

    def train(self):
        '''Train actor critic by computing the loss in batch efficiently'''
        if self.to_train == 1:
            batch = self.sample()
            self.agent.env.set_batch_size(len(batch))
            pdparams, v_preds = self.calc_pdparam_v(batch)
            advs, v_targets = self.calc_advs_v_targets(batch, v_preds)
            policy_loss = self.calc_policy_loss(batch, pdparams, advs)  # from actor
            val_loss = self.calc_val_loss(v_preds, v_targets)  # from critic
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
            # Step LR scheduler once per training iteration
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if not self.shared and hasattr(self, 'critic_lr_scheduler') and self.critic_lr_scheduler is not None:
                self.critic_lr_scheduler.step()
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
        return self.agent.explore_var
