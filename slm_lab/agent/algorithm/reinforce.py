from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib import math_util, util
from slm_lab.agent.net import net_util
from slm_lab.lib.decorator import lab_api
import numpy as np
from collections import deque
import torch

from slm_lab.lib import logger
logger = logger.get_logger(__name__)


class Reinforce(Algorithm):
    '''
    Implementation of REINFORCE (Williams, 1992) with baseline for discrete or continuous actions http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    Algorithm:
        0. Collect n episodes of data
        1. At each timestep in an episode
            - Calculate the advantage of that timestep
            - Multiply the advantage by the negative of the log probability of the action taken
        2. Sum all the values above.
        3. Calculate the gradient of this value with respect to all of the parameters of the network
        4. Update the network parameters using the gradient

    e.g. algorithm_spec:
    "algorithm": {
        "name": "Reinforce",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
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
            center_return=False,
            normalize_return=False,
            normalize_over_n_batch=1,
            explore_var_spec=None,
            entropy_coef_spec=None,
            policy_loss_coef=1.0,
            min_log_prob=False
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            'center_return',  # center by the mean
            'normalize_return', # divide by std
            'normalize_over_n_batch',
            'explore_var_spec',
            'gamma',  # the discount factor
            'entropy_coef_spec',
            'policy_loss_coef',
            'training_frequency',
            'min_log_prob'
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.internal_clock, self.explore_var_spec)
        self.explore_var_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.internal_clock, self.entropy_coef_spec)
        self.normalize_over_n_batch = int(self.normalize_over_n_batch)
        if self.normalize_over_n_batch != 1:
            self.batchs_values = deque(maxlen=self.normalize_over_n_batch)

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPNet'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''
        in_dim = self.body.observation_dim
        out_dim = net_util.get_out_dim(self.body)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim, self.internal_clock)
        self.net_names = ['net']
        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        self.post_init_nets()

    @lab_api
    def calc_pdparam(self, x, net=None):
        '''The pdparam (proba distrib param) will be the logits for discrete prob. dist., or the mean and std for
        continuous prob. dist.'''
        net = self.net if net is None else net
        pdparam = net(x)
        return pdparam

    @lab_api
    def act(self, state):
        body = self.body
        action, action_pd = self.action_policy(state, self, body)
        self.to_log["entropy_act"] = action_pd.entropy().mean().item()
        return action.cpu().squeeze().numpy(), action_pd  # squeeze to handle scalar

    #TODO reset currently doesn't support all memory types
    @lab_api
    def sample(self, reset=True):
        '''Samples a batch from memory'''
        batch = self.memory.sample(reset=reset)
        batch = util.to_torch_batch(batch, self.net.device, self.memory.is_episodic)
        return batch

    def calc_pdparam_batch(self, batch):
        '''Efficiently forward to get pdparam and by batch for loss computation'''
        states = batch['states']
        if self.body.env.is_venv:
            states = math_util.venv_unpack(states)
        pdparam = self.calc_pdparam(states)
        return pdparam

    def calc_ret_advs(self, batch):
        '''Calculate plain returns; which is generalized to advantage in ActorCritic'''
        rets = math_util.calc_returns(batch['rewards'], batch['dones'], self.gamma)
        if self.center_return:
            if self.normalize_over_n_batch != 1:
                self.batchs_values.append(rets)
                mean_rets = torch.cat(list(self.batchs_values), dim=0).mean()
                rets = rets - mean_rets
            else:
                rets = math_util.center_mean(rets)
        if self.normalize_return:
            if self.normalize_over_n_batch != 1:
                std_rets = torch.cat(list(self.batchs_values), dim=0).std()
                rets = rets / std_rets
            else:
                rets = math_util.normalize_var(rets)
        advs = rets
        if self.body.env.is_venv:
            advs = math_util.venv_unpack(advs)
        return advs

    def calc_policy_loss(self, batch, pdparams, advs):
        '''Calculate the actor's policy loss'''
        action_pd = policy_util.init_action_pd(self.body.ActionPD, pdparams)
        actions = batch['actions']
        if self.body.env.is_venv:
            actions = math_util.venv_unpack(actions)
        logger.debug(f'actions {actions}')

        log_probs = action_pd.log_prob(actions)
        if self.min_log_prob != False:
            log_probs = log_probs.clamp(min=self.min_log_prob)
        logger.debug(f'log_probs {log_probs}, advs {advs}')
        self.to_log['max_log_probs'] = log_probs.abs().max()
        policy_loss = - self.policy_loss_coef * (log_probs * advs).mean()
        if self.entropy_coef_spec:
            self.entropy = action_pd.entropy().mean()
            logger.debug(f'entropy {self.entropy}')
            self.to_log["entropy_train"] = self.entropy.item()
            entropy_loss = (-self.entropy_coef_scheduler.val * self.entropy)
            # if policy_loss != 0.0:
            self.to_log["entropy_over_loss"] = (entropy_loss / policy_loss + 1e-12).clamp(min=-100, max=100)
            self.to_log["loss_policy"] = policy_loss
            policy_loss += entropy_loss
        logger.debug(f'Actor policy loss: {policy_loss:g}')
        return policy_loss

    @lab_api
    def train(self):
        if util.in_eval_lab_modes():
            return np.nan
        if self.to_train == 1:
            batch = self.sample()
            pd_param = self.calc_pdparam_batch(batch)
            advs = self.calc_ret_advs(batch)
            loss = self.calc_policy_loss(batch, pd_param, advs)

            # TODO use either arg or attribute but not both (auxilary vs loss penalty)
            if hasattr(self, "auxilary_loss"):
                # if not torch.isnan(self.auxilary_loss):
                action_pd = policy_util.init_action_pd(self.body.ActionPD, pd_param)
                coop_entropy = action_pd.entropy().mean()
                ent_diff = (coop_entropy -
                            (self.auxilary_loss * 0.95) + 0.01) ** 2
                auxilary_loss = ent_diff * self.strat_5_coeff
                self.to_log['loss_auxilary'] = auxilary_loss
                loss += auxilary_loss
                del self.auxilary_loss
                del self.strat_5_coeff

            if hasattr(self, "lr_overwritter"):
                lr_source = self.lr_overwritter
                self.to_log['lr'] = self.lr_overwritter
            else:
                lr_source = self.lr_scheduler
            self.net.train_step(loss, self.optim, lr_source, clock=self.internal_clock, global_net=self.global_net)
            if hasattr(self, "lr_overwritter"):
                del self.lr_overwritter
            else:
                self.to_log['lr'] = np.mean(self.lr_scheduler.get_lr())

            # reset
            self.to_train = 0
            clock = self.body.env.clock
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.env.total_reward}, loss: {loss:g}')
            self.to_log["loss_tot"] = loss.item()
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        self.explore_var_scheduler.update(self, self.internal_clock)
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler.update(self, self.internal_clock)
        return self.explore_var_scheduler.val
