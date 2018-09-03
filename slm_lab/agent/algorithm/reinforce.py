from slm_lab.agent import net
from slm_lab.agent.algorithm import math_util, policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
import torch

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
        "action_policy_update": "no_update",
        "explore_var_start": null,
        "explore_var_end": null,
        "explore_anneal_epi": null,
        "gamma": 0.99,
        "add_entropy": false,
        "entropy_coef": 0.01,
        "training_frequency": 1
    }
    '''

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
            'action_pdtype',
            'action_policy',
            # theoretically, REINFORCE does not have policy update; but in this implementation we have such option
            'action_policy_update',
            'explore_var_start',
            'explore_var_end',
            'explore_anneal_epi',
            'gamma',  # the discount factor
            'add_entropy',
            'entropy_coef',
            'training_frequency',
            'normalize_state'
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.action_policy_update = getattr(policy_util, self.action_policy_update)
        self.body.explore_var = self.explore_var_start

    @lab_api
    def init_nets(self):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPNet'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''
        in_dim = self.body.state_dim
        out_dim = net_util.get_out_dim(self.body)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim)
        self.net_names = ['net']
        self.post_init_nets()

    @lab_api
    def calc_pdparam(self, x, evaluate=True, net=None):
        '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        net = self.net if net is None else net
        if evaluate:
            pdparam = net.wrap_eval(x)
        else:
            net.train()
            pdparam = net(x)
        logger.debug(f'pdparam: {pdparam}')
        return pdparam

    @lab_api
    def act(self, state):
        body = self.body
        logger.debug(f'state: {state}')
        if self.normalize_state:
            state = policy_util.update_online_stats_and_normalize_state(
                body, state)
        logger.debug(f'mean: {body.state_mean}, std: {body.state_std_dev}, n: {body.state_n}')
        logger.debug(f'state after normalize: {state}')
        action, action_pd = self.action_policy(state, self, body)
        # sum for single and multi-action
        body.entropies.append(action_pd.entropy().sum(dim=0))
        body.log_probs.append(action_pd.log_prob(action.float()).sum(dim=0))
        assert not torch.isnan(body.log_probs[-1])
        if len(action.shape) == 0:  # scalar
            return action.cpu().numpy().astype(body.action_space.dtype).item()
        else:
            return action.cpu().numpy()

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batch = self.body.memory.sample()
        batch = util.to_torch_batch(batch, self.net.gpu, self.body.memory.is_episodic)
        return batch

    @lab_api
    def train(self):
        if util.get_lab_mode() == 'enjoy':
            return np.nan
        if self.to_train == 1:
            batch = self.sample()
            loss = self.calc_policy_loss(batch)
            self.net.training_step(loss=loss, global_net=self.global_nets.get('net'))
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Policy loss: {loss}')
            return loss.item()
        else:
            return np.nan

    def calc_policy_loss(self, batch):
        '''Calculate the policy loss for a batch of data.'''
        # use simple returns as advs
        advs = math_util.calc_returns(batch, self.gamma)
        advs = math_util.standardize(advs)
        logger.debug(f'advs: {advs}')
        assert len(self.body.log_probs) == len(advs), f'batch_size of log_probs {len(self.body.log_probs)} vs advs: {len(advs)}'
        log_probs = torch.stack(self.body.log_probs)
        policy_loss = - log_probs * advs
        if self.add_entropy:
            entropies = torch.stack(self.body.entropies)
            policy_loss += (-self.entropy_coef * entropies)
        policy_loss = torch.sum(policy_loss)
        if torch.cuda.is_available() and self.net.gpu:
            policy_loss = policy_loss.cuda()
        logger.debug(f'Actor policy loss: {policy_loss:.4f}')
        return policy_loss

    @lab_api
    def update(self):
        for net_name in self.net_names:
            net = getattr(self, net_name)
            net.update_lr(self.body.env.clock)
        explore_var = self.action_policy_update(self, self.body)
        return explore_var
