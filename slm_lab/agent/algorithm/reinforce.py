from slm_lab.agent import net
from slm_lab.agent.algorithm import math_util, policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch
import pydash as ps

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
        "continuous_action_clip": 2.0,
        "training_frequency": 1
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
            'continuous_action_clip',
            'training_frequency',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.action_policy_update = getattr(policy_util, self.action_policy_update)
        for body in self.agent.nanflat_body_a:
            body.explore_var = self.explore_var_start

    @lab_api
    def init_nets(self):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPdefault'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''
        in_dim = self.body.state_dim
        if self.body.is_discrete:
            out_dim = self.body.action_dim
            if self.net_spec['type'] == 'MLPdefault':
                self.net_spec['type'] = 'MLPNet'
        else:
            out_dim = [self.body.action_dim, self.body.action_dim]
            if self.net_spec['type'] == 'MLPdefault':
                self.net_spec['type'] = 'MLPHeterogenousTails'
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim)
        self.net_names = ['net']
        self.post_init_nets()

    @lab_api
    def calc_pdparam(self, x, evaluate=True):
        '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        if evaluate:
            pdparam = self.net.wrap_eval(x)
        else:
            self.net.train()
            pdparam = self.net(x)
        return pdparam

    @lab_api
    def body_act(self, body, state):
        action, action_pd = self.action_policy(state, self, body)
        body.entropies.append(action_pd.entropy())
        body.log_probs.append(action_pd.log_prob(action.float()))
        if len(action.shape) == 0:  # scalar
            return action.cpu().numpy().astype(body.action_space.dtype).item()
        else:
            return action.cpu().numpy()

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        batch = util.concat_batches(batches)
        batch = util.to_torch_batch(batch, self.net.gpu)
        return batch

    @lab_api
    def train(self):
        if util.get_lab_mode() == 'enjoy':
            return np.nan
        if self.to_train == 1:
            batch = self.sample()
            loss = self.calc_policy_loss(batch)
            self.net.training_step(loss=loss)
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Policy loss: {loss}')
            self.last_loss = loss.item()
        return self.last_loss

    def calc_policy_loss(self, batch):
        '''Calculate the policy loss for a batch of data.'''
        # use simple returns as advs
        advs = math_util.calc_returns(batch, self.gamma)
        # advantage standardization trick
        # guard nan std by setting to 0 and add small const
        adv_std = advs.std()
        adv_std[adv_std != adv_std] = 0
        adv_std += 1e-08
        advs = (advs - advs.mean()) / adv_std
        assert len(self.body.log_probs) == len(advs), f'{len(self.body.log_probs)} vs {len(advs)}'
        log_probs = torch.stack(self.body.log_probs)
        policy_loss = - log_probs * advs
        if self.add_entropy:
            entropies = torch.stack(self.body.entropies)
            policy_loss += (-self.entropy_coef * entropies)
        policy_loss = torch.sum(policy_loss)
        if torch.cuda.is_available() and self.net.gpu:
            policy_loss = policy_loss.cuda()
        return policy_loss

    @lab_api
    def update(self):
        space_clock = util.s_get(self, 'aeb_space.clock')
        for net in [self.net]:
            net.update_lr(space_clock)
        explore_vars = [self.action_policy_update(self, body) for body in self.agent.nanflat_body_a]
        explore_var_a = self.nanflat_to_data_a('explore_var', explore_vars)
        return explore_var_a
