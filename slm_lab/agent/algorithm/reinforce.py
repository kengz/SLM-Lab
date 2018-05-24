from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.algorithm_util import act_fns, act_update_fns, decay_learning_rate
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
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
        util.set_attr(self, self.algorithm_spec, [
            'action_policy',
            # theoretically, REINFORCE does not have policy update; but in this implementation we have such option
            'action_policy_update',
            'explore_var_start', 'explore_var_end', 'explore_anneal_epi',
            'gamma',  # the discount factor
            'add_entropy',
            'entropy_weight',
            'continuous_action_clip',
            'training_frequency',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        # TODO bring policy_update into policy_util
        self.action_policy_update = act_update_fns[self.action_policy_update]
        for body in self.agent.nanflat_body_a:
            body.explore_var = self.explore_var_start
        self.entropy = []
        # To save on a forward pass keep the log probs from each action
        self.saved_log_probs = []

    @lab_api
    def init_nets(self):
        '''
        Initialize the neural network used to learn the Q function from the spec
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
        self.net = NetClass(self.net_spec, self, in_dim, out_dim)
        logger.info(f'Training on gpu: {self.net.gpu}')

    @lab_api
    def body_act(self, body, state):
        action, action_pd = self.action_policy(state, self, body)
        # TODO update logprob if has
        return action

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        # TODO if just want raw rewards, skip conversion to torch batch
        batch = util.concat_dict(batches)
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
            self.saved_log_probs = []
            self.entropy = []
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
        for log_prob, a, e in zip(self.saved_log_probs, advantage, self.entropy):
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
        body = self.agent.nanflat_body_a[0]
        nan_idxs = body.memory.last_nan_idxs
        num_nans = sum(nan_idxs)
        assert len(nan_idxs) == len(self.saved_log_probs)
        assert len(nan_idxs) == len(self.entropy)
        assert len(nan_idxs) - num_nans == advantage.size(0)
        logger.debug2(f'{num_nans} nans encountered when gathering data')
        if num_nans != 0:
            idxs = [x for x in range(len(nan_idxs)) if nan_idxs[x] == 1]
            logger.debug3(f'Nan indexes: {idxs}')
            for idx in idxs[::-1]:
                del self.saved_log_probs[idx]
                del self.entropy[idx]
        assert len(self.saved_log_probs) == advantage.size(0)
        assert len(self.entropy) == advantage.size(0)
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

    def update_learning_rate(self):
        decay_learning_rate(self, [self.net])

    @lab_api
    def update(self):
        self.update_learning_rate()
        '''No update needed to explore var'''
        explore_var = np.nan
        return explore_var

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
