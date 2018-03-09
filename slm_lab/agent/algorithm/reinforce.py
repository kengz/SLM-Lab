from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns, act_update_fns, decay_learning_rate
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import numpy as np
import torch
import pydash as _


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
        self.init_nets()
        self.init_algo_params()
        logger.info(util.self_desc(self))

    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        body = self.agent.nanflat_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        self.is_discrete = body.is_discrete
        net_spec = self.agent.spec['net']
        mem_spec = self.agent.spec['memory']
        net_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
            gpu=_.get(net_spec, 'gpu'),
        ))
        # Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPdefault'. Otherwise the correct type of network is assumed to be specified in the spec.
        # Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        # Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        if net_spec['type'] == 'MLPdefault':
            if self.is_discrete:
                self.net = getattr(net, 'MLPNet')(
                    state_dim, net_spec['hid_layers'], action_dim, **net_kwargs)
            else:
                self.net = getattr(net, 'MLPHeterogenousHeads')(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim], **net_kwargs)
        # If net is recurrent we need to include the length of the sequence to be passed to the recurrent part
        elif net_spec['type'] == 'RecurrentNet':
            if self.is_discrete:
                self.net = getattr(net, net_spec['type'])(
                    state_dim, net_spec['hid_layers'], action_dim, mem_spec['length_history'], **net_kwargs)
            else:
                self.net = getattr(net, net_spec['type'])(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim], mem_spec['length_history'], **net_kwargs)
        else:
            if self.is_discrete:
                self.net = getattr(net, net_spec['type'])(
                    state_dim, net_spec['hid_layers'], action_dim, **net_kwargs)
            else:
                self.net = getattr(net, net_spec['type'])(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim], **net_kwargs)

    def init_algo_params(self):
        '''Initialize other algorithm parameters'''
        algorithm_spec = self.agent.spec['algorithm']
        net_spec = self.agent.spec['net']
        # Automatically selects appropriate discrete or continuous action policy if setting is default
        action_fn = algorithm_spec['action_policy']
        if action_fn == 'default':
            if self.is_discrete:
                self.action_policy = act_fns['softmax']
            else:
                self.action_policy = act_fns['gaussian']
        else:
            self.action_policy = act_fns[action_fn]
        util.set_attr(self, _.pick(algorithm_spec, [
            'gamma',
            'num_epis_to_collect',
            'add_entropy', 'entropy_weight',
            'continuous_action_clip'
        ]))
        util.set_attr(self, _.pick(net_spec, [
            'decay_lr', 'decay_lr_frequency', 'decay_lr_min_timestep', 'gpu'
        ]))
        if not hasattr(self, 'gpu'):
            self.gpu = False
        logger.info(f'Training on gpu: {self.gpu}')
        # To save on a forward pass keep the log probs from each action
        self.saved_log_probs = []
        self.entropy = []
        self.to_train = 0

    @lab_api
    def body_act_discrete(self, body, state):
        return self.action_policy(self, state, body, self.gpu)

    @lab_api
    def body_act_continuous(self, body, state):
        return self.action_policy(self, state, body, self.gpu)

    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample()
                   for body in self.agent.nanflat_body_a]
        batch = util.concat_dict(batches)
        batch = util.to_torch_nested_batch_ex_rewards(batch, self.gpu)
        return batch

    @lab_api
    def train(self):
        if self.to_train == 1:
            logger.debug2(f'Training...')
            # We only care about the rewards from the batch
            rewards = self.sample()['rewards']
            logger.debug3(f'Length first epi: {len(rewards[0])}')
            logger.debug3(f'Len log probs: {len(self.saved_log_probs)}')
            self.net.optim.zero_grad()
            policy_loss = self.get_policy_loss(rewards)
            loss = policy_loss.data[0]
            policy_loss.backward()
            if self.net.clamp_grad:
                logger.debug("Clipping gradient...")
                torch.nn.utils.clip_grad_norm(
                    self.net.parameters(), self.net.clamp_grad_val)
            logger.debug2(f'Gradient norms: {self.net.get_grad_norms()}')
            self.net.optim.step()
            self.to_train = 0
            self.saved_log_probs = []
            self.entropy = []
            logger.debug(f'Policy loss: {loss}')
            return loss
        else:
            return np.nan

    def get_policy_loss(self, batch):
        '''Returns the policy loss for a batch of data.
        For REINFORCE just rewards are passed in as the batch'''
        advantage = self.calc_advantage(batch)
        advantage = self.check_sizes(advantage)
        policy_loss = []
        for log_prob, a, e in zip(self.saved_log_probs, advantage, self.entropy):
            logger.debug3(
                f'log prob: {log_prob.data[0]}, advantage: {a}, entropy: {e.data[0]}')
            if self.add_entropy:
                policy_loss.append(-log_prob * a - self.entropy_weight * e)
            else:
                policy_loss.append(-log_prob * a)
        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss

    def check_sizes(self, advantage):
        '''Checks that log probs, advantage, and entropy all have the same size
           Occassionally they do not, this is caused by first reward of an episode being nan. If they are not the same size, the function removes the elements of the log probs and entropy that correspond to nan rewards.'''
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
            rewards = []
            big_r = 0
            for r in epi_rewards[::-1]:
                big_r = r + self.gamma * big_r
                rewards.insert(0, big_r)
            rewards = torch.Tensor(rewards)
            logger.debug3(f'Rewards: {rewards}')
            rewards = (rewards - rewards.mean()) / (
                rewards.std() + np.finfo(np.float32).eps)
            logger.debug3(f'Normalized rewards: {rewards}')
            advantage.append(rewards)
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

    def get_actor_output(self, x, evaluate=True):
        '''Returns the output of the policy, regardless of the underlying network structure. This makes it easier to handle AC algorithms with shared or distinct params.
           Output will either be the logits for a categorical probability distribution over discrete actions (discrete action space) or the mean and std dev of the action policy (continuous action space)
        '''
        if evaluate:
            out = self.net.wrap_eval(x)
        else:
            self.net.train()
            out = self.net(x)
        return out
