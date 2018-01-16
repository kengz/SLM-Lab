from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns, act_update_fns
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import numpy as np
import torch
import pydash as _


class ReinforceDiscrete(Algorithm):
    '''
    Implementation of REINFORCE (Williams, 1992) with baseline for discrete actions http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    Algorithm:
        1. At each timestep in an episode
            - Calculate the advantage of that timestep
            - Multiply the advantage by the negative of log probability of the action taken
        2. Sum all the values above.
        3. Calculate the gradient of this value with respect to all of the parameters of the network
        4. Update the network parameters using the gradient
    '''

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.init_nets()
        self.init_algo_params()
        self.net.print_nets()  # Print the network architecture
        logger.info(util.self_desc(self))

    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        body = self.agent.flat_nonan_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        net_spec = self.agent.spec['net']
        net_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))
        self.net = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim, **net_kwargs)

    def init_algo_params(self):
        '''Initialize other algorithm parameters'''
        algorithm_spec = self.agent.spec['algorithm']
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        util.set_attr(self, _.pick(algorithm_spec, [
            'gamma',
            'num_epis_to_collect',
        ]))
        # To save on a forward pass keep the log probs from each action
        self.saved_log_probs = []
        self.to_train = 0
        self.flat_nonan_explore_var_a = [
            np.nan] * len(self.agent.flat_nonan_body_a)

    @lab_api
    def body_act_discrete(self, body, state):
        return self.action_policy(self, state, self.net)

    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample()
                   for body in self.agent.flat_nonan_body_a]
        batch = util.concat_dict(batches)
        batch = util.to_torch_nested_batch(batch)
        return batch

    @lab_api
    def train(self):
        if self.to_train == 1:
            # We only care about the rewards from the batch
            rewards = self.sample()['rewards']
            advantage = self.calculate_advantage(rewards)
            logger.debug(f'Length first epi: {len(rewards[0])}')
            logger.debug(f'Len log probs: {len(self.saved_log_probs)}')
            logger.debug(f'Len advantage: {advantage.size(0)}')
            if len(self.saved_log_probs) != advantage.size(0):
                # Caused by first reward of episode being nan
                del self.saved_log_probs[0]
                logger.debug('Deleting first log prob in epi')
            assert len(self.saved_log_probs) == advantage.size(0)
            policy_loss = []
            for log_prob, a in zip(self.saved_log_probs, advantage):
                logger.debug(f'log prob: {log_prob.data[0]}, advantage: {a}')
                policy_loss.append(-log_prob * a)
            self.net.optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            loss = policy_loss.data[0]
            policy_loss.backward()
            if self.net.clamp_grad:
                logger.info("Clipping gradient...")
                torch.nn.utils.clip_grad_norm(
                    self.net.parameters(), self.net.clamp_grad_val)
            logger.debug(f'Gradient norms: {self.net.get_grad_norms()}')
            self.net.optim.step()
            self.to_train = 0
            self.saved_log_probs = []
            logger.debug(f'Policy loss: {loss}')
            return loss
        else:
            return None

    def calculate_advantage(self, raw_rewards):
        advantage = []
        logger.debug(f'Raw rewards: {raw_rewards}')
        for epi_rewards in raw_rewards:
            rewards = []
            R = 0
            for r in epi_rewards[::-1]:
                R = r + self.gamma * R
                rewards.insert(0, R)
            rewards = torch.Tensor(rewards)
            logger.debug(f'Rewards: {rewards}')
            rewards = (rewards - rewards.mean()) / \
                (rewards.std() + np.finfo(np.float32).eps)
            logger.debug(f'Normalized rewards: {rewards}')
            advantage.append(rewards)
        advantage = torch.cat(advantage)
        return advantage

    @lab_api
    def update(self):
        '''No update needed'''
        return self.flat_nonan_explore_var_a
