from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from torch.autograd import Variable
import numpy as np
import torch
import pydash as _


class ReinforceDiscrete(Algorithm):
    '''
    TODO
    Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    '''

    def __init__(self, agent):
        super(ReinforceDiscrete, self).__init__(agent)
        self.agent = agent

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        body = self.agent.flat_nonan_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        net_spec = self.agent.spec['net']
        self.net = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
            clamp_grad=_.get(net_spec, 'clamp_grad'),
        )

        algorithm_spec = self.agent.spec['algorithm']
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        self.num_epi_before_training = algorithm_spec['num_epis_before_training']
        self.gamma = algorithm_spec['gamma']
        # To save on a forward pass keep the log probs from each action
        self.saved_log_probs = []
        self.to_train = 0

    def body_act_discrete(self, body, body_state):
        return self.action_policy(self.agent, body, body_state, self.net)

    def train(self):
        if self.to_train == 1:
            # Only care about the rewards
            rewards = self.agent.memory.get_batch()['rewards']
            logger.debug('Length first epi: {len(rewards[0])}')
            advantage = self.calculate_advantage(rewards)
            assert len(self.saved_log_probs) == advantage.size(0)
            policy_loss = []
            for log_prob, a in zip(self.saved_log_probs, advantage):
                policy_loss.append(-log_prob * a)
            self.net.optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            loss = policy_loss.data[0]
            logger.debug('Policy loss: {loss}')
            policy_loss.backward()
            self.net.optim.step()
            self.to_train = 0
            self.saved_log_probs = []
            return loss
        else:
            return None

    def calculate_advantage(self, batch):
        advantage = []
        for epi in batch:
            rewards = []
            R = 0
            for r in epi[::-1]:
                R = r + self.gamma * R
                rewards.insert(0, R)
            rewards = torch.Tensor(rewards)
            # rewards = (rewards - rewards.mean())
            rewards = (rewards - rewards.mean()) / \
                (rewards.std() + np.finfo(np.float32).eps)
            advantage.append(rewards)
        advantage = torch.cat(advantage)
        return advantage

    def update(self):
        '''No update needed'''
        explore_var = 1.
        return explore_var
