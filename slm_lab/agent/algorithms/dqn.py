import torch
from slm_lab.agent.algorithm.common import *

class DQN:
    '''
    Implementation of the DQN algorithm.
    See Playing Atari with Deep Reinforcement Learning for more info
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

    net: instance of an slm_lab/agent/net
    memory: instance of an slm_lab/agent/memory
    batch_size: how many examples from memory to sample at each training step
    action_selection: function (from common.py) that determines how to select
                      actions
    gamma: Real number in range [0, 1]. Determines how much to discount the
           future
    '''

    # TODO: Change init to something like
    # def __init__(self, spec):
    #     self.net = spec['NetType'](*spec.net.params)

    def __init__(self,
                 net,
                 memory,
                 batch_size=32,
                 action_selection=select_action_epsilon_greedy,
                 gamma=0.98):
        super(DQN,self).__init__()
        self.net = net
        self.memory = memory
        self.batch_size = batch_size
        self.action_selection = action_selection
        self.gamma = gamma

    def train_a_batch(self):
        batch = self.memory.get_batch(self.batch_size)
        q_targets = self.compute_q_target_values(batch)
        loss = self.net.training_step(batch['states'], q_targets)
        return loss

    def compute_q_target_values(self, batch):
        q_vals = self.net.eval(batch['states'])
        # Make future reward 0 if the current state is terminal
        q_targets_all = batch['rewards'] + self.gamma * \
            (1 - batch['terminals']) * self.net.eval(batch['next_states'])
        q_targets_max = torch.max(q_targets_all, axis=1)
        # Reshape q_targets_max to q_targets all shape
        q_targets_max = q_targets_max.expand(-1, q_targets_all.shape[1])
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_vals
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_max, batch['actions']) + \
                    torch.mul(q_vals, (1 - batch['actions']))
        return q_targets

    def select_action(self, state, epsilon_or_tau):
        return self.action_selection(self.net, state, epsilon_or_tau)

    def update(self):
        # TODO:
        pass
