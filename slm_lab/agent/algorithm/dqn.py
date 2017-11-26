import torch
from slm_lab.agent.algorithm.algorithm_utils import act_fns, update_fns
from slm_lab.agent.net import nets
from slm_lab.agent.memory.base_memory import ReplayMemory

class DQNBase:
    '''
    Implementation of the base DQN algorithm.
    See Sergey Levine's lecture xxx for more details
    TODO: add link
          more detailed comments

    net: instance of an slm_lab/agent/net
    memory: instance of an slm_lab/agent/memory
    batch_size: how many examples from memory to sample at each training step
    action_selection: function (from common.py) that determines how to select
                      actions
    gamma: Real number in range [0, 1]. Determines how much to discount the
           future
    state_dim: dimension of the state space
    action_dim: dimensions of the action space
    '''

    def __init__(self, spec, state_dim, action_dim):
        super(DQNBase,self).__init__()
        spec['net_layer_params'][0] = state_dim
        spec['net_layer_params'][-1] = action_dim
        self.net = nets[spec['net_type']](
            *spec['net_layer_params'],
            *spec['net_other_params'])
        # TODO: three nets for different part of Q function
        # In base algorithm should all be pointer to the same
        # net - then update compute q target values and action
        # functions
        self.memory = ReplayMemory(
            spec['memory_size'],
            state_dim,
            action_dim)
        self.batch_size = spec['batch_size']
        self.action_selection = act_fns[spec['action_selection']]
        self.gamma = spec['gamma']
        self.epsilon_tau_start = spec['epsilon_tau_start']
        self.epsilon_tau_end = spec['epsilon_tau_end']
        self.epsilon_or_tau = self.epsilon_tau_start
        self.decay_steps = spec['decay_steps']
        self.training_iters_per_batch = 1

    def train_a_batch(self):
        # TODO: Fix for training iters
        batch = self.memory.get_batch(self.batch_size)
        for i in range(self.training_iters_per_batch):
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

    def select_action(self, state):
        return self.action_selection(
                    self.net,
                    state,
                    self.epsilon_or_tau)

    def update(self):
        # TODO:
        # Update epsilon or boltzmann
        pass
