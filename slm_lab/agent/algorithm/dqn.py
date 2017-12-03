import torch
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.algorithm.algorithm_utils import act_fns, update_fns
from slm_lab.agent.net import nets
from slm_lab.agent.memory import Replay


class DQNBase(Algorithm):
    '''
    Implementation of the base DQN algorithm.
    See Sergey Levine's lecture xxx for more details
    TODO add link
          more detailed comments

    net: instance of an slm_lab/agent/net
    memory: instance of an slm_lab/agent/memory
    batch_size: how many examples from memory to sample at each training step
    action_selection: function (from common.py) that determines how to select actions
    gamma: Real number in range [0, 1]. Determines how much to discount the future
    state_dim: dimension of the state space
    action_dim: dimensions of the action space
    '''

    def __init__(self, agent):
        super(DQNBase, self).__init__(agent)

    def post_body_init(self):
        '''
        Initializes all of the memory parameters to a blank memory
        Can also be used to clear the memory
        '''
        # TODO generalize
        default_body = self.agent.bodies[0]
        state_dim = default_body.state_dim
        action_dim = default_body.action_dim
        net_spec = self.agent.spec['net']
        net_spec['net_layer_params'][0] = state_dim
        net_spec['net_layer_params'][-1] = action_dim
        # TODO pull out net init from algo if possible
        self.net = nets[net_spec['net_type']](
            *net_spec['net_layer_params'],
            *net_spec['net_other_params'])
        # TODO three nets for different part of Q function
        # In base algorithm should all be pointer to the same net - then update compute q target values and action functions
        self.batch_size = net_spec['batch_size']
        self.gamma = net_spec['gamma']

        algorithm_spec = self.agent.spec['algorithm']
        self.action_selection = act_fns[algorithm_spec['action_selection']]

        # explore_var is epsilon, tau or etc.
        self.explore_var_start = algorithm_spec['explore_var_start']
        self.explore_var_end = algorithm_spec['explore_var_end']
        self.explore_var = self.explore_var_start
        self.decay_steps = algorithm_spec['decay_steps']
        self.training_iters_per_batch = 1
        self.training_frequency = 1

    def train_a_batch(self):
        # TODO Fix for training iters, docstring
        t = self.agent.agent_space.aeb_space.clock['t']
        if t % self.training_frequency == 0:
            batch = self.agent.memory.get_batch(self.batch_size)
            for i in range(self.training_iters_per_batch):
                q_targets = self.compute_q_target_values(batch)
                loss = self.net.training_step(batch['states'], q_targets)
            return loss
        else:
            return None

    def compute_q_target_values(self, batch):
        q_vals = self.net.eval(batch['states'])
        # Make future reward 0 if the current state is done
        q_targets_all = batch['rewards'] + self.gamma * \
            (1 - batch['dones']) * self.net.eval(batch['next_states'])
        q_targets_max = torch.max(q_targets_all, axis=1)
        # Reshape q_targets_max to q_targets_all shape
        q_targets_max = q_targets_max.expand(-1, q_targets_all.shape[1])
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_vals
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_max, batch['actions']) + \
            torch.mul(q_vals, (1 - batch['actions']))
        return q_targets

    def body_act_discrete(self, body, body_state):
        return self.action_selection(
            self.net,
            body_state,
            self.explore_var)

    def update(self, action, reward, state, done):
        # TODO make proper
        # Update epsilon or boltzmann
        self.anneal_epi = 20
        epi = self.agent.agent_space.aeb_space.clock['e']
        rise = self.explore_var_end - self.explore_var_start
        slope = rise / float(self.anneal_epi)
        self.explore_var = max(
            slope * epi + self.explore_var_start, self.explore_var_end)
