import numpy as np
import torch
from torch.autograd import Variable
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.algorithm.algorithm_util import act_fns, update_fns
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
        '''Initializes the part of algorithm needing a body to exist first.'''
        # TODO generalize
        default_body = self.agent.bodies[0]
        state_dim = default_body.state_dim
        action_dim = default_body.action_dim
        net_spec = self.agent.spec['net']
        net_spec['net_layer_params'][0] = state_dim
        net_spec['net_layer_params'][-1] = action_dim
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
        self.explore_anneal_epi = algorithm_spec['explore_anneal_epi']
        self.training_iters_per_batch = 1
        self.training_frequency = 1

    def compute_q_target_values(self, batch):
        # Make future reward 0 if the current state is done
        float_data_list = [
            'states', 'actions', 'rewards', 'dones', 'next_states']
        for k in float_data_list:
            batch[k] = Variable(torch.from_numpy(batch[k]).float())
        # print('batch')
        # print(batch['states'])
        # print(batch['actions'])
        # print(batch['rewards'])
        # print(batch['dones'])
        # print(1 - batch['dones'])
        q_vals = self.net.wrap_eval(batch['states'])
        # print(f'q_vals {q_vals}')
        q_targets_all = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data),
                      self.net.wrap_eval(batch['next_states']))
        # print(f'q_targets_all {q_targets_all}')
        q_targets_max, _idx = torch.max(q_targets_all, dim=1)
        # print(f'q_targets_max {q_targets_max}')
        # print(f'q_targets_all size {q_targets_all.size()}')

        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_vals
        # So that the loss for these actions is 0
        q_targets_max.unsqueeze_(1)
        # print(f'q_targets_max {q_targets_max}')
        q_targets = torch.mul(q_targets_max, batch['actions'].data) + \
            torch.mul(q_vals, (1 - batch['actions'].data))
        # print(f'q_targets {q_targets}')
        return q_targets

    def train(self):
        # TODO Fix for training iters, docstring
        t = self.agent.agent_space.aeb_space.clock['t']
        if t % self.training_frequency == 0:
            batch = self.agent.memory.get_batch(self.batch_size)
            for i in range(self.training_iters_per_batch):
                q_targets = self.compute_q_target_values(batch)
                y = Variable(q_targets)
                loss = self.net.training_step(batch['states'], y)
                # print(f'loss {loss.data[0]}\n')
            return loss.data[0]
        else:
            return None

    def body_act_discrete(self, body, body_state):
        # TODO can handle identical bodies now; to use body_net for specific body.
        return self.action_selection(
            self.net,
            body_state,
            self.explore_var)

    def update(self):
        '''Update epsilon or boltzmann for policy after net training'''
        epi = self.agent.agent_space.aeb_space.clock['e']
        rise = self.explore_var_end - self.explore_var_start
        slope = rise / float(self.explore_anneal_epi)
        self.explore_var = max(
            slope * epi + self.explore_var_start, self.explore_var_end)
        return self.explore_var
