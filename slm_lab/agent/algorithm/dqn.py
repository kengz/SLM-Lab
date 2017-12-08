import numpy as np
import pydash as _
import sys
import torch
from copy import deepcopy
from slm_lab.agent.algorithm.algorithm_util import act_fns, update_fns
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.memory import Replay
from slm_lab.agent.net import nets
from slm_lab.agent.net.common import *
from torch.autograd import Variable


class DQNBase(Algorithm):
    '''
    Implementation of the base DQN algorithm.
    See Sergey Levine's lecture xxx for more details
    TODO add link
          more detailed comments

    net: instance of an slm_lab/agent/net
    memory: instance of an slm_lab/agent/memory
    batch_size: how many examples from memory to sample at each training step
    action_policy: function (from common.py) that determines how to select actions
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
        net_spec['net_layers'][0] = state_dim
        net_spec['net_layers'][-1] = action_dim
        # TODO set optimizer choice, loss_fn, from leftover net_spec
        self.net = nets[net_spec['net_type']](
            *net_spec['net_layers'],
            optim_param=_.get(net_spec, 'optim_param'))
        print(self.net)
        self.target_net = nets[net_spec['net_type']](
            *net_spec['net_layers'],
            optim_param=_.get(net_spec, 'optim_param'))
        self.act_select_net = self.net
        self.eval_net = self.net
        self.batch_size = net_spec['batch_size']

        # TODO adjust learning rate http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
        # TODO hackish optimizer learning rate, also it fails for SGD wtf

        algorithm_spec = self.agent.spec['algorithm']
        # TODO use module get attr for this instead of dict
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        self.gamma = algorithm_spec['gamma']

        # explore_var is epsilon, tau or etc.
        self.explore_var_start = algorithm_spec['explore_var_start']
        self.explore_var_end = algorithm_spec['explore_var_end']
        self.explore_var = self.explore_var_start
        self.explore_anneal_epi = algorithm_spec['explore_anneal_epi']

        self.training_min_timestep = algorithm_spec['training_min_timestep']
        self.training_frequency = algorithm_spec['training_frequency']
        self.training_epoch = algorithm_spec['training_epoch']
        self.training_iters_per_batch = algorithm_spec['training_iters_per_batch']

        # Network update params
        self.update_type = 'replace'
        self.update_frequency = 1
        self.polyak_weight = 0.9

    def compute_q_target_values(self, batch):
        q_vals = self.net.wrap_eval(batch['states'])
        # Use act_select network to select actions in next state
        # Depending on the algorithm this is either the current
        # net or target net
        q_next_st_act_vals = self.act_select_net.wrap_eval(
            batch['next_states'])
        _val, q_next_actions = torch.max(q_next_st_act_vals, dim=1)
        # Select q_next_st_vals_max based on action selected in q_next_actions
        # Evaluate the action selection using the eval net
        # Depending on the algorithm this is either the current
        # net or target net
        q_next_st_vals = self.eval_net.wrap_eval(batch['next_states'])
        idx = torch.from_numpy(np.array(list(range(self.batch_size))))
        q_next_st_vals_max = q_next_st_vals[idx, q_next_actions]
        q_next_st_vals_max.unsqueeze_(1)
        # Compute final q_target using reward and estimated
        # best Q value from the next state if there is one
        # Make future reward 0 if the current state is done
        q_targets_max = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), q_next_st_vals_max)
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_vals
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_max, batch['actions'].data) + \
            torch.mul(q_vals, (1 - batch['actions'].data))
        return q_targets

    def train(self):
        # TODO docstring
        t = self.agent.agent_space.aeb_space.clock['total_t']
        if (t > self.training_min_timestep and t % self.training_frequency == 0):
            # print('Training')
            total_loss = 0.0
            for _b in range(self.training_epoch):
                batch = self.agent.memory.get_batch(self.batch_size)
                batch_loss = 0.0
                '''Package data into pytorch variables'''
                float_data_list = [
                    'states', 'actions', 'rewards', 'dones', 'next_states']
                for k in float_data_list:
                    batch[k] = Variable(torch.from_numpy(batch[k]).float())

                for _i in range(self.training_iters_per_batch):
                    q_targets = self.compute_q_target_values(batch)
                    y = Variable(q_targets)
                    loss = self.net.training_step(batch['states'], y)
                    batch_loss += loss.data[0]
                    # print(f'loss {loss.data[0]}')
                batch_loss /= self.training_iters_per_batch
                # print(f'batch_loss {batch_loss}')
                total_loss += batch_loss
            # print(f'total_loss {total_loss}')
            return total_loss
        else:
            # print('NOT training')
            return None

    def body_act_discrete(self, body, body_state):
        # TODO can handle identical bodies now; to use body_net for specific body.
        return self.action_policy(
            self.net,
            body_state,
            self.explore_var)

    def update(self):
        t = self.agent.agent_space.aeb_space.clock['total_t']
        if t % 100 == 0:
            print(f'Total time step: {t}')
        '''Update epsilon or boltzmann for policy after net training'''
        epi = self.agent.agent_space.aeb_space.clock['e']
        rise = self.explore_var_end - self.explore_var_start
        slope = rise / float(self.explore_anneal_epi)
        self.explore_var = max(
            slope * (epi - 1) + self.explore_var_start, self.explore_var_end)
        # print(f'Explore var: {self.explore_var}')

        '''Update target net with current net'''
        if self.update_type == 'replace':
            if t % self.update_frequency == 0:
                # print('Updating net by replacing')
                self.target_net = deepcopy(self.net)
        elif self.update_type == 'polyak':
            # print('Updating net by averaging')
            avg_params = self.polyak_weight * flatten_params(self.target_net) + \
                (1 - self.polyak_weight) * flatten_params(self.net)
            self.target_net = load_params(self.target_net, avg_params)
        else:
            print('Unknown network update type.')
            print('Should be "replace" or "polyak". Exiting ...')
            sys.exit()
        return self.explore_var


class DQN(DQNBase):
    # TODO: Check this is working
    def __init__(self, agent):
        super(DQN, self).__init__(agent)

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        super(DQN, self).post_body_init()
        self.act_select_net = self.target_net
        self.eval_net = self.target_net
        # Network update params
        net_spec = self.agent.spec['net']
        self.update_type = net_spec['network_update_type']
        self.update_frequency = net_spec['network_update_frequency']
        self.polyak_weight = net_spec['network_update_weight']
        print(
            f'Network update: type: {self.update_type}, frequency: {self.update_frequency}, weight: {self.polyak_weight}')

    def update(self):
        super(DQN, self).update()
        self.act_select_net = self.target_net
        self.eval_net = self.target_net


class DoubleDQN(DQNBase):
    # TODO: Check this is working
    def __init__(self, agent):
        super(DoubleDQN, self).__init__(agent)

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        super(DoubleDQN, self).post_body_init()
        self.act_select_net = self.net
        self.eval_net = self.target_net
        # Network update params
        net_spec = self.agent.spec['net']
        self.update_type = net_spec['network_update_type']
        self.update_frequency = net_spec['network_update_frequency']
        self.polyak_weight = net_spec['network_update_weight']

    def update(self):
        super(DoubleDQN, self).update()
        self.act_select_net = self.net
        self.eval_net = self.target_net
