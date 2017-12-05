import sys
import numpy as np
import torch
import copy
from torch.autograd import Variable
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.algorithm.algorithm_util import act_fns, update_fns
from slm_lab.agent.net import nets
from slm_lab.agent.net.common import *
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
        # TODO expose optim and other params of net to interface
        self.net = nets[net_spec['net_type']](
            *net_spec['net_layer_params'],
            *net_spec['net_other_params'],
            net_spec['lr'])
        self.target_net = nets[net_spec['net_type']](
            *net_spec['net_layer_params'],
            *net_spec['net_other_params'],
            net_spec['lr'])
        self.act_select_net = self.net
        self.eval_net = self.net
        self.batch_size = net_spec['batch_size']
        self.gamma = net_spec['gamma']
            *net_spec['net_other_params'])

        # TODO adjust learning rate http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
        # TODO hackish optimizer learning rate, also it fails for SGD wtf
        for param_group in self.net.optim.param_groups:
            param_group['lr']=net_spec['lr']
        # TODO three nets for different part of Q function
        # In base algorithm should all be pointer to the same net - then update compute q target values and action functions
        # TODO other params from spec did not get used yet

        algorithm_spec=self.agent.spec['algorithm']
        # TODO use module get attr for this instead of dict
        self.action_selection=act_fns[algorithm_spec['action_selection']]
        self.gamma=algorithm_spec['gamma']

        # explore_var is epsilon, tau or etc.
        self.explore_var_start=algorithm_spec['explore_var_start']
        self.explore_var_end=algorithm_spec['explore_var_end']
        self.explore_var=self.explore_var_start
        self.explore_anneal_epi=algorithm_spec['explore_anneal_epi']

        self.initial_data_gather_steps=algorithm_spec['initial_data_gather_steps']
        self.training_iters_per_batch=algorithm_spec['training_iters_per_batch']
        self.training_frequency=algorithm_spec['training_frequency']

        # Network update params
        self.update_type="replace"
        self.update_frequency=1
        self.polyak_weight=0.9

    def compute_q_target_values(self, batch):
        # Make future reward 0 if the current state is done
        q_vals=self.net.wrap_eval(batch['states'])
        # Use act_select network to select actions in next state
        # Depending on the algorithm this is either the current
        # net or target net
        q_next_st_act_vals=self.act_select_net.wrap_eval(batch['next_states'])
        _, q_next_actions=torch.max(q_next_st_act_vals, dim = 1)
        # Select q_next_st_vals_max based on action selected in q_next_actions
        # Evaluate the action selection using the eval net
        # Depending on the algorithm this is either the current
        # net or target net
        q_next_st_vals=self.eval_net.wrap_eval(batch['next_states'])
        idx=torch.from_numpy(np.array(list(range(self.batch_size))))
        q_next_st_vals_max=q_next_st_vals[idx, q_next_actions]
        q_next_st_vals_max.unsqueeze_(1)
        # Compute final q_target using reward and estimated
        # best Q value from the next state if there is one
        # Make future reward 0 if the current state is done
        q_targets_max=batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), q_next_st_vals_max)
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_vals
        # So that the loss for these actions is 0
        q_targets=torch.mul(q_targets_max, batch['actions'].data) + \
            torch.mul(q_vals, (1 - batch['actions'].data))
        return q_targets

    def train(self):
        # TODO docstring
        t=self.agent.agent_space.aeb_space.clock['total_t']
        if t % self.training_frequency == 0 and \
            t > self.initial_data_gather_steps:
            # print("Training")
            batch=self.agent.memory.get_batch(self.batch_size)

            ''' Package data into pytorch variables '''
            float_data_list=[
                'states', 'actions', 'rewards', 'dones', 'next_states']
            for k in float_data_list:
                batch[k] = Variable(torch.from_numpy(batch[k]).float())

            for i in range(self.training_iters_per_batch):
                q_targets = self.compute_q_target_values(batch)
                y = Variable(q_targets)
                loss = self.net.training_step(batch['states'], y)
                # print(f'loss {loss.data[0]}')
            return loss.data[0]
        else:
            # print("NOT training")
            return None

    def body_act_discrete(self, body, body_state):
        # TODO can handle identical bodies now; to use body_net for specific body.
        return self.action_selection(
            self.net,
            body_state,
            self.explore_var)

    def update(self):
        t=self.agent.agent_space.aeb_space.clock['total_t']
        if t % 100 == 0:
            print("Total time step: {}".format(t))
        '''Update epsilon or boltzmann for policy after net training'''
        epi=self.agent.agent_space.aeb_space.clock['e']
        rise=self.explore_var_end - self.explore_var_start
        slope=rise / float(self.explore_anneal_epi)
        self.explore_var=max(
            slope * (epi - 1) + self.explore_var_start, self.explore_var_end)
        # print("Explore var: {}".format(self.explore_var))

        '''Update target net with current net'''
        if self.update_type == "replace":
            if t % self.update_frequency == 0:
                print("Updating net by replacing")
                self.target_net=copy.deepcopy(self.net)
        elif self.update_type == "polyak":
            # print("Updating net by averaging")
            avg_params=self.polyak_weight * flatten_params(self.target_net) + \
                         (1 - self.polyak_weight) * flatten_params(self.net)
            self.target_net=load_params(self.target_net, avg_params)
        else:
            print("Unknown network update type.")
            print("Should be 'replace' or 'polyak'. Exiting ...")
            sys.exit()
        return self.explore_var


class DQN(DQNBase):

    def __init__(self, agent):
        super(DQN, self).__init__(agent)

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        super(DQN, self).post_body_init()
        self.act_select_net=self.target_net
        self.eval_net=self.target_net
        # Network update params
        algorithm_spec=self.agent.spec['algorithm']
        self.update_type=algorithm_spec['update_type']
        self.update_frequency=algorithm_spec['update_frequency']
        self.polyak_weight=algorithm_spec['polyak_weight']


class DoubleDQN(DQNBase):

    def __init__(self, agent):
        super(DoubleDQN, self).__init__(agent)

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        super(DoubleDQN, self).post_body_init()
        self.act_select_net=self.net
        self.eval_net=self.target_net
        # Network update params
        algorithm_spec=self.agent.spec['algorithm']
        self.update_type=algorithm_spec['update_type']
        self.update_frequency=algorithm_spec['update_frequency']
        self.polyak_weight=algorithm_spec['polyak_weight']
