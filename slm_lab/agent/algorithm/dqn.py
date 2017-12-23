from copy import deepcopy
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns, act_update_fns
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from torch.autograd import Variable
import numpy as np
import pydash as _
import sys
import torch


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
        body = self.agent.flat_nonan_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        net_spec = self.agent.spec['net']
        self.net = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        self.target_net = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        self.online_net = self.net
        self.eval_net = self.net
        self.batch_size = net_spec['batch_size']
        # Default network update params for base
        self.update_type = 'replace'
        self.update_frequency = 1
        self.polyak_weight = 0.9

        # TODO adjust learning rate http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate
        # TODO hackish optimizer learning rate, also it fails for SGD wtf

        algorithm_spec = self.agent.spec['algorithm']
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        # explore_var is epsilon, tau or etc.
        self.explore_var_start = algorithm_spec['explore_var_start']
        self.explore_var_end = algorithm_spec['explore_var_end']
        self.explore_var = self.explore_var_start
        self.explore_anneal_epi = algorithm_spec['explore_anneal_epi']
        self.gamma = algorithm_spec['gamma']

        self.training_min_timestep = algorithm_spec['training_min_timestep']
        self.training_frequency = algorithm_spec['training_frequency']
        self.training_epoch = algorithm_spec['training_epoch']
        self.training_iters_per_batch = algorithm_spec['training_iters_per_batch']
        # TODO standardize agent and env print self
        logger.info(str(self.net))

    def compute_q_target_values(self, batch):
        q_vals = self.net.wrap_eval(batch['states'])
        # Use act_select network to select actions in next state
        # Depending on the algorithm this is either the current
        # net or target net
        q_next_st_act_vals = self.online_net.wrap_eval(
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

    def sample(self):
        # TODO generalize to gather from all bodies
        batch = self.agent.body_a[(0, 0)].memory.sample(self.batch_size)
        # Package data into pytorch variables
        float_data_names = [
            'states', 'actions', 'rewards', 'dones', 'next_states']
        for k in float_data_names:
            batch[k] = Variable(torch.from_numpy(batch[k]).float())
        return batch

    def train(self):
        # TODO docstring
        t = util.s_get(self, 'aeb_space.clock').get('total_t')
        if (t > self.training_min_timestep and t % self.training_frequency == 0):
            # print('Training')
            total_loss = 0.0
            for _b in range(self.training_epoch):
                batch = self.sample()
                batch_loss = 0.0
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

    def body_act_discrete(self, body, state):
        # TODO can handle identical bodies now; to use body_net for specific body.
        return self.action_policy(body, state, self.net, self.explore_var)

    def update(self):
        t = util.s_get(self, 'aeb_space.clock').get('total_t')
        # if t % 100 == 0:
        # print(f'Total time step: {t}')
        '''Update epsilon or boltzmann for policy after net training'''
        # TODO refactor these info algorithm_util
        epi = util.s_get(self, 'aeb_space.clock').get('e')
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
            avg_params = self.polyak_weight * net_util.flatten_params(self.target_net) + \
                (1 - self.polyak_weight) * net_util.flatten_params(self.net)
            self.target_net = net_util.load_params(self.target_net, avg_params)
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
        # TODO rename this as action_net or policy_net
        self.online_net = self.target_net
        self.eval_net = self.target_net
        # Network update params
        net_spec = self.agent.spec['net']
        self.update_type = net_spec['update_type']
        self.update_frequency = net_spec['update_frequency']
        self.polyak_weight = net_spec['polyak_weight']
        print(
            f'Network update: type: {self.update_type}, frequency: {self.update_frequency}, weight: {self.polyak_weight}')

    def update(self):
        super(DQN, self).update()
        self.online_net = self.target_net
        self.eval_net = self.target_net


class DoubleDQN(DQNBase):
    # TODO: Check this is working
    def __init__(self, agent):
        super(DoubleDQN, self).__init__(agent)

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        super(DoubleDQN, self).post_body_init()
        self.online_net = self.net
        self.eval_net = self.target_net
        # Network update params
        net_spec = self.agent.spec['net']
        self.update_type = net_spec['update_type']
        self.update_frequency = net_spec['update_frequency']
        self.polyak_weight = net_spec['polyak_weight']

    def update(self):
        super(DoubleDQN, self).update()
        self.online_net = self.net
        self.eval_net = self.target_net


class MultitaskDQN(DQNBase):
    # TODO: Check this is working
    # TODO auto-architecture to handle multi-head, multi-tail nets
    def __init__(self, agent):
        super(MultitaskDQN, self).__init__(agent)

    def post_body_init(self):
        super(MultitaskDQN, self).post_body_init()
        '''Re-initialize nets with multi-task dimensions'''
        self.state_dims = [
            body.state_dim for body in self.agent.flat_nonan_body_a]
        self.action_dims = [
            body.action_dim for body in self.agent.flat_nonan_body_a]
        self.total_state_dim = sum(self.state_dims)
        self.total_action_dim = sum(self.action_dims)
        print(
            f'multitask state_dims: {self.state_dims}, sum {self.total_state_dim}')
        print(
            f'multitask action_dims: {self.action_dims}, sum {self.total_action_dim}')
        net_spec = self.agent.spec['net']
        self.net = getattr(net, net_spec['type'])(
            self.total_state_dim, net_spec['hid_layers'], self.total_action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        print(self.net)
        self.target_net = getattr(net, net_spec['type'])(
            self.total_state_dim, net_spec['hid_layers'], self.total_action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        self.online_net = self.net
        self.eval_net = self.net

    def sample(self):
        # TODO loop over, gather per e.
        # TODO generalize for any number of e (len of body_a)
        batch_1 = self.agent.body_a[(0, 0)].memory.sample(self.batch_size)
        batch_2 = self.agent.body_a[(1, 0)].memory.sample(self.batch_size)
        # print("Inside get batch")
        # print("Batch 1: ")
        # print(batch_1)
        # print("Batch 2: ")
        # print(batch_2)
        # Package data into pytorch variables
        float_data_names = [
            'states', 'actions', 'rewards', 'dones', 'next_states']
        for k in float_data_names:
            batch_1[k] = Variable(torch.from_numpy(batch_1[k]).float())
            batch_2[k] = Variable(torch.from_numpy(batch_2[k]).float())
        # Concat state
        combined_states = torch.cat(
            [batch_1['states'], batch_2['states']], dim=1)
        combined_next_states = torch.cat(
            [batch_1['next_states'], batch_2['next_states']], dim=1)
        batch = {'states': combined_states,
                 'next_states': combined_next_states}
        # use recursive packaging to carry sub data
        batch['sub_1'] = batch_1
        batch['sub_2'] = batch_2
        return batch

    def compute_q_target_values(self, batch):
        batch_1 = batch['sub_1']
        batch_2 = batch['sub_2']
        # print("batch: {}".format(batch['states'].size()))
        # print("batch: {}".format(batch['states']))
        q_vals = self.net.wrap_eval(batch['states'])
        # Use act_select network to select actions in next state
        # Depending on the algorithm this is either the current
        # net or target net
        q_next_st_act_vals = self.online_net.wrap_eval(
            batch['next_states'])

        # Select two sets of next actions
        # TODO Generalize to more than two tasks
        _val, q_next_actions_1 = torch.max(
            q_next_st_act_vals[:, :self.action_dims[0]], dim=1)
        _val, q_next_actions_2 = torch.max(
            q_next_st_act_vals[:, self.action_dims[0]:], dim=1)
        # Shift next actions_2 so they have the right indices
        q_next_actions_2 = torch.add(q_next_actions_2, self.action_dims[0])
        # print("Q next actions 1: {}".format(q_next_actions_1.size()))
        # print("Q next actions 2: {}".format(q_next_actions_2.size()))
        # Select q_next_st_vals_max based on action selected in q_next_actions
        # Evaluate the action selection using the eval net
        # Depending on the algorithm this is either the current
        # net or target net
        q_next_st_vals = self.eval_net.wrap_eval(batch['next_states'])
        # print("Q next st vals: {}".format(q_next_st_vals.size()))
        idx = torch.from_numpy(np.array(list(range(self.batch_size))))
        # Calculate values for two sets of actions
        q_next_st_vals_max_1 = q_next_st_vals[idx, q_next_actions_1]
        q_next_st_vals_max_1.unsqueeze_(1)
        q_next_st_vals_max_2 = q_next_st_vals[idx, q_next_actions_2]
        q_next_st_vals_max_2.unsqueeze_(1)
        # print("Q next st vals max 1: {}".format(q_next_st_vals_max_1.size()))
        # print("Q next st vals max 2: {}".format(q_next_st_vals_max_2.size()))
        # Compute final q_target using reward and estimated
        # best Q value from the next state if there is one
        # Make future reward 0 if the current state is done
        # Do it individually first, then combine
        # Each individual target should automatically expand
        # to the dimension of the relevant action space
        q_targets_max_1 = (batch_1['rewards'].data + self.gamma * torch.mul(
            (1 - batch_1['dones'].data), q_next_st_vals_max_1)).numpy()
        q_targets_max_2 = (batch_2['rewards'].data + self.gamma * torch.mul(
            (1 - batch_2['dones'].data), q_next_st_vals_max_2)).numpy()
        # print("Q targets max 1: {}".format(q_targets_max_1))
        # print("Q targets max 2: {}".format(q_targets_max_2))
        # print("Q targets max 1: {}".format(q_targets_max_1.shape))
        # print("Q targets max 2: {}".format(q_targets_max_2.shape))
        # Concat to form full size targets
        q_targets_max_1 = torch.from_numpy(
            np.broadcast_to(q_targets_max_1,
                            (q_targets_max_1.shape[0], self.action_dims[0])))
        q_targets_max_2 = torch.from_numpy(
            np.broadcast_to(q_targets_max_2,
                            (q_targets_max_2.shape[0], self.action_dims[1])))
        # print("Q targets max broadcast 1: {}".format(q_targets_max_1.size()))
        # print("Q targets max broadcast 2: {}".format(q_targets_max_2.size()))
        q_targets_max = torch.cat([q_targets_max_1, q_targets_max_2], dim=1)
        # print("Q targets max: {}".format(q_targets_max.size()))
        # Also concat actions - each batch should have only two
        # non zero dimensions
        combined_actions = torch.cat(
            [batch_1['actions'], batch_2['actions']], dim=1)
        # print("Batch 1 actions: {}".format(batch_1['actions']))
        # print("Batch 2 actions: {}".format(batch_2['actions']))
        # print("Combined actions: {}".format(combined_actions))
        # print("Combined actions size: {}".format(combined_actions.size()))
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_vals
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_max, combined_actions.data) + \
            torch.mul(q_vals, (1 - combined_actions.data))
        # print("Q targets size: {}".format(q_targets.size()))
        # exit()
        return q_targets

    def act(self, state_a):
        '''Non-atomizable act to override agent.act(), do a single pass on the entire state_a instead of composing body_act'''
        flat_nonan_action_a = self.action_policy(
            self.agent.flat_nonan_body_a, state_a, self.net, self.explore_var)
        return super(MultitaskDQN, self).flat_nonan_to_action_a(flat_nonan_action_a)
