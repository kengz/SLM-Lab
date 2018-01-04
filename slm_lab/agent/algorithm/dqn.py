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


class VanillaDQN(Algorithm):
    '''
    Implementation of a simple DQN algorithm.
    '''

    def __init__(self, agent):
        super(VanillaDQN, self).__init__(agent)
        '''
        After initialization VanillaDQN has an attribute self.agent which contains a reference to the entire Agent acting in the environment.
        Agent components:
            - algorithm (with a net: neural network function approximator, and a policy: how to act in the environment). One algorithm per agent, shared across all bodies of the agent
            - memory (one per body)
        '''
        logger.info(f'Torch random seed: {torch.initial_seed()}')

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first. A body is a part of an Agent. Agents may have 1 to k bodies. Bodies do the acting in environments, and contain:
            - Memory (holding experiences obtained by acting in the environment)
            - State and action dimentions for an environment
            - Boolean var for if the action space is discrete
            '''
        body = self.agent.flat_nonan_body_a[0]  # singleton algo
        # Initialize the neural network used to learn the Q function from the spec
        state_dim = body.state_dim  # dimension of the environment state, e.g. 4
        action_dim = body.action_dim  # dimension of the environment actions, e.g. 2
        net_spec = self.agent.spec['net']
        self.net = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        # Prints the network architecture to stdout
        self.net.print_nets()
        # Initialize the other algorithm parameters
        # self.batch_size: how many examples to learn from each training iteration
        self.batch_size = net_spec['batch_size']
        algorithm_spec = self.agent.spec['algorithm']
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        self.action_policy_update = act_update_fns[algorithm_spec['action_policy_update']]
        # explore_var is epsilon or tau depening on the action policy
        # explore var start, end, and anneal_epi control the trade off between exploration and exploitaton
        self.explore_var_start = algorithm_spec['explore_var_start']
        self.explore_var_end = algorithm_spec['explore_var_end']
        self.explore_var = self.explore_var_start
        self.explore_anneal_epi = algorithm_spec['explore_anneal_epi']
        self.gamma = algorithm_spec['gamma']  # the discount rate
        # self.training_min_timestep: how long to wait before starting training
        self.training_min_timestep = algorithm_spec['training_min_timestep']
        # self.training_frequency: how often to train
        self.training_frequency = algorithm_spec['training_frequency']
        # self.training_epoch: how many batches to train each time
        self.training_epoch = algorithm_spec['training_epoch']
        # self.training_iters_per_batch: how many times to train each batch
        self.training_iters_per_batch = algorithm_spec['training_iters_per_batch']

    def compute_q_target_values(self, batch):
        '''Computes the target Q values for a batch of experiences'''
        # Calculate the Q values of the current and next states
        q_sts = self.net.wrap_eval(batch['states'])
        q_next_st = self.net.wrap_eval(batch['next_states'])
        logger.debug(f'Q next states: {q_next_st.size()}')
        # Get the max for each next state
        q_next_st_max, _ = torch.max(q_next_st, dim=1)
        # Expand the dims so that q_next_st_max can be broadcast
        q_next_st_max.unsqueeze_(1)
        logger.debug(f'Q next_states max {q_next_st_max.size()}')
        # Compute q_targets using reward and estimated best Q value from the next state if there is one
        # Make future reward 0 if the current state is done
        q_targets_max = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), q_next_st_max)
        logger.debug(f'Q targets max: {q_targets_max.size()}')
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_sts
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_max, batch['actions'].data) + \
            torch.mul(q_sts, (1 - batch['actions'].data))
        logger.debug(f'Q targets: {q_targets.size()}')
        return q_targets

    def sample(self):
        '''Samples a batch from memory of size self.batch_size'''
        batches = [body.memory.sample(self.batch_size)
                   for body in self.agent.flat_nonan_body_a]
        batch = util.concat_dict(batches)
        util.to_torch_batch(batch)
        return batch

    def train(self):
        '''Completes one training step for the agent if it is time to train.
           i.e. the environment timestep is greater than the minimum training
           timestep and a multiple of the training_frequency.
           Each training step consists of sampling n batches from the agent's memory.
              For each of the batches, the target Q values (q_targets) are computed and
              a single training step is taken k times
           Otherwise this function does nothing.
        '''
        t = util.s_get(self, 'aeb_space.clock').get('total_t')
        if (t > self.training_min_timestep and t % self.training_frequency == 0):
            logger.debug(f'Training at t: {t}')
            total_loss = 0.0
            for _b in range(self.training_epoch):
                batch = self.sample()
                batch_loss = 0.0
                for _i in range(self.training_iters_per_batch):
                    q_targets = self.compute_q_target_values(batch)
                    y = Variable(q_targets)
                    loss = self.net.training_step(batch['states'], y)
                    batch_loss += loss.data[0]
                batch_loss /= self.training_iters_per_batch
                total_loss += batch_loss
            total_loss /= self.training_epoch
            logger.debug(f'total_loss {total_loss}')
            return total_loss
        else:
            logger.debug('NOT training')
            return np.nan

    def body_act_discrete(self, body, state):
        ''' Selects and returns a discrete using the action policy'''
        return self.action_policy(body, state, self.net, self.explore_var)

    def update(self):
        '''Updates the explore variables'''
        space_clock = util.s_get(self, 'aeb_space.clock')
        self.action_policy_update(self, space_clock)
        return self.explore_var


class DQNBase(VanillaDQN):
    '''
    Implementation of the base DQN algorithm.
    This is more general than the VanillaDQN since it allows
    for two different networks (through self.net and self.target_net).
    If desired, self.target_net can be updated more slowly
    to stabilize learning. It also allows for different nets to be used to
    select the action in the next state and to evaluate the value of that
    action through self.online_net and self.eval_net
    Setting all nets to self.net reduces to the VanillaDQN case.
    See Sergey Levine's lecture xxx for more details
    TODO add link
          more detailed comments

    net: instance of an slm_lab/agent/net
    memory: instance of an slm_lab/agent/memory
    batch_size: how many examples from memory to sample at each training step
    action_policy: function (from algorithm_util.py) that determines how to select actions
    gamma: Real number in range [0, 1]. Determines how much to discount the future
    state_dim: dimension of the state space
    action_dim: dimensions of the action space
    '''

    def __init__(self, agent):
        super(DQNBase, self).__init__(agent)

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        body = self.agent.flat_nonan_body_a[0]  # singleton algo
        # Initialize networks
        state_dim = body.state_dim
        action_dim = body.action_dim
        net_spec = self.agent.spec['net']
        self.net = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        self.net.print_nets()
        self.target_net = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        self.online_net = self.target_net
        self.eval_net = self.target_net
        self.batch_size = net_spec['batch_size']
        # Default network update params for base
        self.update_type = 'replace'
        self.update_frequency = 1
        self.polyak_weight = 0.9
        # Initialize other algorithm parameters
        algorithm_spec = self.agent.spec['algorithm']
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        self.action_policy_update = act_update_fns[algorithm_spec['action_policy_update']]
        # explore_var is epsilon, tau or etc.
        self.explore_var_start = algorithm_spec['explore_var_start']
        self.explore_var_end = algorithm_spec['explore_var_end']
        self.explore_var = self.explore_var_start
        self.explore_anneal_epi = algorithm_spec['explore_anneal_epi']
        self.gamma = algorithm_spec['gamma']
        # These parameter control how often and how much to train
        self.training_min_timestep = algorithm_spec['training_min_timestep']
        self.training_frequency = algorithm_spec['training_frequency']
        self.training_epoch = algorithm_spec['training_epoch']
        self.training_iters_per_batch = algorithm_spec['training_iters_per_batch']

    def compute_q_target_values(self, batch):
        '''Computes the target Q values for a batch of experiences'''
        q_sts = self.net.wrap_eval(batch['states'])
        # Use act_select network to select actions in next state
        # TODO parametrize usage of eval or target_net
        # Depending on the algorithm this is either the current net or target net
        q_next_st_acts = self.online_net.wrap_eval(batch['next_states'])
        _val, q_next_acts = torch.max(q_next_st_acts, dim=1)
        logger.debug(f'Q next action: {q_next_acts.size()}')
        # Select q_next_st_maxs based on action selected in q_next_acts
        # Evaluate the action selection using the eval net
        # Depending on the algorithm this is either the current net or target net
        q_next_sts = self.eval_net.wrap_eval(batch['next_states'])
        logger.debug(f'Q next_states: {q_next_sts.size()}')
        idx = torch.from_numpy(np.array(list(range(self.batch_size))))
        q_next_st_maxs = q_next_sts[idx, q_next_acts]
        q_next_st_maxs.unsqueeze_(1)
        logger.debug(f'Q next_states max {q_next_st_maxs.size()}')
        # Compute final q_target using reward and estimated best Q value from the next state if there is one
        # Make future reward 0 if the current state is done
        q_targets_max = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), q_next_st_maxs)
        logger.debug(f'Q targets max: {q_targets_max.size()}')
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_sts
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_max, batch['actions'].data) + \
            torch.mul(q_sts, (1 - batch['actions'].data))
        logger.debug(f'Q targets: {q_targets.size()}')
        return q_targets

    def sample(self):
        return super(DQNBase, self).sample()

    def train(self):
        return super(DQNBase, self).train()

    def body_act_discrete(self, body, state):
        return super(DQNBase, self).body_act_discrete(body, state)

    def update(self):
        '''Updates self.target_net and the explore variables'''
        space_clock = util.s_get(self, 'aeb_space.clock')
        # update explore_var
        self.action_policy_update(self, space_clock)
        # Update target net with current net
        t = space_clock.get('t')
        if self.update_type == 'replace':
            if t % self.update_frequency == 0:
                logger.debug('Updating target_net by replacing')
                self.target_net = deepcopy(self.net)
                self.online_net = self.target_net
                self.eval_net = self.target_net
        elif self.update_type == 'polyak':
            logger.debug('Updating net by averaging')
            avg_params = self.polyak_weight * net_util.flatten_params(self.target_net) + \
                (1 - self.polyak_weight) * net_util.flatten_params(self.net)
            self.target_net = net_util.load_params(self.target_net, avg_params)
            self.online_net = self.target_net
            self.eval_net = self.target_net
        else:
            logger.error(
                'Unknown net.update_type. Should be "replace" or "polyak". Exiting.')
            sys.exit()
        return self.explore_var


class DQN(DQNBase):
    def __init__(self, agent):
        super(DQN, self).__init__(agent)

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        super(DQN, self).post_body_init()
        # Network update params
        net_spec = self.agent.spec['net']
        self.update_type = net_spec['update_type']
        self.update_frequency = net_spec['update_frequency']
        self.polyak_weight = net_spec['polyak_weight']
        logger.debug(
            f'Network update: type: {self.update_type}, frequency: {self.update_frequency}, weight: {self.polyak_weight}')
        logger.info(util.self_desc(self))


class DoubleDQN(DQNBase):
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
        logger.info(util.self_desc(self))

    def update(self):
        super(DoubleDQN, self).update()
        space_clock = util.s_get(self, 'aeb_space.clock')
        t = space_clock.get('t')
        if self.update_type == 'replace':
            if t % self.update_frequency == 0:
                self.online_net = self.net
                self.eval_net = self.target_net
        elif self.update_type == 'polyak':
            self.online_net = self.net
            self.eval_net = self.target_net
        return self.explore_var


class MultitaskDQN(DQNBase):
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
        net_spec = self.agent.spec['net']
        self.net = getattr(net, net_spec['type'])(
            self.total_state_dim, net_spec['hid_layers'], self.total_action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        self.net.print_nets()
        self.target_net = getattr(net, net_spec['type'])(
            self.total_state_dim, net_spec['hid_layers'], self.total_action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
        )
        self.online_net = self.target_net
        self.eval_net = self.target_net
        logger.info(util.self_desc(self))

    def sample(self):
        # NOTE the purpose of multi-body is to parallelize and get more batch_sizes
        batches = [body.memory.sample(self.batch_size)
                   for body in self.agent.flat_nonan_body_a]
        # Package data into pytorch variables
        for batch_b in batches:
            util.to_torch_batch(batch_b)
        # Concat state
        combined_states = torch.cat(
            [batch_b['states'] for batch_b in batches], dim=1)
        combined_next_states = torch.cat(
            [batch_b['next_states'] for batch_b in batches], dim=1)
        batch = {'states': combined_states,
                 'next_states': combined_next_states}
        # use recursive packaging to carry sub data
        batch['batches'] = batches
        return batch

    def compute_q_target_values(self, batch):
        batches = batch['batches']
        q_sts = self.net.wrap_eval(batch['states'])
        logger.debug(f'Q sts: {q_sts}')
        # TODO parametrize usage of eval or target_net
        q_next_st_acts = self.online_net.wrap_eval(
            batch['next_states'])
        logger.debug(f'Q next st act vals: {q_next_st_acts}')
        start_idx = 0
        q_next_acts = []
        for body in self.agent.flat_nonan_body_a:
            end_idx = start_idx + body.action_dim
            _val, q_next_act_b = torch.max(
                q_next_st_acts[:, start_idx:end_idx], dim=1)
            # Shift action so that they have the right indices in combined layer
            q_next_act_b += start_idx
            logger.debug(
                f'Q next action for body {body.aeb}: {q_next_act_b.size()}')
            logger.debug(f'Q next action for body {body.aeb}: {q_next_act_b}')
            q_next_acts.append(q_next_act_b)
            start_idx = end_idx

        # Select q_next_st_maxs based on action selected in q_next_acts
        q_next_sts = self.eval_net.wrap_eval(batch['next_states'])
        logger.debug(f'Q next_states: {q_next_sts.size()}')
        logger.debug(f'Q next_states: {q_next_sts}')
        idx = torch.from_numpy(np.array(list(range(self.batch_size))))
        q_next_st_maxs = []
        for q_next_act_b in q_next_acts:
            q_next_st_max_b = q_next_sts[idx, q_next_act_b]
            q_next_st_max_b.unsqueeze_(1)
            logger.debug(f'Q next_states max {q_next_st_max_b.size()}')
            logger.debug(f'Q next_states max {q_next_st_max_b}')
            q_next_st_maxs.append(q_next_st_max_b)

        # Compute final q_target using reward and estimated best Q value from the next state if there is one. Make future reward 0 if the current state is done. Do it individually first, then combine. Each individual target should automatically expand to the dimension of the relevant action space
        q_targets_maxs = []
        for b, batch_b in enumerate(batches):
            q_targets_max_b = (batch_b['rewards'].data + self.gamma * torch.mul(
                (1 - batch_b['dones'].data), q_next_st_maxs[b])).numpy()
            q_targets_max_b = torch.from_numpy(
                np.broadcast_to(
                    q_targets_max_b,
                    (q_targets_max_b.shape[0], self.action_dims[b])))
            q_targets_maxs.append(q_targets_max_b)
            logger.debug(f'Q targets max: {q_targets_max_b.size()}')
        q_targets_maxs = torch.cat(q_targets_maxs, dim=1)
        logger.debug(f'Q targets maxes: {q_targets_maxs.size()}')
        logger.debug(f'Q targets maxes: {q_targets_maxs}')
        # Also concat actions - each batch should have only two non zero dimensions
        actions = [batch_b['actions'] for batch_b in batches]
        combined_actions = torch.cat(actions, dim=1)
        logger.debug(f'combined_actions: {combined_actions.size()}')
        logger.debug(f'combined_actions: {combined_actions}')
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_sts
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_maxs, combined_actions.data) + \
            torch.mul(q_sts, (1 - combined_actions.data))
        logger.debug(f'Q targets: {q_targets.size()}')
        logger.debug(f'Q targets: {q_targets}')
        return q_targets

    def act(self, state_a):
        '''Non-atomizable act to override agent.act(), do a single pass on the entire state_a instead of composing body_act'''
        flat_nonan_action_a = self.action_policy(
            self.agent.flat_nonan_body_a, state_a, self.net, self.explore_var)
        return super(MultitaskDQN, self).flat_nonan_to_action_a(flat_nonan_action_a)
