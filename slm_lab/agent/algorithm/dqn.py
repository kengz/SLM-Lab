from copy import deepcopy
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns, act_update_fns, decay_learning_rate
from slm_lab.agent.algorithm.sarsa import SARSA
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import numpy as np
import pydash as ps
import torch

logger = logger.get_logger(__name__)


class VanillaDQN(SARSA):
    '''
    Implementation of a simple DQN algorithm.
    Algorithm:
        1. Collect some examples by acting in the environment and store them in a replay memory
        2. Every K steps sample N examples from replay memory
        3. For each example calculate the target (bootstrapped estimate of the discounted value of the state and action taken), y, using a neural network to approximate the Q function. s' is the next state following the action actually taken.
                y_t = r_t + gamma * argmax_a Q(s_t', a)
        4. For each example calculate the current estimate of the discounted value of the state and action taken
                x_t = Q(s_t, a_t)
        5. Calculate L(x, y) where L is a regression loss (eg. mse)
        6. Calculate the gradient of L with respect to all the parameters in the network and update the network parameters using the gradient
        7. Repeat steps 3 - 6 M times
        8. Repeat steps 2 - 7 Z times
        9. Repeat steps 1 - 8

    For more information on Q-Learning see Sergey Levine's lectures 6 and 7 from CS294-112 Fall 2017
    https://www.youtube.com/playlist?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3
    '''

    def __init__(self, agent):
        '''
        After initialization VanillaDQN has an attribute self.agent which contains a reference to the entire Agent acting in the environment.
        Agent components:
            - algorithm (with a net: neural network function approximator, and a policy: how to act in the environment). One algorithm per agent, shared across all bodies of the agent
            - memory (one per body)
        '''
        super(VanillaDQN, self).__init__(agent)

    @lab_api
    def post_body_init(self):
        '''
        Initializes the part of algorithm needing a body to exist first. A body is a part of an Agent. Agents may have 1 to k bodies. Bodies do the acting in environments, and contain:
            - Memory (holding experiences obtained by acting in the environment)
            - State and action dimentions for an environment
            - Boolean var for if the action space is discrete
        '''
        super(VanillaDQN, self).post_body_init()

    @lab_api
    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        self.body = self.agent.nanflat_body_a[0]  # single-body algo
        if 'Recurrent' in self.net_spec['type']:
            raise ValueError('Recurrent networks does not work with DQN family of algorithms.')

        if self.algorithm_spec['name'] == 'VanillaDQN':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_weight']), 'Network update not available for VanillaDQN; use DQN.'
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self, self.body)
        logger.info(f'Training on gpu: {self.net.gpu}')

    @lab_api
    def init_algorithm_params(self):
        util.set_attr(self, self.algorithm_spec, [
            'action_policy',
            'action_policy_update',
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_start', 'explore_var_end', 'explore_anneal_epi',
            'gamma',  # the discount factor
            'training_epoch',  # how many batches to train each time
            'training_frequency',  # how often to train (once a few timesteps)
            'training_iters_per_batch',  # how many times to train each batch
            'training_min_timestep',  # how long before starting training
        ])
        super(VanillaDQN, self).init_algorithm_params()

    def compute_q_target_values(self, batch):
        '''Computes the target Q values for a batch of experiences'''
        # Calculate the Q values of the current and next states
        q_sts = self.net.wrap_eval(batch['states'])
        q_next_st = self.net.wrap_eval(batch['next_states'])
        logger.debug2(f'Q next states: {q_next_st.size()}')
        # Get the max for each next state
        q_next_st_max, _ = torch.max(q_next_st, dim=1)
        # Expand the dims so that q_next_st_max can be broadcast
        q_next_st_max.unsqueeze_(1)
        logger.debug2(f'Q next_states max {q_next_st_max.size()}')
        # Compute q_targets using reward and estimated best Q value from the next state if there is one
        # Make future reward 0 if the current state is done
        q_targets_max = batch['rewards'].data + self.gamma * torch.mul((1 - batch['dones'].data), q_next_st_max)
        logger.debug2(f'Q targets max: {q_targets_max.size()}')
        # We only want to train the network for the action selected in the current state
        # For all other actions we set the q_target = q_sts so that the loss for these actions is 0
        q_targets = torch.mul(q_targets_max, batch['actions'].data) + torch.mul(q_sts, (1 - batch['actions'].data))
        logger.debug2(f'Q targets: {q_targets.size()}')
        return q_targets

    @lab_api
    def sample(self):
        '''Samples a batch from memory of size self.memory_spec['batch_size']'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        batch = util.concat_dict(batches)
        util.to_torch_batch(batch, self.net.gpu)
        return batch

    @lab_api
    def train(self):
        '''
        Completes one training step for the agent if it is time to train.
        i.e. the environment timestep is greater than the minimum training timestep and a multiple of the training_frequency.
        Each training step consists of sampling n batches from the agent's memory.
        For each of the batches, the target Q values (q_targets) are computed and a single training step is taken k times
        Otherwise this function does nothing.
        '''
        t = util.s_get(self, 'aeb_space.clock').get('total_t')
        if (t > self.training_min_timestep and t % self.training_frequency == 0):
            logger.debug3(f'Training at t: {t}')
            total_loss = 0.0
            for _b in range(self.training_epoch):
                batch = self.sample()
                batch_loss = 0.0
                for _i in range(self.training_iters_per_batch):
                    q_targets = self.compute_q_target_values(batch)
                    if torch.cuda.is_available() and self.net.gpu:
                        q_targets = q_targets.cuda()
                    y = Variable(q_targets)
                    loss = self.net.training_step(batch['states'], y)
                    batch_loss += loss.data[0]
                batch_loss /= self.training_iters_per_batch
                total_loss += batch_loss
            total_loss /= self.training_epoch
            logger.debug(f'total_loss {total_loss}')
            return total_loss
        else:
            logger.debug3('NOT training')
            return np.nan

    @lab_api
    def body_act_discrete(self, body, state):
        ''' Selects and returns a discrete action for body using the action policy'''
        return super(VanillaDQN, self).body_act_discrete(body, state)

    def update_explore_var(self):
        '''Updates the explore variables'''
        return super(VanillaDQN, self).update_explore_var()

    def update_learning_rate(self):
        super(VanillaDQN, self).update_learning_rate()

    @lab_api
    def update(self):
        '''Update the agent after training'''
        return super(VanillaDQN, self).update()


class DQNBase(VanillaDQN):
    '''
    Implementation of the base DQN algorithm.
    The algorithm follows the same general approach as VanillaDQN but is more general since it allows
    for two different networks (through self.net and self.target_net).

    self.net is used to act, and is the network trained.
    self.target_net is used to estimate the maximum value of the Q-function in the next state when calculating the target (see VanillaDQN comments).
    self.target_net is updated periodically to either match self.net (self.net.update_type = "replace") or to be a weighted average of self.net and the previous self.target_net (self.net.update_type = "polyak")
    If desired, self.target_net can be updated slowly, and this can help to stabilize learning.

    It also allows for different nets to be used to select the action in the next state and to evaluate the value of that action through self.online_net and self.eval_net. This can help reduce the tendency of DQN's to overestimate the value of the Q-function. Following this approach leads to the DoubleDQN algorithm.

    Setting all nets to self.net reduces to the VanillaDQN case.

    net: instance of an slm_lab/agent/net
    memory: instance of an slm_lab/agent/memory
    batch_size: how many examples from memory to sample at each training step
    action_policy: function (from algorithm_util.py) that determines how to select actions
    gamma: Real number in range [0, 1]. Determines how much to discount the future
    state_dim: dimension of the state space
    action_dim: dimensions of the action space
    '''

    @lab_api
    def init_nets(self):
        '''Initialize networks'''
        self.body = self.agent.nanflat_body_a[0]  # single-body algo
        if 'Recurrent' in self.net_spec['type']:
            raise ValueError('Recurrent networks does not work with DQN family of algorithms.')
        memory_name = self.memory_spec['name']
        if 'Atari' in memory_name:
            # Make adjustments for Atari mode
            # TODO should be auto-set from preprocessor
            self.body.state_dim = (84, 84, 4)
            logger.debug3(f'State dim: {self.body.state_dim}')
        elif 'Stack' in memory_name:
            # Make adjustments for StackedReplay memory
            if 'MLP' not in self.net_spec['type']:
                raise ValueError('StackedReplay should only be used with MLPs, to stack states with ConvNets use Atari memory.')
            self.body.state_dim = self.body.state_dim * self.memory_spec['stack_len']
            logger.debug3(f'State dim: {self.body.state_dim}')

        if self.algorithm_spec['name'] == 'DQNBase':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_weight']), 'Network update not available for DQNBase; use DQN.'
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self, self.body)
        self.target_net = NetClass(self, self.body)
        self.online_net = self.target_net
        self.eval_net = self.target_net
        logger.info(f'Training on gpu: {self.net.gpu}')

    def compute_q_target_values(self, batch):
        '''Computes the target Q values for a batch of experiences. Note that the net references may differ based on algorithm.'''
        q_sts = self.net.wrap_eval(batch['states'])
        # Use online_net to select actions in next state
        q_next_st_acts = self.online_net.wrap_eval(batch['next_states'])
        _val, q_next_acts = torch.max(q_next_st_acts, dim=1)
        logger.debug2(f'Q next action: {q_next_acts.size()}')
        # Select q_next_st_maxs based on action selected in q_next_acts
        # Evaluate the action selection using the eval net
        q_next_sts = self.eval_net.wrap_eval(batch['next_states'])
        logger.debug2(f'Q next_states: {q_next_sts.size()}')
        idx = torch.from_numpy(np.array(list(range(self.memory_spec['batch_size']))))
        if torch.cuda.is_available() and self.net.gpu:
            idx = idx.cuda()
        q_next_st_maxs = q_next_sts[idx, q_next_acts]
        q_next_st_maxs.unsqueeze_(1)
        logger.debug2(f'Q next_states max {q_next_st_maxs.size()}')
        # Compute final q_target using reward and estimated best Q value from the next state if there is one. Make next state Q value 0 if the current state is done
        q_targets_max = batch['rewards'].data + self.gamma * torch.mul((1 - batch['dones'].data), q_next_st_maxs)
        logger.debug2(f'Q targets max: {q_targets_max.size()}')
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_sts
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_max, batch['actions'].data) + torch.mul(q_sts, (1 - batch['actions'].data))
        logger.debug2(f'Q targets: {q_targets.size()}')
        return q_targets

    def update_nets(self):
        space_clock = util.s_get(self, 'aeb_space.clock')
        t = space_clock.get('total_t')
        if self.net.update_type == 'replace':
            if t % self.net.update_frequency == 0:
                logger.debug('Updating target_net by replacing')
                self.target_net = net_util.load_params(self.target_net, net_util.flatten_params(self.net))
                self.online_net = self.target_net
                self.eval_net = self.target_net
        elif self.net.update_type == 'polyak':
            logger.debug('Updating net by averaging')
            avg_params = self.net.polyak_weight * net_util.flatten_params(self.target_net) + (1 - self.net.polyak_weight) * net_util.flatten_params(self.net)
            self.target_net = net_util.load_params(self.target_net, avg_params)
            self.online_net = self.target_net
            self.eval_net = self.target_net
        else:
            raise ValueError('Unknown net.update_type. Should be "replace" or "polyak". Exiting.')

    @lab_api
    def update(self):
        '''Updates self.target_net and the explore variables'''
        self.update_nets()
        return super(DQNBase, self).update()


class DQN(DQNBase):
    @lab_api
    def init_nets(self):
        super(DQN, self).init_nets()


class DoubleDQN(DQN):
    @lab_api
    def init_nets(self):
        super(DoubleDQN, self).init_nets()
        self.online_net = self.net
        self.eval_net = self.target_net

    def update_nets(self):
        res = super(DoubleDQN, self).update_nets()
        space_clock = util.s_get(self, 'aeb_space.clock')
        t = space_clock.get('total_t')
        if self.net.update_type == 'replace':
            if t % self.net.update_frequency == 0:
                self.online_net = self.net
                self.eval_net = self.target_net
        elif self.net.update_type == 'polyak':
            self.online_net = self.net
            self.eval_net = self.target_net


class MultitaskDQN(DQN):
    '''
    Simplest Multi-task DQN implementation.
    States and action dimensions are concatenated, and a single shared network is reponsible for processing concatenated states, and generating one action per environment from a single output layer.
    '''

    @lab_api
    def init_nets(self):
        '''Initialize nets with multi-task dimensions, and set net params'''
        self.state_dims = [body.state_dim for body in self.agent.nanflat_body_a]
        self.action_dims = [body.action_dim for body in self.agent.nanflat_body_a]
        # NOTE use a virtual body with joined inputs
        body = deepcopy(self.agent.nanflat_body_a[0])
        body.state_dim = sum(self.state_dims)
        body.action_dim = sum(self.action_dims)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self, body)
        self.target_net = NetClass(self, body)
        self.online_net = self.target_net
        self.eval_net = self.target_net
        logger.info(f'Training on gpu: {self.net.gpu}')

    @lab_api
    def sample(self):
        # NOTE the purpose of multi-body is to parallelize and get more batch_sizes
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        # Package data into pytorch variables
        for batch_b in batches:
            util.to_torch_batch(batch_b, self.net.gpu)
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
        logger.debug3(f'Q sts: {q_sts}')
        # TODO parametrize usage of eval or target_net
        q_next_st_acts = self.online_net.wrap_eval(
            batch['next_states'])
        logger.debug3(f'Q next st act vals: {q_next_st_acts}')
        start_idx = 0
        q_next_acts = []
        for body in self.agent.nanflat_body_a:
            end_idx = start_idx + body.action_dim
            _val, q_next_act_b = torch.max(
                q_next_st_acts[:, start_idx:end_idx], dim=1)
            # Shift action so that they have the right indices in combined layer
            q_next_act_b += start_idx
            logger.debug2(
                f'Q next action for body {body.aeb}: {q_next_act_b.size()}')
            logger.debug3(f'Q next action for body {body.aeb}: {q_next_act_b}')
            q_next_acts.append(q_next_act_b)
            start_idx = end_idx

        # Select q_next_st_maxs based on action selected in q_next_acts
        q_next_sts = self.eval_net.wrap_eval(batch['next_states'])
        logger.debug2(f'Q next_states: {q_next_sts.size()}')
        logger.debug3(f'Q next_states: {q_next_sts}')
        idx = torch.from_numpy(np.array(list(range(self.memory_spec['batch_size']))))
        if torch.cuda.is_available() and self.net.gpu:
            idx = idx.cuda()
        q_next_st_maxs = []
        for q_next_act_b in q_next_acts:
            q_next_st_max_b = q_next_sts[idx, q_next_act_b]
            q_next_st_max_b.unsqueeze_(1)
            logger.debug2(f'Q next_states max {q_next_st_max_b.size()}')
            logger.debug3(f'Q next_states max {q_next_st_max_b}')
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
            if torch.cuda.is_available() and self.net.gpu:
                q_targets_max_b = q_targets_max_b.cuda()
            q_targets_maxs.append(q_targets_max_b)
            logger.debug2(f'Q targets max: {q_targets_max_b.size()}')
        q_targets_maxs = torch.cat(q_targets_maxs, dim=1)
        logger.debug2(f'Q targets maxes: {q_targets_maxs.size()}')
        logger.debug3(f'Q targets maxes: {q_targets_maxs}')
        # Also concat actions - each batch should have only two non zero dimensions
        actions = [batch_b['actions'] for batch_b in batches]
        combined_actions = torch.cat(actions, dim=1)
        logger.debug2(f'combined_actions: {combined_actions.size()}')
        logger.debug3(f'combined_actions: {combined_actions}')
        # We only want to train the network for the action selected
        # For all other actions we set the q_target = q_sts
        # So that the loss for these actions is 0
        q_targets = torch.mul(q_targets_maxs, combined_actions.data) + torch.mul(q_sts, (1 - combined_actions.data))
        logger.debug2(f'Q targets: {q_targets.size()}')
        logger.debug3(f'Q targets: {q_targets}')
        return q_targets

    def act(self, state_a):
        '''Non-atomizable act to override agent.act(), do a single pass on the entire state_a instead of composing body_act'''
        nanflat_action_a = self.action_policy(
            self.agent.nanflat_body_a, state_a, self.net, self.nanflat_explore_var_a, self.net.gpu)
        action_a = self.nanflat_to_data_a('action', nanflat_action_a)
        return action_a


class MultiHeadDQN(MultitaskDQN):
    '''Multi-task DQN with separate state and action processors per environment'''

    @lab_api
    def init_nets(self):
        '''Initialize nets with multi-task dimensions, and set net params'''
        # NOTE: Separate init from MultitaskDQN despite similarities so that this implementation can support arbitrary sized state and action heads (e.g. multiple layers)
        body_list = self.agent.nanflat_body_a
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self, body_list)
        self.target_net = NetClass(self, body_list)
        self.online_net = self.target_net
        self.eval_net = self.target_net
        logger.info(f'Training on gpu: {self.net.gpu}')

    @lab_api
    def sample(self):
        '''Samples one batch per environment'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        # Package data into pytorch variables
        for batch_b in batches:
            util.to_torch_batch(batch_b, self.net.gpu)
        batch = {'states': [], 'next_states': []}
        for b in batches:
            batch['states'].append(b['states'])
            batch['next_states'].append(b['next_states'])
        batch['batches'] = batches
        return batch

    def compute_q_target_values(self, batch):
        batches = batch['batches']
        # NOTE: q_sts, q_next_st_acts and q_next_sts are lists
        q_sts = self.net.wrap_eval(batch['states'])
        logger.debug3(f'Q sts: {q_sts}')
        q_next_st_acts = self.online_net.wrap_eval(
            batch['next_states'])
        logger.debug3(f'Q next st act vals: {q_next_st_acts}')
        q_next_acts = []
        for i, q in enumerate(q_next_st_acts):
            _val, q_next_act_b = torch.max(q, dim=1)
            logger.debug3(f'Q next action for body {i}: {q_next_act_b}')
            q_next_acts.append(q_next_act_b)
        # Select q_next_st_maxs based on action selected in q_next_acts
        q_next_sts = self.eval_net.wrap_eval(batch['next_states'])
        logger.debug3(f'Q next_states: {q_next_sts}')
        idx = torch.from_numpy(np.array(list(range(self.memory_spec['batch_size']))))
        if torch.cuda.is_available() and self.net.gpu:
            idx = idx.cuda()
        q_next_st_maxs = []
        for q_next_st_val_b, q_next_act_b in zip(q_next_sts, q_next_acts):
            q_next_st_max_b = q_next_st_val_b[idx, q_next_act_b]
            q_next_st_max_b.unsqueeze_(1)
            logger.debug2(f'Q next_states max {q_next_st_max_b.size()}')
            logger.debug3(f'Q next_states max {q_next_st_max_b}')
            q_next_st_maxs.append(q_next_st_max_b)
        # Compute q_targets per environment using reward and estimated best Q value from the next state if there is one
        # Make future reward 0 if the current state is done
        q_targets_maxs = []
        for b, batch_b in enumerate(batches):
            q_targets_max_b = batch_b['rewards'].data + self.gamma * torch.mul((1 - batch_b['dones'].data), q_next_st_maxs[b])
            q_targets_maxs.append(q_targets_max_b)
            logger.debug2(
                f'Batch {b}, Q targets max: {q_targets_max_b.size()}')
        # As in the standard DQN we only want to train the network for the action selected
        # For all other actions we set the q_target = q_sts
        # So that the loss for these actions is 0
        q_targets = []
        for b, batch_b in enumerate(batches):
            q_targets_b = torch.mul(q_targets_maxs[b], batch_b['actions'].data) + torch.mul(q_sts[b], (1 - batch_b['actions'].data))
            q_targets.append(q_targets_b)
            logger.debug2(f'Batch {b}, Q targets: {q_targets_b.size()}')
        return q_targets

    @lab_api
    def train(self):
        '''
        Completes one training step for the agent if it is time to train.
        i.e. the environment timestep is greater than the minimum training timestep and a multiple of the training_frequency.
        Each training step consists of sampling n batches from the agent's memory.
        For each of the batches, the target Q values (q_targets) are computed and a single training step is taken k times
        Otherwise this function does nothing.
        '''
        t = util.s_get(self, 'aeb_space.clock').get('total_t')
        if (t > self.training_min_timestep and t % self.training_frequency == 0):
            logger.debug3(f'Training at t: {t}')
            nanflat_loss_a = np.zeros(self.agent.body_num)
            for _b in range(self.training_epoch):
                batch_losses = np.zeros(self.agent.body_num)
                batch = self.sample()
                for _i in range(self.training_iters_per_batch):
                    q_targets = self.compute_q_target_values(batch)
                    if torch.cuda.is_available() and self.net.gpu:
                        q_targets = [q.cuda() for q in q_targets]
                    y = [Variable(q) for q in q_targets]
                    losses = self.net.training_step(batch['states'], y)
                    logger.debug(f'losses {losses}')
                    batch_losses += losses
                batch_losses /= self.training_iters_per_batch
                nanflat_loss_a += batch_losses
            nanflat_loss_a /= self.training_epoch
            loss_a = self.nanflat_to_data_a('loss', nanflat_loss_a)
            return loss_a
        else:
            logger.debug3('NOT training')
            return np.nan

    def update_nets(self):
        # NOTE: Once polyak updating for multi-headed networks is supported via updates to flatten_params and load_params then this can be removed
        space_clock = util.s_get(self, 'aeb_space.clock')
        t = space_clock.get('total_t')
        if self.net.update_type == 'replace':
            if t % self.net.update_frequency == 0:
                self.target_net = net_util.load_params(self.target_net, net_util.flatten_params(self.net))
                self.online_net = self.target_net
                self.eval_net = self.target_net
        elif self.net.update_type == 'polyak':
            raise NotImplementedError('"polyak" updating not supported yet for MultiHeadDQN, please use "replace" instead. Exiting.')
        else:
            raise ValueError('Unknown net.update_type. Should be "replace" or "polyak". Exiting.')
