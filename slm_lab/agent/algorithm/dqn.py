from copy import deepcopy
from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.sarsa import SARSA
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
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
        self.body = self.agent.nanflat_body_a[0]  # single-body algo
        super(VanillaDQN, self).post_body_init()

    @lab_api
    def init_algorithm_params(self):
        # set default
        util.set_attr(self, dict(
            action_pdtype='Argmax',
            action_policy='epsilon_greedy',
            action_policy_update='linear_decay',
            explore_var_start=1.0,
            explore_var_end=0.1,
            explore_anneal_epi=100,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_policy',
            'action_policy_update',
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_start', 'explore_var_end', 'explore_anneal_epi',
            'gamma',  # the discount factor
            'training_epoch',  # how many batches to train each time
            'training_frequency',  # how often to train (once a few timesteps)
            'training_epoch',  # how many times to train each batch
            'training_min_timestep',  # how long before starting training
        ])
        super(VanillaDQN, self).init_algorithm_params()

    @lab_api
    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        if 'Recurrent' in self.net_spec['type']:
            raise ValueError('Recurrent networks does not work with DQN family of algorithms.')

        if self.algorithm_spec['name'] == 'VanillaDQN':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_coef']), 'Network update not available for VanillaDQN; use DQN.'
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, self, self.body.state_dim, self.body.action_dim)
        logger.info(f'Training on gpu: {self.net.gpu}')

    def calc_q_targets(self, batch):
        '''Computes the target Q values for a batch of experiences'''
        q_preds = self.net.wrap_eval(batch['states'])
        next_q_preds = self.net.wrap_eval(batch['next_states'])
        # Bellman equation: compute max_q_targets using reward and max estimated Q values (0 if no next_state)
        max_next_q_preds, _ = torch.max(next_q_preds, dim=1)
        max_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * max_next_q_preds
        max_q_targets.unsqueeze_(1)
        # To train only for action taken, set q_target = q_pred for action not taken so that loss is 0
        q_targets = (max_q_targets * batch['actions']) + (q_preds * (1 - batch['actions']))
        if torch.cuda.is_available() and self.net.gpu:
            q_targets = q_targets.cuda()
        return q_targets

    @lab_api
    def sample(self):
        '''Samples a batch from memory of size self.memory_spec['batch_size']'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        batch = util.concat_batches(batches)
        batch = util.to_torch_batch(batch, self.net.gpu)
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
        total_t = util.s_get(self, 'aeb_space.clock').get('total_t')
        self.to_train = (total_t > self.training_min_timestep and total_t % self.training_frequency == 0)
        if self.to_train == 1:
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.sample()
                with torch.no_grad():
                    q_targets = self.calc_q_targets(batch)
                loss = self.net.training_step(batch['states'], q_targets)
                total_loss += loss
            loss = total_loss / self.training_epoch
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Loss: {loss}')
            self.last_loss = loss.item()
        else:
            self.last_loss = np.nan
        return self.last_loss

    @lab_api
    def body_act(self, body, state):
        '''Selects and returns a discrete action for body using the action policy'''
        return super(VanillaDQN, self).body_act(body, state)

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
    action_policy: function (from policy_util.py) that determines how to select actions
    gamma: Real number in range [0, 1]. Determines how much to discount the future
    state_dim: dimension of the state space
    action_dim: dimensions of the action space
    '''

    @lab_api
    def init_nets(self):
        '''Initialize networks'''
        if 'Recurrent' in self.net_spec['type']:
            raise ValueError('Recurrent networks does not work with DQN family of algorithms.')
        if self.algorithm_spec['name'] == 'DQNBase':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_coef']), 'Network update not available for DQNBase; use DQN.'
        in_dim, out_dim = self.body.state_dim, self.body.action_dim
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, self, in_dim, out_dim)
        self.target_net = NetClass(self.net_spec, self, in_dim, out_dim)
        self.online_net = self.target_net
        self.eval_net = self.target_net
        logger.info(f'Training on gpu: {self.net.gpu}')

    def calc_q_targets(self, batch):
        '''Computes the target Q values for a batch of experiences. Note that the net references may differ based on algorithm.'''
        q_preds = self.net.wrap_eval(batch['states'])
        # Use online_net to select actions in next state
        online_next_q_preds = self.online_net.wrap_eval(batch['next_states'])
        # Use eval_net to calculate next_q_preds for actions chosen by online_net
        next_q_preds = self.eval_net.wrap_eval(batch['next_states'])
        # Bellman equation: compute max_q_targets using reward and max estimated Q values (0 if no next_state)
        _, action_idxs = torch.max(online_next_q_preds, dim=1)
        batch_size = len(batch['dones'])
        max_next_q_preds = next_q_preds[range(batch_size), action_idxs]
        max_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * max_next_q_preds
        max_q_targets.unsqueeze_(1)
        # To train only for action taken, set q_target = q_pred for action not taken so that loss is 0
        q_targets = (max_q_targets * batch['actions']) + (q_preds * (1 - batch['actions']))
        if torch.cuda.is_available() and self.net.gpu:
            q_targets = q_targets.cuda()
        return q_targets

    def update_nets(self):
        space_clock = util.s_get(self, 'aeb_space.clock')
        total_t = space_clock.get('total_t')
        if self.net.update_type == 'replace':
            if total_t % self.net.update_frequency == 0:
                logger.debug('Updating target_net by replacing')
                self.target_net.load_state_dict(self.net.state_dict())
                self.online_net = self.target_net
                self.eval_net = self.target_net
        elif self.net.update_type == 'polyak':
            logger.debug('Updating net by averaging')
            net_util.polyak_update(self.net, self.target_net, self.net.polyak_coef)
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
        total_t = space_clock.get('total_t')
        if self.net.update_type == 'replace':
            if total_t % self.net.update_frequency == 0:
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
        self.body_list = self.agent.nanflat_body_a
        self.state_dims = [body.state_dim for body in self.body_list]
        self.action_dims = [body.action_dim for body in self.body_list]
        in_dim = sum(self.state_dims)
        out_dim = sum(self.action_dims)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, self, in_dim, out_dim)
        self.target_net = NetClass(self.net_spec, self, in_dim, out_dim)
        self.online_net = self.target_net
        self.eval_net = self.target_net
        logger.info(f'Training on gpu: {self.net.gpu}')

    @lab_api
    def calc_pdparam(self, x, evaluate=True):
        '''
        Calculate pdparams for multi-action by chunking the network logits output
        '''
        pdparam = super(MultitaskDQN, self).calc_pdparam(x, evaluate=evaluate)
        pdparam = torch.cat(torch.split(pdparam, self.action_dims, dim=1))
        return pdparam

    @lab_api
    def act(self, state_a):
        '''Non-atomizable act to override agent.act(), do a single pass on the entire state_a instead of composing body_act'''
        # TODO tmp correction. state_a has 1 dim too deep
        new_data = [s[0] for s in state_a]
        state_a = np.array(new_data, dtype=np.float).flatten()

        state = torch.from_numpy(state_a).float().unsqueeze_(dim=0)
        if torch.cuda.is_available() and self.net_spec['gpu']:
            state = state.cuda()
        pdparam = self.calc_pdparam(state, evaluate=False)
        # use multi-policy. note arg change
        action_a, action_pd_a = self.action_policy(pdparam, self, self.body_list)
        for idx, body in enumerate(self.body_list):
            action_pd = action_pd_a[idx]
            body.entropies.append(action_pd.entropy())
            body.log_probs.append(action_pd.log_prob(action_a[idx].float()))
        return action_a.numpy()

    @lab_api
    def sample(self):
        # NOTE the purpose of multi-body is to parallelize and get more batch_sizes
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        # Package data into pytorch variables
        for body_batch in batches:
            body_batch = util.to_torch_batch(body_batch, self.net.gpu)
        # Concat state
        combined_states = torch.cat(
            [body_batch['states'] for body_batch in batches], dim=1)
        combined_next_states = torch.cat(
            [body_batch['next_states'] for body_batch in batches], dim=1)
        batch = {'states': combined_states, 'next_states': combined_next_states}
        # retain body-batches for body-wise q_targets calc
        batch['body_batches'] = batches
        return batch

    def calc_q_targets(self, batch):
        '''Compute the target Q values for multitask network by iterating through the slices corresponding to bodies, and computing the singleton function'''
        q_preds = self.net.wrap_eval(batch['states'])
        # Use online_net to select actions in next state
        online_next_q_preds = self.online_net.wrap_eval(
            batch['next_states'])
        next_q_preds = self.eval_net.wrap_eval(batch['next_states'])
        start_idx = 0
        multi_q_targets = []
        # iterate over body, use slice with proper idx offset
        for b, body_batch in enumerate(batch['body_batches']):
            body = self.agent.nanflat_body_a[b]
            end_idx = start_idx + body.action_dim
            _, action_idxs = torch.max(online_next_q_preds[:, start_idx:end_idx], dim=1)
            # Offset action index properly
            action_idxs += start_idx
            batch_size = len(body_batch['dones'])
            max_next_q_preds = next_q_preds[range(batch_size), action_idxs]
            max_q_targets = body_batch['rewards'] + self.gamma * (1 - body_batch['dones']) * max_next_q_preds
            max_q_targets.unsqueeze_(1)
            q_targets = (max_q_targets * body_batch['actions']) + (q_preds[:, start_idx:end_idx] * (1 - body_batch['actions']))
            multi_q_targets.append(q_targets)
            start_idx = end_idx
        q_targets = torch.cat(multi_q_targets, dim=1)
        if torch.cuda.is_available() and self.net.gpu:
            q_targets = q_targets.cuda()
        return q_targets


class HydraDQN(MultitaskDQN):
    '''Multi-task DQN with separate state and action processors per environment'''

    @lab_api
    def init_nets(self):
        '''Initialize nets with multi-task dimensions, and set net params'''
        # NOTE: Separate init from MultitaskDQN despite similarities so that this implementation can support arbitrary sized state and action heads (e.g. multiple layers)
        self.body_list = self.agent.nanflat_body_a
        self.state_dims = in_dims = [body.state_dim for body in self.body_list]
        self.action_dims = out_dims = [body.action_dim for body in self.body_list]
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, self, in_dims, out_dims)
        self.target_net = NetClass(self.net_spec, self, in_dims, out_dims)
        self.online_net = self.target_net
        self.eval_net = self.target_net
        logger.info(f'Training on gpu: {self.net.gpu}')

    @lab_api
    def calc_pdparam(self, x, evaluate=True):
        '''
        Calculate pdparams for multi-action by chunking the network logits output
        '''
        x = torch.cat(torch.split(x, self.state_dims, dim=1)).unsqueeze_(dim=1)
        pdparam = SARSA.calc_pdparam(self, x, evaluate=evaluate)
        return pdparam

    @lab_api
    def sample(self):
        '''Samples one batch per environment'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        batch = {'states': [], 'next_states': []}
        for body_batch in batches:
            body_batch = util.to_torch_batch(body_batch, self.net.gpu)
            batch['states'].append(body_batch['states'])
            batch['next_states'].append(body_batch['next_states'])
        # retain body-batches for body-wise q_targets calc
        batch['body_batches'] = batches
        return batch

    def calc_q_targets(self, batch):
        '''Compute the target Q values for hydra network by iterating through the tails corresponding to bodies, and computing the singleton function'''
        q_preds = self.net.wrap_eval(batch['states'])
        online_next_q_preds = self.online_net.wrap_eval(batch['next_states'])
        next_q_preds = self.eval_net.wrap_eval(batch['next_states'])
        multi_q_targets = []
        # iterate over body, use proper output tail
        for b, body_batch in enumerate(batch['body_batches']):
            _, action_idxs = torch.max(online_next_q_preds[b], dim=1)
            batch_size = len(body_batch['dones'])
            max_next_q_preds = next_q_preds[b][range(batch_size), action_idxs]
            max_q_targets = body_batch['rewards'] + self.gamma * (1 - body_batch['dones']) * max_next_q_preds
            max_q_targets.unsqueeze_(1)
            q_targets = (max_q_targets * body_batch['actions']) + (q_preds[b] * (1 - body_batch['actions']))
            if torch.cuda.is_available() and self.net.gpu:
                q_targets = q_targets.cuda()
            multi_q_targets.append(q_targets)
        # return as list for compatibility with net output in training_step
        q_targets = multi_q_targets
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
        total_t = util.s_get(self, 'aeb_space.clock').get('total_t')
        self.to_train = (total_t > self.training_min_timestep and total_t % self.training_frequency == 0)
        if self.to_train == 1:
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.sample()
                with torch.no_grad():
                    q_targets = self.calc_q_targets(batch)
                loss = self.net.training_step(batch['states'], q_targets)
                total_loss += loss
            loss = total_loss / self.training_epoch
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Loss: {loss}')
            self.last_loss = loss.item()
        else:
            self.last_loss = np.nan
        return self.last_loss
