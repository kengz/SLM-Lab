from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.sarsa import SARSA
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util, util
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

    e.g. algorithm_spec
    "algorithm": {
        "name": "VanillaDQN",
        "action_pdtype": "Argmax",
        "action_policy": "epsilon_greedy",
        "explore_var_spec": {
            "name": "linear_decay",
            "start_val": 1.0,
            "end_val": 0.1,
            "start_step": 10,
            "end_step": 1000,
        },
        "gamma": 0.99,
        "training_batch_epoch": 8,
        "training_epoch": 4,
        "training_frequency": 10,
        "training_start_step": 10,
        "normalize_state": true
    }
    '''

    @lab_api
    def init_algorithm_params(self):
        # set default
        util.set_attr(self, dict(
            action_pdtype='Argmax',
            action_policy='epsilon_greedy',
            explore_var_spec=None,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_spec',
            'gamma',  # the discount factor
            'training_batch_epoch',  # how many gradient updates per batch
            'training_epoch',  # how many batches to train each time
            'training_frequency',  # how often to train (once a few timesteps)
            'training_start_step',  # how long before starting training
            'normalize_state',
        ])
        super(VanillaDQN, self).init_algorithm_params()

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network used to learn the Q function from the spec'''
        if self.algorithm_spec['name'] == 'VanillaDQN':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_coef']), 'Network update not available for VanillaDQN; use DQN.'
        if global_nets is None:
            in_dim = self.body.state_dim
            out_dim = net_util.get_out_dim(self.body)
            NetClass = getattr(net, self.net_spec['type'])
            self.net = NetClass(self.net_spec, in_dim, out_dim)
            self.net_names = ['net']
        else:
            util.set_attr(self, global_nets)
            self.net_names = list(global_nets.keys())
        self.post_init_nets()

    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        states = batch['states']
        next_states = batch['next_states']
        q_preds = self.net(states)
        with torch.no_grad():
            next_q_preds = self.net(next_states)
        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        # Bellman equation: compute max_q_targets using reward and max estimated Q values (0 if no next_state)
        max_next_q_preds, _ = next_q_preds.max(dim=-1, keepdim=True)
        max_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * max_next_q_preds
        logger.debug(f'act_q_preds: {act_q_preds}\nmax_q_targets: {max_q_targets}')
        q_loss = self.net.loss_fn(act_q_preds, max_q_targets)

        # TODO use the same loss_fn but do not reduce yet
        if 'Prioritized' in util.get_class_name(self.body.memory):  # PER
            errors = torch.abs(max_q_targets - act_q_preds.detach())
            self.body.memory.update_priorities(errors)
        return q_loss

    @lab_api
    def act(self, state):
        '''Selects and returns a discrete action for body using the action policy'''
        return super(VanillaDQN, self).act(state)

    @lab_api
    def sample(self):
        '''Samples a batch from memory of size self.memory_spec['batch_size']'''
        batch = self.body.memory.sample()
        if self.normalize_state:
            batch = policy_util.normalize_states_and_next_states(self.body, batch)
        batch = util.to_torch_batch(batch, self.net.device, self.body.memory.is_episodic)
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
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            total_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.sample()
                for _ in range(self.training_batch_epoch):
                    loss = self.calc_q_loss(batch)
                    self.net.training_step(loss=loss, lr_clock=clock)
                    total_loss += loss
            loss = total_loss / (self.training_epoch * self.training_batch_epoch)
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, total_t: {clock.total_t}, t: {clock.t}, total_reward so far: {self.body.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

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
    '''

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize networks'''
        if self.algorithm_spec['name'] == 'DQNBase':
            assert all(k not in self.net_spec for k in ['update_type', 'update_frequency', 'polyak_coef']), 'Network update not available for DQNBase; use DQN.'
        if global_nets is None:
            in_dim = self.body.state_dim
            out_dim = net_util.get_out_dim(self.body)
            NetClass = getattr(net, self.net_spec['type'])
            self.net = NetClass(self.net_spec, in_dim, out_dim)
            self.target_net = NetClass(self.net_spec, in_dim, out_dim)
            self.net_names = ['net', 'target_net']
        else:
            util.set_attr(self, global_nets)
            self.net_names = list(global_nets.keys())
        self.post_init_nets()
        self.online_net = self.target_net
        self.eval_net = self.target_net

    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        states = batch['states']
        next_states = batch['next_states']
        q_preds = self.net(states)
        with torch.no_grad():
            # Use online_net to select actions in next state
            online_next_q_preds = self.online_net(next_states)
            # Use eval_net to calculate next_q_preds for actions chosen by online_net
            next_q_preds = self.eval_net(next_states)
        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
        max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
        max_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * max_next_q_preds
        logger.debug(f'act_q_preds: {act_q_preds}\nmax_q_targets: {max_q_targets}')
        q_loss = self.net.loss_fn(act_q_preds, max_q_targets)

        # TODO use the same loss_fn but do not reduce yet
        if 'Prioritized' in util.get_class_name(self.body.memory):  # PER
            errors = torch.abs(max_q_targets - act_q_preds.detach())
            self.body.memory.update_priorities(errors)
        return q_loss

    def update_nets(self):
        total_t = self.body.env.clock.total_t
        if total_t % self.net.update_frequency == 0:
            if self.net.update_type == 'replace':
                net_util.copy(self.net, self.target_net)
            elif self.net.update_type == 'polyak':
                net_util.polyak_update(self.net, self.target_net, self.net.polyak_coef)
            else:
                raise ValueError('Unknown net.update_type. Should be "replace" or "polyak". Exiting.')

    @lab_api
    def update(self):
        '''Updates self.target_net and the explore variables'''
        self.update_nets()
        return super(DQNBase, self).update()


class DQN(DQNBase):
    '''
    DQN class

    e.g. algorithm_spec
    "algorithm": {
        "name": "DQN",
        "action_pdtype": "Argmax",
        "action_policy": "epsilon_greedy",
        "explore_var_spec": {
            "name": "linear_decay",
            "start_val": 1.0,
            "end_val": 0.1,
            "start_step": 10,
            "end_step": 1000,
        },
        "gamma": 0.99,
        "training_batch_epoch": 8,
        "training_epoch": 4,
        "training_frequency": 10,
        "training_start_step": 10
    }
    '''
    @lab_api
    def init_nets(self, global_nets=None):
        super(DQN, self).init_nets(global_nets)


class DoubleDQN(DQN):
    '''
    Double-DQN (DDQN) class

    e.g. algorithm_spec
    "algorithm": {
        "name": "DDQN",
        "action_pdtype": "Argmax",
        "action_policy": "epsilon_greedy",
        "explore_var_spec": {
            "name": "linear_decay",
            "start_val": 1.0,
            "end_val": 0.1,
            "start_step": 10,
            "end_step": 1000,
        },
        "gamma": 0.99,
        "training_batch_epoch": 8,
        "training_epoch": 4,
        "training_frequency": 10,
        "training_start_step": 10
    }
    '''
    @lab_api
    def init_nets(self, global_nets=None):
        super(DoubleDQN, self).init_nets(global_nets)
        self.online_net = self.net
        self.eval_net = self.target_net
