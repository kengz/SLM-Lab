from copy import deepcopy
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns, act_update_fns, decay_learning_rate
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import numpy as np
import pydash as _
import sys
import torch


class SARSA(Algorithm):
    '''Implementation of SARSA.

    Algorithm:
    Repeat:
        1. Collect some examples by acting in the environment and store them in an on policy replay memory (either batch or episodic)
        2. For each example calculate the target (bootstrapped estimate of the discounted value of the state and action taken), y, using a neural network to approximate the Q function. s_t' is the next state following the action actually taken, a_t. a_t' is the action actually taken in the next state s_t'.
                y_t = r_t + gamma * Q(s_t', a_t')
        4. For each example calculate the current estimate of the discounted value of the state and action taken
                x_t = Q(s_t, a_t)
        5. Calculate L(x, y) where L is a regression loss (eg. mse)
        6. Calculate the gradient of L with respect to all the parameters in the network and update the network parameters using the gradient
    '''

    def __init__(self, agent):
        '''
        After initialization SARSA has an attribute self.agent which contains a reference to the entire Agent acting in the environment.
        Agent components:
            - algorithm (with a net: neural network function approximator, and a policy: how to act in the environment). One algorithm per agent, shared across all bodies of the agent
            - memory (one per body)
        '''
        super(SARSA, self).__init__(agent)

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first. A body is a part of an Agent. Agents may have 1 to k bodies. Bodies do the acting in environments, and contain:
            - Memory (holding experiences obtained by acting in the environment)
            - State and action dimentions for an environment
            - Boolean var for if the action space is discrete
        '''
        self.init_nets()
        self.init_algo_params()
        logger.info(util.self_desc(self))

    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        body = self.agent.nanflat_body_a[0]  # single-body algo
        state_dim = body.state_dim  # dimension of the environment state, e.g. 4
        action_dim = body.action_dim  # dimension of the environment actions, e.g. 2
        net_spec = self.agent.spec['net']
        net_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))
        self.net = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim, **net_kwargs)
        self.set_net_attributes()

    def set_net_attributes(self):
        '''Initializes additional parameters from the net spec. Called by init_nets'''
        net_spec = self.agent.spec['net']
        util.set_attr(self, _.pick(net_spec, [
            'decay_lr', 'decay_lr_frequency', 'decay_lr_min_timestep',
        ]))

    def init_algo_params(self):
        '''Initialize other algorithm parameters.'''
        algorithm_spec = self.agent.spec['algorithm']
        net_spec = self.agent.spec['net']
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        self.action_policy_update = act_update_fns[algorithm_spec['action_policy_update']]
        self.set_other_algo_attributes()
        self.nanflat_explore_var_a = [
            self.explore_var_start] * self.agent.body_num

    def set_other_algo_attributes(self):
        '''Initializes additional parameters from the algorithm spec. Called by init_algo_params'''
        algorithm_spec = self.agent.spec['algorithm']
        util.set_attr(self, _.pick(algorithm_spec, [
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_start', 'explore_var_end', 'explore_anneal_epi',
            'gamma',  # the discount factor
            'training_frequency',  # how often to train for batch training (once each training_frequency time steps)
            'num_epis_to_collect',  # how many episodes to collect before training for episodic training
        ]))

    def compute_q_target_values(self, batch):
        '''Computes the target Q values for a batch of experiences'''
        # TODO update for sarsa
        return q_targets

    def sample(self):
        '''Samples a batch from memory of size self.batch_size'''
        # TODO update for sarsa
        return batch

    @lab_api
    def train(self):
        '''Completes one training step for the agent if it is time to train.
           i.e. the environment timestep is greater than the minimum training
           timestep and a multiple of the training_frequency.
           # TODO sarsa comments
           Otherwise this function does nothing.
        '''
        t = util.s_get(self, 'aeb_space.clock').get('total_t')
        if (t > self.training_min_timestep and t % self.training_frequency == 0):
            logger.debug3(f'Training at t: {t}')
            # TODO update for sarsa
            return total_loss
        else:
            logger.debug3('NOT training')
            return np.nan

    @lab_api
    def body_act_discrete(self, body, state):
        ''' Selects and returns a discrete action for body using the action policy'''
        return self.action_policy(body, state, self.net, self.nanflat_explore_var_a[body.nanflat_a_idx])

    def update_explore_var(self):
        '''Updates the explore variables'''
        space_clock = util.s_get(self, 'aeb_space.clock')
        nanflat_explore_var_a = self.action_policy_update(self, space_clock)
        explore_var_a = self.nanflat_to_data_a(
            'explore_var', nanflat_explore_var_a)
        return explore_var_a

    def update_learning_rate(self):
        decay_learning_rate(self, [self.net])

    @lab_api
    def update(self):
        '''Update the agent after training'''
        self.update_learning_rate()
        return self.update_explore_var()
