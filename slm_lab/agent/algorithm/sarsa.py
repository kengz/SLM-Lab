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
        self.state_dim = body.state_dim  # dimension of the environment state, e.g. 4
        self.action_dim = body.action_dim  # dimension of the environment actions, e.g. 2
        net_spec = self.agent.spec['net']
        net_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))
        self.net = getattr(net, net_spec['type'])(
            self.state_dim, net_spec['hid_layers'], self.action_dim, **net_kwargs)
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
        self.to_train = 0
        self.set_memory_flag()

    def set_memory_flag(self):
        '''Flags if memory is episodic or discrete. This affects how self.sample() handles the batch it gets back from memory'''
        body = self.agent.nanflat_body_a[0]
        memory = body.memory.__class__.__name__
        if (memory.find('OnPolicyReplay') != -1) or (memory.find('OnPolicyNStepReplay') != -1):
            self.is_episodic = True
        elif (memory.find('OnPolicyBatchReplay') != -1) or (memory.find('OnPolicyNStepBatchReplay') != -1):
            self.is_episodic = False
        else:
            logger.warn(f'Error: Memory {memory} not recognized')
            raise NotImplementedError

    def compute_q_target_values(self, batch):
        '''Computes the target Q values for a batch of experiences'''
        # Calculate the Q values of the current and next states
        q_sts = self.net.wrap_eval(batch['states'])
        q_next_st = self.net.wrap_eval(batch['next_states'])
        q_next_actions = batch['next_actions']
        logger.debug2(f'Q next states: {q_next_st.size()}')
        # Get the q value for the next action that was actually taken
        idx = torch.from_numpy(np.array(list(range(q_next_st.size(0)))))
        q_next_st_vals = q_next_st[idx, q_next_actions.squeeze_(1).data.long()]
        # Expand the dims so that q_next_st_vals can be broadcast
        q_next_st_vals.unsqueeze_(1)
        logger.debug2(f'Q next_states vals {q_next_st_vals.size()}')
        logger.debug3(f'Q next_states {q_next_st}')
        logger.debug3(f'Q next actions {q_next_actions}')
        logger.debug3(f'Q next_states vals {q_next_st_vals}')
        logger.debug3(f'Dones {batch["dones"]}')
        # Compute q_targets using reward and estimated best Q value from the next state if there is one
        # Make future reward 0 if the current state is done
        q_targets_actual = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), q_next_st_vals)
        logger.debug2(f'Q targets actual: {q_targets_actual.size()}')
        logger.debug3(f'Q states {q_sts}')
        logger.debug3(f'Q targets actual: {q_targets_actual}')
        # We only want to train the network for the action selected in the current state
        # For all other actions we set the q_target = q_sts so that the loss for these actions is 0
        q_targets = torch.mul(q_targets_actual, batch['actions_onehot'].data) + \
            torch.mul(q_sts, (1 - batch['actions_onehot'].data))
        logger.debug2(f'Q targets: {q_targets.size()}')
        logger.debug3(f'Q targets: {q_targets}')
        return q_targets

    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample()
                   for body in self.agent.nanflat_body_a]
        batch = util.concat_dict(batches)
        if self.is_episodic:
            util.to_torch_nested_batch(batch)
            # Add next action to batch
            batch['actions_onehot'] = []
            batch['next_actions'] = []
            for acts in batch['actions']:
                # The next actions are the actions shifted by one time step
                # For episodic training is does not matter that the action in the last state is set to zero since there is no corresponding next state. The Q target is just the reward received in the terminal state.
                next_acts = torch.zeros_like(acts)
                next_acts[:-1] = acts[1:]
                # Convert actions to one hot (both representations are needed for SARSA)
                acts_onehot = util.convert_to_one_hot(acts, self.action_dim)
                batch['actions_onehot'].append(acts_onehot)
                batch['next_actions'].append(next_acts)
            # Flatten the batch to train all at once
            batch = util.concat_episodes(batch)
        else:
            util.to_torch_batch(batch)
            # Batch only useful to train with if it has more than one element
            # Train function checks for this and skips training if batch is too small
            if batch['states'].size(0) > 1:
                batch['next_actions'] = torch.zeros_like(batch['actions'])
                batch['next_actions'][:-1] = batch['actions'][1:]
                batch['actions_onehot'] = util.convert_to_one_hot(batch['actions'], self.action_dim)
                batch_elems = ['states', 'actions', 'actions_onehot', 'rewards', 'dones', 'next_states', 'next_actions']
                for k in batch_elems:
                    if batch[k].dim() == 1:
                        batch[k].unsqueeze_(1)
                if batch['dones'].data[-1].int().eq_(0).numpy()[0]:
                    logger.debug(f'Popping last element')
                    # The last experience in the batch is not terminal the batch has to be shortened by one element since the algorithm does not yet have access to the next action taken for the final experience
                    for k in batch_elems:
                        batch[k] = batch[k][:-1]
        return batch

    @lab_api
    def train(self):
        '''Completes one training step for the agent if it is time to train.
           Otherwise this function does nothing.
        '''
        t = util.s_get(self, 'aeb_space.clock').get('total_t')
        if self.to_train == 1:
            logger.debug3(f'Training at t: {t}')
            batch = self.sample()
            if batch['states'].size(0) < 2:
                logger.info(f'Batch too small to train with, skipping...')
                self.to_train = 0
                return np.nan
            q_targets = self.compute_q_target_values(batch)
            y = Variable(q_targets)
            loss = self.net.training_step(batch['states'], y)
            logger.debug(f'loss {loss.data[0]}')
            self.to_train = 0
            return loss.data[0]
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
