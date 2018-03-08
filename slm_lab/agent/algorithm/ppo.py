from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns, act_update_fns, decay_learning_rate
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import numpy as np
import torch
import pydash as _


class PPO(Algorithm):
    '''
    Implementation of PPO
    Original paper: "Proximal Policy Optimization Algorithms"
    https://arxiv.org/pdf/1707.06347.pdf

    Adapted from OpenAI baselines, CPU version https://github.com/openai/baselines/tree/master/baselines/ppo1
    Algorithm:
    for iteration = 1, 2, 3, ... do
        for actor = 1, 2, 3, ..., N do
            run policy pi_old in env for T timesteps
            compute advantage A_1, ..., A_T
        end for
        optimize surrogate L wrt theta, with K epochs and minibatch size M <= NT
    end for
    '''

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.init_nets()
        self.init_algo_params()
        logger.info(util.self_desc(self))

    @lab_api
    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        body = self.agent.nanflat_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        self.is_discrete = body.is_discrete
        net_spec = self.agent.spec['net']
        mem_spec = self.agent.spec['memory']
        net_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))

    @lab_api
    def init_algo_params(self):
        '''Initialize other algorithm parameters'''
        algorithm_spec = self.agent.spec['algorithm']
        net_spec = self.agent.spec['net']
        # Automatically selects appropriate discrete or continuous action policy if setting is default
        action_fn = algorithm_spec['action_policy']
        if action_fn == 'default':
            if self.is_discrete:
                self.action_policy = act_fns['softmax']
            else:
                self.action_policy = act_fns['gaussian']
        else:
            self.action_policy = act_fns[action_fn]
        util.set_attr(self, _.pick(algorithm_spec, [
            'gamma',
            'num_epis_to_collect',
            'add_entropy', 'entropy_weight',
            'continuous_action_clip'
        ]))
        util.set_attr(self, _.pick(net_spec, [
            'decay_lr', 'decay_lr_frequency', 'decay_lr_min_timestep',
        ]))
        # To save on a forward pass keep the log probs from each action
        self.saved_log_probs = []
        self.entropy = []
        self.to_train = 0

    @lab_api
    def body_act_discrete(self, body, state):
        return self.action_policy(self, state, body)

    @lab_api
    def body_act_continuous(self, body, state):
        return self.action_policy(self, state, body)

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample()
                   for body in self.agent.nanflat_body_a]
        batch = util.concat_dict(batches)
        batch = util.to_torch_nested_batch_ex_rewards(batch)
        return batch

    @lab_api
    def train(self):
        if self.to_train == 1:
            logger.debug2(f'Training...')
            # We only care about the rewards from the batch
            rewards = self.sample()['rewards']
            logger.debug3(f'Length first epi: {len(rewards[0])}')
            logger.debug3(f'Len log probs: {len(self.saved_log_probs)}')
            self.net.optim.zero_grad()
            policy_loss = self.get_policy_loss(rewards)
            loss = policy_loss.data[0]
            policy_loss.backward()
            if self.net.clamp_grad:
                logger.debug("Clipping gradient...")
                torch.nn.utils.clip_grad_norm(
                    self.net.parameters(), self.net.clamp_grad_val)
            logger.debug2(f'Gradient norms: {self.net.get_grad_norms()}')
            self.net.optim.step()
            self.to_train = 0
            self.saved_log_probs = []
            self.entropy = []
            logger.debug(f'Policy loss: {loss}')
            return loss
        else:
            return np.nan

    @lab_api
    def update(self):
        self.update_learning_rate()
        '''No update needed to explore var'''
        explore_var = np.nan
        return explore_var
