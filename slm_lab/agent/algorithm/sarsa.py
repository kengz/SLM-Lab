from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch

logger = logger.get_logger(__name__)


class SARSA(Algorithm):
    '''
    Implementation of SARSA.

    Algorithm:
    Repeat:
        1. Collect some examples by acting in the environment and store them in an on policy replay memory (either batch or episodic)
        2. For each example calculate the target (bootstrapped estimate of the discounted value of the state and action taken), y, using a neural network to approximate the Q function. s_t' is the next state following the action actually taken, a_t. a_t' is the action actually taken in the next state s_t'.
                y_t = r_t + gamma * Q(s_t', a_t')
        4. For each example calculate the current estimate of the discounted value of the state and action taken
                x_t = Q(s_t, a_t)
        5. Calculate L(x, y) where L is a regression loss (eg. mse)
        6. Calculate the gradient of L with respect to all the parameters in the network and update the network parameters using the gradient

    e.g. algorithm_spec
    "algorithm": {
        "name": "SARSA",
        "action_pdtype": "default",
        "action_policy": "boltzmann",
        "explore_var_spec": {
            "name": "linear_decay",
            "start_val": 1.0,
            "end_val": 0.1,
            "start_step": 10,
            "end_step": 1000,
        },
        "gamma": 0.99,
        "training_frequency": 10,
    }
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters.'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            explore_var_spec=None,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_spec',
            'gamma',  # the discount factor
            'training_frequency',  # how often to train for batch training (once each training_frequency time steps)
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        super().init_algorithm_params()  # Initialize common schedulers

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize the neural network used to learn the Q function from the spec'''
        if 'Recurrent' in self.net_spec['type']:
            self.net_spec.update(seq_len=self.net_spec['seq_len'])
        in_dim = self.agent.state_dim
        out_dim = net_util.get_out_dim(self.agent)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim)
        self.net_names = ['net']
        # init net optimizer and its lr scheduler
        # steps_per_schedule: frames processed per scheduler.step() call
        steps_per_schedule = self.training_frequency * self.agent.env.num_envs
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec, steps_per_schedule)
        net_util.set_global_nets(self, global_nets)
        self.end_init_nets()

    @lab_api
    def calc_pdparam(self, x, net=None):
        '''
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        net = self.net if net is None else net
        pdparam = net(x)
        return pdparam

    @lab_api
    def act(self, state):
        '''Note, SARSA is discrete-only'''
        action = self.action_policy(state, self)
        return self.to_action(action)

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batch = self.agent.memory.sample()
        # this is safe for next_action at done since the calculated act_next_q_preds will be multiplied by (1 - batch['terminateds'])
        batch['next_actions'] = np.zeros_like(batch['actions'])
        batch['next_actions'][:-1] = batch['actions'][1:]
        batch = util.to_torch_batch(batch, self.net.device, self.agent.memory.is_episodic)
        return batch

    def calc_q_loss(self, batch):
        '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
        states = batch['states']
        next_states = batch['next_states']
        if self.agent.env.is_venv:
            states = math_util.venv_unpack(states)
            next_states = math_util.venv_unpack(next_states)
        q_preds = self.net(states)
        with torch.no_grad():
            next_q_preds = self.net(next_states)
        if self.agent.env.is_venv:
            q_preds = math_util.venv_pack(q_preds, self.agent.env.num_envs)
            next_q_preds = math_util.venv_pack(next_q_preds, self.agent.env.num_envs)
        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        act_next_q_preds = next_q_preds.gather(-1, batch['next_actions'].long().unsqueeze(-1)).squeeze(-1)
        act_q_targets = batch['rewards'] + self.gamma * (1 - batch['terminateds']) * act_next_q_preds
        logger.debug(f'act_q_preds: {act_q_preds}\nact_q_targets: {act_q_targets}')
        q_loss = self.net.loss_fn(act_q_preds, act_q_targets)
        return q_loss

    @lab_api
    def train(self):
        '''
        Completes one training step for the agent if it is time to train.
        Otherwise this function does nothing.
        '''
        if self.to_train == 1:
            batch = self.sample()
            self.agent.env.set_batch_size(len(batch))
            loss = self.calc_q_loss(batch)
            self.net.train_step(loss, self.optim, self.lr_scheduler, global_net=self.global_net)
            self.agent.env.tick_opt_step()
            # Step LR scheduler once per training iteration
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # reset
            self.to_train = 0
            logger.debug(f'Trained {self.name} at epi: {self.agent.env.get("epi")}, frame: {self.agent.env.get("frame")}, t: {self.agent.env.get("t")}, total_reward so far: {self.agent.env.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        '''Update the agent after training'''
        self.agent.explore_var = self.explore_var_scheduler.update(self, self.agent.env)
        return self.agent.explore_var
