from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
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
        "action_policy_update": "linear_decay",
        "explore_var_start": 1.5,
        "explore_var_end": 0.3,
        "explore_anneal_epi": 10,
        "gamma": 0.99,
        "training_frequency": 10
    }
    '''

    @lab_api
    def post_body_init(self):
        '''
        Initializes the part of algorithm needing a body to exist first. A body is a part of an Agent. Agents may have 1 to k bodies. Bodies do the acting in environments, and contain:
            - Memory (holding experiences obtained by acting in the environment)
            - State and action dimensions for an environment
            - Boolean var for if the action space is discrete
        '''
        self.body = self.agent.nanflat_body_a[0]  # single-body algo
        self.init_algorithm_params()
        self.init_nets()
        logger.info(util.self_desc(self))

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters.'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            action_policy_update='no_update',
            explore_var_start=np.nan,
            explore_var_end=np.nan,
            explore_anneal_epi=np.nan,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            'action_policy_update',
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_start',
            'explore_var_end',
            'explore_anneal_epi',
            'gamma',  # the discount factor
            'training_frequency',  # how often to train for batch training (once each training_frequency time steps)
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.action_policy_update = getattr(policy_util, self.action_policy_update)
        for body in self.agent.nanflat_body_a:
            body.explore_var = self.explore_var_start

    @lab_api
    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        if 'Recurrent' in self.net_spec['type']:
            self.net_spec.update(seq_len=self.net_spec['seq_len'])
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, self.body.state_dim, self.body.action_dim)
        self.net_names = ['net']
        self.post_init_nets()

    @lab_api
    def calc_pdparam(self, x, evaluate=True):
        '''
        To get the pdparam for action policy sampling, do a forward pass of the appropriate net, and pick the correct outputs.
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        if evaluate:
            pdparam = self.net.wrap_eval(x)
        else:
            self.net.train()
            pdparam = self.net(x)
        logger.debug(f'pdparam: {pdparam}')
        return pdparam

    @lab_api
    def body_act(self, body, state):
        '''Note, SARSA is discrete-only'''
        action, action_pd = self.action_policy(state, self, body)
        body.entropies.append(action_pd.entropy())
        body.log_probs.append(action_pd.log_prob(action.float()))
        assert not torch.isnan(body.log_probs[-1])
        if len(action.shape) == 0:  # scalar
            return action.cpu().numpy().astype(body.action_space.dtype).item()
        else:
            return action.cpu().numpy()

    def calc_q_targets(self, batch):
        '''Computes the target Q values for a batch of experiences'''
        q_preds = self.net.wrap_eval(batch['states'])
        next_q_preds = self.net.wrap_eval(batch['next_states'])
        action_idxs = batch['next_actions'].long()
        # Get the q value for the next action that was actually taken
        batch_size = len(batch['dones'])
        act_next_q_preds = next_q_preds[range(batch_size), action_idxs]
        # Bellman equation: compute max_q_targets using reward and max estimated Q values (0 if no next_state)
        act_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * act_next_q_preds
        act_q_targets.unsqueeze_(1)
        # To train only for action taken, set q_target = q_pred for action not taken so that loss is 0
        q_targets = (act_q_targets * batch['one_hot_actions']) + (q_preds * (1 - batch['one_hot_actions']))
        if torch.cuda.is_available() and self.net.gpu:
            q_targets = q_targets.cuda()
        logger.debug(f'q_targets: {q_targets}')
        return q_targets

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batches = []
        for body in self.agent.nanflat_body_a:
            body_batch = body.memory.sample()
            # one-hot actions to calc q_targets
            if body.is_discrete:
                body_batch['one_hot_actions'] = util.to_one_hot(body_batch['actions'], body.action_space.high)
            batches.append(body_batch)
        batch = util.concat_batches(batches)
        # this is safe for next_action at done since the calculated act_next_q_preds will be multiplied by (1 - batch['dones'])
        batch['next_actions'] = np.zeros_like(batch['actions'])
        batch['next_actions'][:-1] = batch['actions'][1:]
        batch = util.to_torch_batch(batch, self.net.gpu)
        return batch

    @lab_api
    def train(self):
        '''
        Completes one training step for the agent if it is time to train.
        Otherwise this function does nothing.
        '''
        if util.get_lab_mode() == 'enjoy':
            return np.nan
        if self.to_train == 1:
            batch = self.sample()
            with torch.no_grad():
                q_targets = self.calc_q_targets(batch)
            loss = self.net.training_step(batch['states'], q_targets)
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Loss: {loss}')
            self.last_loss = loss.item()
        return self.last_loss

    @lab_api
    def update(self):
        '''Update the agent after training'''
        space_clock = util.s_get(self, 'aeb_space.clock')
        for net in [self.net]:
            net.update_lr(space_clock)
        explore_vars = [self.action_policy_update(self, body) for body in self.agent.nanflat_body_a]
        explore_var_a = self.nanflat_to_data_a('explore_var', explore_vars)
        return explore_var_a
