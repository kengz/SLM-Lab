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
            'action_policy',
            'action_policy_update',
            # explore_var is epsilon, tau or etc. depending on the action policy
            # these control the trade off between exploration and exploitaton
            'explore_var_start', 'explore_var_end', 'explore_anneal_epi',
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
        self.net = NetClass(self.net_spec, self, self.body.state_dim, self.body.action_dim)
        logger.info(f'Training on gpu: {self.net.gpu}')

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
        return pdparam

    @lab_api
    def body_act(self, body, state):
        '''Note, SARSA is discrete-only'''
        action, action_pd = self.action_policy(state, self, body)
        body.entropies.append(action_pd.entropy())
        body.log_probs.append(action_pd.log_prob(action.float()))
        if len(action.size()) == 0:  # scalar
            return action.numpy().astype(body.action_space.dtype)
        else:
            return action.numpy()

    def compute_q_target_values(self, batch):
        '''Computes the target Q values for a batch of experiences'''
        # TODO recheck if no_grad is proper; also use calc_pdparam
        with torch.no_grad():
            # Calculate the Q values of the current and next states
            q_sts = self.net.wrap_eval(batch['states'])
            q_next_st = self.net.wrap_eval(batch['next_states'])
            q_next_actions = batch['next_actions']
            logger.debug2(f'Q next states: {q_next_st.size()}')
            # Get the q value for the next action that was actually taken
            idx = torch.from_numpy(np.array(range(q_next_st.size(0))))
            if torch.cuda.is_available() and self.net.gpu:
                idx = idx.cuda()
            q_next_st_vals = q_next_st[idx, q_next_actions.squeeze_(1).data.long()]
            # Expand the dims so that q_next_st_vals can be broadcast
            q_next_st_vals.unsqueeze_(1)
            logger.debug2(f'Q next_states vals {q_next_st_vals.size()}')
            logger.debug3(f'Q next_states {q_next_st}')
            logger.debug3(f'Q next actions {q_next_actions}')
            logger.debug3(f'Q next_states vals {q_next_st_vals}')
            logger.debug3(f'Dones {batch["dones"]}')
            # Compute q_targets using reward and Q value corresponding to the action taken in the next state if there is one. Make next state Q value 0 if the current state is done
            q_targets_actual = batch['rewards'].data + self.gamma * torch.mul((1 - batch['dones'].data), q_next_st_vals)
            logger.debug2(f'Q targets actual: {q_targets_actual.size()}')
            logger.debug3(f'Q states {q_sts}')
            logger.debug3(f'Q targets actual: {q_targets_actual}')
            # We only want to train the network for the action selected in the current state
            # For all other actions we set the q_target = q_sts so that the loss for these actions is 0
            q_targets = torch.mul(q_targets_actual, batch['actions_onehot'].data) + torch.mul(q_sts, (1 - batch['actions_onehot'].data))
            logger.debug2(f'Q targets: {q_targets.size()}')
            logger.debug3(f'Q targets: {q_targets}')
            if torch.cuda.is_available() and self.net.gpu:
                q_targets = q_targets.cuda()
            return q_targets

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        batch = util.concat_dict(batches)
        if self.body.memory.is_episodic:
            util.to_torch_nested_batch(batch, self.net.gpu)
            # Add next action to batch
            batch['actions_onehot'] = []
            batch['next_actions'] = []
            for acts in batch['actions']:
                # The next actions are the actions shifted by one time step
                # For episodic training is does not matter that the action in the last state is set to zero since there is no corresponding next state. The Q target is just the reward received in the terminal state.
                next_acts = torch.zeros_like(acts)
                next_acts[:-1] = acts[1:]
                # Convert actions to one hot (both representations are needed for SARSA)
                acts_onehot = util.convert_to_one_hot(acts, self.body.action_dim, self.net.gpu)
                batch['actions_onehot'].append(acts_onehot)
                batch['next_actions'].append(next_acts)
            # Flatten the batch to train all at once
            batch = util.concat_episodes(batch)
        else:
            util.to_torch_batch(batch, self.net.gpu)
            # Batch only useful to train with if it has more than one element
            # Train function checks for this and skips training if batch is too small
            if batch['states'].size(0) > 1:
                batch['next_actions'] = torch.zeros_like(batch['actions'])
                batch['next_actions'][:-1] = batch['actions'][1:]
                batch['actions_onehot'] = util.convert_to_one_hot(batch['actions'], self.body.action_dim, self.net.gpu)
                batch_elems = ['states', 'actions', 'actions_onehot', 'rewards', 'dones', 'next_states', 'next_actions']
                for k in batch_elems:
                    if batch[k].dim() == 1:
                        batch[k].unsqueeze_(1)
                # If the last experience in the batch is not terminal the batch has to be shortened by one element since the algorithm does not yet have access to the next action taken for the final experience
                if batch['dones'].data[-1].int().eq_(0).cpu().numpy()[0]:
                    logger.debug(f'Popping last element')
                    for k in batch_elems:
                        batch[k] = batch[k][:-1]
        return batch

    @lab_api
    def train(self):
        '''
        Completes one training step for the agent if it is time to train.
        Otherwise this function does nothing.
        '''
        if self.to_train == 1:
            batch = self.sample()
            if batch['states'].size(0) < 2:
                logger.info(f'Batch too small to train with, skipping...')
                self.to_train = 0
                return np.nan
            q_targets = self.compute_q_target_values(batch)
            y = q_targets
            loss = self.net.training_step(batch['states'], y)
            self.to_train = 0
            logger.debug(f'loss {loss.item()}')
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        '''Update the agent after training'''
        for net in [self.net]:
            net.update_lr()
        explore_vars = [self.action_policy_update(self, body) for body in self.agent.nanflat_body_a]
        explore_var_a = self.nanflat_to_data_a('explore_var', explore_vars)
        return explore_var_a
