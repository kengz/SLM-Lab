from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
import torch

logger = logger.get_logger(__name__)


class Reinforce(Algorithm):
    '''
    Implementation of REINFORCE (Williams, 1992) with baseline for discrete or continuous actions http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
    Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    Algorithm:
        0. Collect n episodes of data
        1. At each timestep in an episode
            - Calculate the advantage of that timestep
            - Multiply the advantage by the negative of the log probability of the action taken
        2. Sum all the values above.
        3. Calculate the gradient of this value with respect to all of the parameters of the network
        4. Update the network parameters using the gradient

    e.g. algorithm_spec:
    "algorithm": {
        "name": "Reinforce",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "training_frequency": 1,
        "normalize_state": true
    }
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            explore_var_spec=None,
            entropy_coef_spec=None,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, REINFORCE does not have policy update; but in this implementation we have such option
            'explore_var_spec',
            'gamma',  # the discount factor
            'entropy_coef_spec',
            'training_frequency',
            'normalize_state',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.body.explore_var = self.explore_var_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.body.entropy_coef = self.entropy_coef_scheduler.start_val

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPNet'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''
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

    @lab_api
    def calc_pdparam(self, x, evaluate=True, net=None):
        '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        net = self.net if net is None else net
        if evaluate:
            pdparam = net.wrap_eval(x)
        else:
            net.train()
            pdparam = net(x)
        logger.debug(f'pdparam: {pdparam}')
        return pdparam

    @lab_api
    def act(self, state):
        body = self.body
        if self.normalize_state:
            state = policy_util.update_online_stats_and_normalize_state(body, state)
        action, action_pd = self.action_policy(state, self, body)
        # sum for single and multi-action
        body.entropies.append(action_pd.entropy().sum(dim=0))
        body.log_probs.append(action_pd.log_prob(action.float()).sum(dim=0))
        assert not torch.isnan(body.log_probs[-1])
        if len(action.shape) == 0:  # scalar
            return action.cpu().numpy().astype(body.action_space.dtype).item()
        else:
            return action.cpu().numpy()

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batch = self.body.memory.sample()
        if self.normalize_state:
            batch = policy_util.normalize_states_and_next_states(self.body, batch)
        batch = util.to_torch_batch(batch, self.net.device, self.body.memory.is_episodic)
        return batch

    @lab_api
    def train(self):
        if util.get_lab_mode() in ('enjoy', 'eval'):
            self.body.flush()
            return np.nan
        clock = self.body.env.clock
        if self.to_train == 1:
            batch = self.sample()
            loss = self.calc_policy_loss(batch)
            self.net.training_step(loss=loss, lr_clock=clock)
            # reset
            self.to_train = 0
            self.body.flush()
            logger.debug(f'Trained {self.name} at epi: {clock.get("epi")}, total_t: {clock.get("total_t")}, t: {clock.get("t")}, total_reward so far: {self.body.memory.total_reward}, loss: {loss:.8f}')

            return loss.item()
        else:
            return np.nan

    def calc_policy_loss(self, batch):
        '''Calculate the policy loss for a batch of data.'''
        # use simple returns as advs
        advs = math_util.calc_returns(batch, self.gamma)
        advs = math_util.standardize(advs)
        logger.debug(f'advs: {advs}')
        assert len(self.body.log_probs) == len(advs), f'batch_size of log_probs {len(self.body.log_probs)} vs advs: {len(advs)}'
        log_probs = torch.stack(self.body.log_probs)
        policy_loss = - log_probs * advs
        if self.entropy_coef_spec is not None:
            entropies = torch.stack(self.body.entropies)
            policy_loss += (-self.body.entropy_coef * entropies)
            # Store mean entropy for debug logging
            self.body.mean_entropy = torch.mean(torch.tensor(self.body.entropies)).item()
        policy_loss = torch.sum(policy_loss)
        logger.debug(f'Actor policy loss: {policy_loss:.4f}')
        return policy_loss

    @lab_api
    def update(self):
        net_util.try_store_grad_norm(self)
        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        if self.entropy_coef_spec is not None:
            self.body.entropy_coef = self.entropy_coef_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var
