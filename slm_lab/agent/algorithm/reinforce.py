from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np

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
    }
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            center_return=False,
            explore_var_spec=None,
            entropy_coef_spec=None,
            policy_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            'center_return',  # center by the mean
            'explore_var_spec',
            'gamma',  # the discount factor
            'entropy_coef_spec',
            'policy_loss_coef',
            'training_frequency',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.agent.explore_var = self.explore_var_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.agent.entropy_coef = self.entropy_coef_scheduler.start_val
            self.agent.mt.register_algo_var('entropy_coef', self.agent)
            self.agent.mt.register_algo_var('entropy', self.agent)

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPNet'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''
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
        '''The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.'''
        net = self.net if net is None else net
        pdparam = net(x)
        return pdparam

    @lab_api
    def act(self, state):
        action = self.action_policy(state, self)
        return self.to_action(action)

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batch = self.agent.memory.sample()
        batch = util.to_torch_batch(batch, self.net.device, self.agent.memory.is_episodic)
        return batch

    def calc_pdparam_batch(self, batch):
        '''Efficiently forward to get pdparam and by batch for loss computation'''
        states = batch['states']
        if self.agent.env.is_venv:
            states = math_util.venv_unpack(states)
        pdparam = self.calc_pdparam(states)
        return pdparam

    def calc_ret_advs(self, batch):
        '''Calculate plain returns; which is generalized to advantage in ActorCritic'''
        rets = math_util.calc_returns(batch['rewards'], batch['terminateds'], self.gamma)
        if self.center_return:
            rets = math_util.center_mean(rets)
        advs = rets
        if self.agent.env.is_venv:
            advs = math_util.venv_unpack(advs)
        logger.debug(f'advs: {advs}')
        return advs

    def calc_policy_loss(self, batch, pdparams, advs):
        '''Calculate the actor's policy loss'''
        action_pd = policy_util.init_action_pd(self.agent.ActionPD, pdparams)
        actions = batch['actions']
        if self.agent.env.is_venv:
            actions = math_util.venv_unpack(actions)
        log_probs = policy_util.reduce_multi_action(action_pd.log_prob(actions))
        advs = advs.view(-1)  # Ensure advs is 1D to match log_probs shape
        # Normalize advantages (like PPO) for more stable gradient updates
        if len(advs) > 1:
            advs = math_util.standardize(advs)
        policy_loss = - self.policy_loss_coef * (log_probs * advs).mean()
        if self.entropy_coef_spec:
            entropy = policy_util.reduce_multi_action(action_pd.entropy()).mean()
            self.agent.entropy = entropy.detach()  # Update value for logging
            policy_loss += (-self.agent.entropy_coef * entropy)
        logger.debug(f'Actor policy loss: {policy_loss:g}')
        return policy_loss

    @lab_api
    def train(self):
        if self.to_train == 1:
            batch = self.sample()
            self.agent.env.set_batch_size(len(batch))
            pdparams = self.calc_pdparam_batch(batch)
            advs = self.calc_ret_advs(batch)
            loss = self.calc_policy_loss(batch, pdparams, advs)
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
        self.agent.explore_var = self.explore_var_scheduler.update(self, self.agent.env)
        if self.entropy_coef_spec is not None:
            self.agent.entropy_coef = self.entropy_coef_scheduler.update(self, self.agent.env)
        return self.agent.explore_var
