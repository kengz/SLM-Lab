# Action policy module
# Constructs action probability distribution used by agent to sample action and calculate log_prob, entropy, etc.
from gymnasium import spaces
# LazyFrames removed - modern gymnasium handles frame stacking efficiently
from slm_lab.lib import distribution, logger, math_util, util
from torch import distributions
import numpy as np
import torch

logger = logger.get_logger(__name__)

# register custom distributions
setattr(distributions, 'Argmax', distribution.Argmax)
setattr(distributions, 'GumbelSoftmax', distribution.GumbelSoftmax)
setattr(distributions, 'MultiCategorical', distribution.MultiCategorical)
# probability distributions constraints for different action types; the first in the list is the default
ACTION_PDS = {
    'continuous': ['Normal', 'Beta', 'Gumbel', 'LogNormal'],
    'multi_continuous': ['Normal', 'MultivariateNormal'],  # Normal treats dimensions independently (standard for SAC/PPO)
    'discrete': ['Categorical', 'Argmax', 'GumbelSoftmax'],
    'multi_discrete': ['MultiCategorical'],
    'multi_binary': ['Bernoulli'],
}


def get_action_type(env) -> str:
    '''Get action type for distribution selection using environment attributes'''
    if env.is_discrete:
        if isinstance(env.action_space, spaces.MultiBinary):
            return 'multi_binary'
        return 'multi_discrete' if env.is_multi else 'discrete'
    else:
        return 'multi_continuous' if env.is_multi else 'continuous'


# action_policy base methods


def reduce_multi_action(tensor):
    '''Reduce tensor across action dimensions for multi-dimensional continuous actions.
    Sum along last dim if >1D (continuous multi-action), otherwise return as-is.
    Used for log_prob and entropy which return per-action-dim values for Normal dist.
    '''
    return tensor.sum(dim=-1) if tensor.dim() > 1 else tensor


def get_action_pd_cls(action_pdtype, action_type):
    '''
    Verify and get the action prob. distribution class for construction
    Called by agent at init to set the agent's ActionPD
    '''
    pdtypes = ACTION_PDS[action_type]
    assert action_pdtype in pdtypes, f'Pdtype {action_pdtype} is not compatible/supported with action_type {action_type}. Options are: {pdtypes}'
    ActionPD = getattr(distributions, action_pdtype)
    return ActionPD


def guard_tensor(state, agent):
    '''Guard-cast tensor before being input to network'''
    # Modern gymnasium handles frame stacking efficiently, no LazyFrames needed
    if not isinstance(state, np.ndarray):
        state = np.array(state, dtype=np.float32)
    elif state.dtype != np.float32:
        state = state.astype(np.float32)
    state = torch.from_numpy(state)
    if not agent.env.is_venv:
        # singleton state, unsqueeze as minibatch for net input
        state = state.unsqueeze(dim=0)
    return state


def calc_pdparam(state, algorithm):
    '''
    Prepare the state and run algorithm.calc_pdparam to get pdparam for action_pd
    @param tensor:state For pdparam = net(state)
    @param algorithm The algorithm containing self.net and agent
    @returns tensor:pdparam
    @example

    pdparam = calc_pdparam(state, algorithm)
    action_pd = ActionPD(logits=pdparam)  # e.g. ActionPD is Categorical
    action = action_pd.sample()
    '''
    if not torch.is_tensor(state):  # dont need to cast from numpy
        state = guard_tensor(state, algorithm.agent)
        state = state.to(algorithm.net.device)
    pdparam = algorithm.calc_pdparam(state)
    return pdparam


def init_action_pd(ActionPD, pdparam):
    '''
    Initialize the action_pd for discrete or continuous actions:
    - discrete: action_pd = ActionPD(logits)
    - continuous: action_pd = ActionPD(loc, scale)
    '''
    args = ActionPD.arg_constraints
    if 'logits' in args:  # discrete
        # for relaxed discrete dist. with reparametrizable discrete actions
        pd_kwargs = {'temperature': torch.tensor(1.0)} if hasattr(ActionPD, 'temperature') else {}
        action_pd = ActionPD(logits=pdparam, **pd_kwargs)
    else:  # continuous, args = loc and scale
        if isinstance(pdparam, list):  # multi-dim actions from multi-head network
            loc, scale = pdparam
        else:  # 1D actions - single tensor of shape [batch, 2] for [loc, log_scale]
            loc, scale = pdparam.split(1, dim=-1)  # keeps [batch, 1] shape for sum(-1)
        # scale (stdev) must be > 0, log-clamp-exp (CleanRL standard: -5 to 2)
        scale = torch.clamp(scale, min=-5, max=2).exp()
        if 'covariance_matrix' in args:  # split output
            # construct covars from a batched scale tensor
            covars = torch.diag_embed(scale)
            action_pd = ActionPD(loc=loc, covariance_matrix=covars)
        else:
            action_pd = ActionPD(loc=loc, scale=scale)
    return action_pd


def sample_action(ActionPD, pdparam):
    '''
    Convenience method to sample action(s) from action_pd = ActionPD(pdparam)
    Works with batched pdparam too
    @returns tensor:action Sampled action(s)
    @example

    # policy contains:
    pdparam = calc_pdparam(state, algorithm)
    action = sample_action(algorithm.agent.ActionPD, pdparam)
    '''
    action_pd = init_action_pd(ActionPD, pdparam)
    action = action_pd.sample()
    return action


# action_policy used by agent


def default(state, algorithm) -> torch.Tensor:
    '''Plain policy by direct sampling from a default action probability defined by agent.ActionPD'''
    pdparam = calc_pdparam(state, algorithm)
    action = sample_action(algorithm.agent.ActionPD, pdparam)
    return action


def random(state, algorithm) -> torch.Tensor:
    '''Random action using gym.action_space.sample(), with the same format as default()'''
    if algorithm.agent.env.is_venv:
        _action = [algorithm.agent.action_space.sample() for _ in range(algorithm.agent.env.num_envs)]
    else:
        _action = [algorithm.agent.action_space.sample()]
    action = torch.from_numpy(np.array(_action))
    return action


def epsilon_greedy(state, algorithm):
    '''Epsilon-greedy policy: with probability epsilon, do random action, otherwise do greedy argmax.'''
    epsilon = algorithm.agent.explore_var
    if epsilon > np.random.rand():
        return random(state, algorithm)
    else:
        # Epsilon-greedy must use argmax (greedy), NOT stochastic sampling
        pdparam = calc_pdparam(state, algorithm)
        action = pdparam.argmax(dim=-1)
        return action


def boltzmann(state, algorithm):
    '''
    Boltzmann policy: adjust pdparam with temperature tau; the higher the more randomness/noise in action.
    '''
    tau = algorithm.agent.explore_var
    pdparam = calc_pdparam(state, algorithm)
    pdparam /= tau
    action = sample_action(algorithm.agent.ActionPD, pdparam)
    return action



# action policy update methods

class VarScheduler:
    '''
    Variable scheduler for decaying variables such as explore_var (epsilon, tau) and entropy

    e.g. spec
    "explore_var_spec": {
        "name": "linear_decay",
        "start_val": 1.0,
        "end_val": 0.1,
        "start_step": 0,
        "end_step": 800,
    },
    '''

    def __init__(self, var_decay_spec=None):
        self._updater_name = 'no_decay' if var_decay_spec is None else var_decay_spec['name']
        self._updater = getattr(math_util, self._updater_name)
        util.set_attr(self, dict(
            start_val=np.nan,
        ))
        util.set_attr(self, var_decay_spec, [
            'start_val',
            'end_val',
            'start_step',
            'end_step',
        ])
        if not getattr(self, 'end_val', None):
            self.end_val = self.start_val

    def update(self, algorithm, clock):
        '''Get an updated value for var'''
        if (util.in_eval_lab_mode()) or self._updater_name == 'no_decay':
            return self.end_val
        # Handle both old Clock objects and new ClockWrapper environments
        step = clock.get() if hasattr(clock, 'get') else clock.get('frame')
        val = self._updater(self.start_val, self.end_val, self.start_step, self.end_step, step)
        return val
