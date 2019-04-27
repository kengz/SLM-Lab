# Action policy module
# Constructs action probability distribution used by agent to sample action and calculate log_prob, entropy, etc.
from slm_lab.env.wrapper import LazyFrames
from slm_lab.lib import distribution, logger, math_util, util
from torch import distributions
import numpy as np
import pydash as ps
import torch

logger = logger.get_logger(__name__)

# register custom distributions
setattr(distributions, 'Argmax', distribution.Argmax)
setattr(distributions, 'GumbelCategorical', distribution.GumbelCategorical)
setattr(distributions, 'MultiCategorical', distribution.MultiCategorical)
# probability distributions constraints for different action types; the first in the list is the default
ACTION_PDS = {
    'continuous': ['Normal', 'Beta', 'Gumbel', 'LogNormal'],
    'multi_continuous': ['MultivariateNormal'],
    'discrete': ['Categorical', 'Argmax', 'GumbelCategorical'],
    'multi_discrete': ['MultiCategorical'],
    'multi_binary': ['Bernoulli'],
}


# action_policy base methods

def get_action_pd_cls(action_pdtype, action_type):
    '''
    Verify and get the action prob. distribution class for construction
    Called by body at init to set its own ActionPD
    '''
    pdtypes = ACTION_PDS[action_type]
    assert action_pdtype in pdtypes, f'Pdtype {action_pdtype} is not compatible/supported with action_type {action_type}. Options are: {pdtypes}'
    ActionPD = getattr(distributions, action_pdtype)
    return ActionPD


def try_preprocess(state, algorithm, body, append=True):
    '''Try calling preprocess as implemented in body's memory to use for net input'''
    if isinstance(state, LazyFrames):
        state = state.__array__()  # from global env preprocessor
    if hasattr(body.memory, 'preprocess_state'):
        state = body.memory.preprocess_state(state, append=append)
    state = torch.from_numpy(state).float()
    if not body.env.is_venv or util.in_eval_lab_modes():
        # singleton state, unsqueeze as minibatch for net input
        state = state.unsqueeze(dim=0)
    else:  # venv state at train is already batched = num_envs
        pass
    return state


def calc_pdparam(state, algorithm, body, append=True):
    '''
    Prepare the state and run algorithm.calc_pdparam to get pdparam for action_pd
    @param tensor:state For pdparam = net(state)
    @param algorithm The algorithm containing self.net
    @param body Body which links algorithm to the env which the action is for
    @returns tensor:pdparam
    @example

    pdparam = calc_pdparam(state, algorithm, body)
    action_pd = ActionPD(logits=pdparam)  # e.g. ActionPD is Categorical
    action = action_pd.sample()
    '''
    if not torch.is_tensor(state):  # dont need to cast from numpy
        state = try_preprocess(state, algorithm, body, append=append)
        state = state.to(algorithm.net.device)
    pdparam = algorithm.calc_pdparam(state)
    return pdparam


def init_action_pd(ActionPD, pdparam):
    '''
    Initialize the action_pd for discrete or continuous actions:
    - discrete: action_pd = ActionPD(logits)
    - continuous: action_pd = ActionPD(loc, scale)
    '''
    if 'logits' in ActionPD.arg_constraints:  # discrete
        action_pd = ActionPD(logits=pdparam)
    else:  # continuous, args = loc and scale
        assert len(pdparam) == 2, pdparam
        # scale (stdev) must be >0, use softplus
        if pdparam[1] < 5:
            pdparam[1] = torch.log(1 + torch.exp(pdparam[1])) + 1e-8
        action_pd = ActionPD(*pdparam)
    return action_pd


def sample_action(ActionPD, pdparam):
    '''
    Convenience method to sample action(s) from action_pd = ActionPD(pdparam)
    Works with batched pdparam too
    @returns tensor:action Sampled action(s)
    @example

    # policy contains:
    pdparam = calc_pdparam(state, algorithm, body)
    action = sample_action(body.ActionPD, pdparam)
    '''
    action_pd = init_action_pd(ActionPD, pdparam)
    action = action_pd.sample()
    return action


def calc_action_pd(state, algorithm, body):
    '''
    Do calc_pdparam from state and get action_pd to calc log_prob, entropy, etc.
    This is used for batched loss calculation for efficiency
    '''
    pdparam = calc_pdparam(state, algorithm, body)
    action_pd = init_action_pd(body.ActionPD, pdparam)
    return action_pd


# action_policy used by agent


def default(state, algorithm, body):
    '''Plain policy by direct sampling from a default action probability defined by body.ActionPD'''
    pdparam = calc_pdparam(state, algorithm, body)
    action = sample_action(body.ActionPD, pdparam)
    return action


def random(state, algorithm, body):
    '''Random action using gym.action_space.sample(), with the same format as default()'''
    if body.env.is_venv and not util.in_eval_lab_modes():
        _action = [body.action_space.sample() for _ in range(body.env.num_envs)]
    else:
        _action = body.action_space.sample()
    action = torch.tensor([_action])
    return action


def epsilon_greedy(state, algorithm, body):
    '''Epsilon-greedy policy: with probability epsilon, do random action, otherwise do default sampling.'''
    epsilon = body.explore_var
    if epsilon > np.random.rand():
        return random(state, algorithm, body)
    else:
        return default(state, algorithm, body)


def boltzmann(state, algorithm, body):
    '''
    Boltzmann policy: adjust pdparam with temperature tau; the higher the more randomness/noise in action.
    '''
    tau = body.explore_var
    pdparam = calc_pdparam(state, algorithm, body)
    pdparam /= tau
    action = sample_action(body.ActionPD, pdparam)
    return action


# multi-body action_policy used by agent

# TODO fix later using similar batch action method

def multi_default(states, algorithm, body_list, pdparam):
    '''
    Apply default policy body-wise
    Note, for efficiency, do a single forward pass to calculate pdparam, then call this policy like:
    @example

    pdparam = self.calc_pdparam(state)
    action_a = self.action_policy(pdparam, self, body_list)
    '''
    # assert pdparam has been chunked
    assert len(pdparam.shape) > 1 and len(pdparam) == len(body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
    action_list = []
    for idx, sub_pdparam in enumerate(pdparam):
        body = body_list[idx]
        try_preprocess(states[idx], algorithm, body, append=True)  # for consistency with singleton inner logic
        action = sample_action(body.ActionPD, sub_pdparam)
        action_list.append(action)
    action_a = torch.tensor(action_list, device=algorithm.net.device).unsqueeze(dim=1)
    return action_a


def multi_random(states, algorithm, body_list, pdparam):
    '''Apply random policy body-wise.'''
    action_list = []
    for idx, body in body_list:
        action = random(states[idx], algorithm, body)
        action_list.append(action)
    action_a = torch.tensor(action_list, device=algorithm.net.device).unsqueeze(dim=1)
    return action_a


def multi_epsilon_greedy(states, algorithm, body_list, pdparam):
    '''Apply epsilon-greedy policy body-wise'''
    assert len(pdparam) > 1 and len(pdparam) == len(body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
    action_list = []
    for idx, sub_pdparam in enumerate(pdparam):
        body = body_list[idx]
        epsilon = body.explore_var
        if epsilon > np.random.rand():
            action = random(states[idx], algorithm, body)
        else:
            try_preprocess(states[idx], algorithm, body, append=True)  # for consistency with singleton inner logic
            action = sample_action(body.ActionPD, sub_pdparam)
        action_list.append(action)
    action_a = torch.tensor(action_list, device=algorithm.net.device).unsqueeze(dim=1)
    return action_a


def multi_boltzmann(states, algorithm, body_list, pdparam):
    '''Apply Boltzmann policy body-wise'''
    assert len(pdparam) > 1 and len(pdparam) == len(body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
    action_list = []
    for idx, sub_pdparam in enumerate(pdparam):
        body = body_list[idx]
        try_preprocess(states[idx], algorithm, body, append=True)  # for consistency with singleton inner logic
        tau = body.explore_var
        sub_pdparam /= tau
        action = sample_action(body.ActionPD, sub_pdparam)
        action_list.append(action)
    action_a = torch.tensor(action_list, device=algorithm.net.device).unsqueeze(dim=1)
    return action_a


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
        if (util.in_eval_lab_modes()) or self._updater_name == 'no_decay':
            return self.end_val
        step = clock.get()
        val = self._updater(self.start_val, self.end_val, self.start_step, self.end_step, step)
        return val


# misc calc methods

def guard_multi_pdparams(pdparams, body):
    '''Guard pdparams for multi action'''
    action_dim = body.action_dim
    is_multi_action = ps.is_iterable(action_dim)
    if is_multi_action:
        assert ps.is_list(pdparams)
        pdparams = [t.clone() for t in pdparams]  # clone for grad safety
        assert len(pdparams) == len(action_dim), pdparams
        # transpose into (batch_size, [action_dims])
        pdparams = [list(torch.split(t, action_dim, dim=0)) for t in torch.cat(pdparams, dim=1)]
    return pdparams


def calc_log_probs(algorithm, net, body, batch):
    # TODO retire this
    '''
    Method to calculate log_probs fresh from batch data
    Body already stores log_prob from self.net. This is used for PPO where log_probs needs to be recalculated.
    '''
    states, actions = batch['states'], batch['actions']
    action_dim = body.action_dim
    is_multi_action = ps.is_iterable(action_dim)
    # construct log_probs for each state-action
    pdparams = algorithm.calc_pdparam(states, net=net)
    pdparams = guard_multi_pdparams(pdparams, body)
    assert len(pdparams) == len(states), f'batch_size of pdparams: {len(pdparams)} vs states: {len(states)}'

    pdtypes = ACTION_PDS[body.action_type]
    ActionPD = getattr(distributions, body.action_pdtype)

    log_probs = []
    for idx, pdparam in enumerate(pdparams):
        if not is_multi_action:  # already cloned  for multi_action above
            pdparam = pdparam.clone()  # clone for grad safety
        action_pd = init_action_pd(ActionPD, pdparam)
        log_probs.append(action_pd.log_prob(actions[idx].float()).sum(dim=0))
    log_probs = torch.stack(log_probs)
    assert not torch.isnan(log_probs).any(), f'log_probs: {log_probs}, \npdparams: {pdparams} \nactions: {actions}'
    return log_probs


def update_online_stats(body, state):
    '''
    Method to calculate the running mean and standard deviation of the state space.
    See https://www.johndcook.com/blog/standard_deviation/ for more details
    for n >= 1
        M_n = M_n-1 + (state - M_n-1) / n
        S_n = S_n-1 + (state - M_n-1) * (state - M_n)
        variance = S_n / (n - 1)
        std_dev = sqrt(variance)
    '''
    logger.debug(f'mean: {body.state_mean}, std: {body.state_std_dev}, num examples: {body.state_n}')
    # Assumes only one state is given
    if ('Atari' in util.get_class_name(body.memory)):
        assert state.ndim == 3
    elif getattr(body.memory, 'raw_state_dim', False):
        assert state.size == body.memory.raw_state_dim
    else:
        assert state.size == body.state_dim or state.shape == body.state_dim
    mean = body.state_mean
    body.state_n += 1
    if np.isnan(mean).any():
        assert np.isnan(body.state_std_dev_int)
        assert np.isnan(body.state_std_dev)
        body.state_mean = state
        body.state_std_dev_int = 0
        body.state_std_dev = 0
    else:
        assert body.state_n > 1
        body.state_mean = mean + (state - mean) / body.state_n
        body.state_std_dev_int = body.state_std_dev_int + (state - mean) * (state - body.state_mean)
        body.state_std_dev = np.sqrt(body.state_std_dev_int / (body.state_n - 1))
        # Guard against very small std devs
        if (body.state_std_dev < 1e-8).any():
            body.state_std_dev[np.where(body.state_std_dev < 1e-8)] += 1e-8
    logger.debug(f'new mean: {body.state_mean}, new std: {body.state_std_dev}, num examples: {body.state_n}')


def normalize_state(body, state):
    '''
    Normalizes one or more states using a running mean and standard deviation
    Details of the normalization from Deep RL Bootcamp, L6
    https://www.youtube.com/watch?v=8EcdaCk9KaQ&feature=youtu.be
    '''
    same_shape = False if type(state) == list else state.shape == body.state_mean.shape
    has_preprocess = getattr(body.memory, 'preprocess_state', False)
    if ('Atari' in util.get_class_name(body.memory)):
        # never normalize atari, it has its own normalization step
        logger.debug('skipping normalizing for Atari, already handled by preprocess')
        return state
    elif ('Replay' in util.get_class_name(body.memory)) and has_preprocess:
        # normalization handled by preprocess_state function in the memory
        logger.debug('skipping normalizing, already handled by preprocess')
        return state
    elif same_shape:
        # if not atari, always normalize the state the first time we see it during act
        # if the shape is not transformed in some way
        if np.sum(body.state_std_dev) == 0:
            return np.clip(state - body.state_mean, -10, 10)
        else:
            return np.clip((state - body.state_mean) / body.state_std_dev, -10, 10)
    else:
        # broadcastable sample from an un-normalized memory so we should normalize
        logger.debug('normalizing sample from memory')
        if np.sum(body.state_std_dev) == 0:
            return np.clip(state - body.state_mean, -10, 10)
        else:
            return np.clip((state - body.state_mean) / body.state_std_dev, -10, 10)


# TODO Not currently used, this will crash for more exotic memory structures
# def unnormalize_state(body, state):
#     '''
#     Un-normalizes one or more states using a running mean and new_std_dev
#     '''
#     return state * body.state_mean + body.state_std_dev


def update_online_stats_and_normalize_state(body, state):
    '''
    Convenience combination function for updating running state mean and std_dev and normalizing the state in one go.
    '''
    update_online_stats(body, state)
    state = normalize_state(body, state)
    return state


def normalize_states_and_next_states(body, batch, episodic_flag=None):
    '''
    Convenience function for normalizing the states and next states in a batch of data
    '''
    episodic = episodic_flag if episodic_flag is not None else body.memory.is_episodic
    logger.debug(f'Episodic: {episodic}, episodic_flag: {episodic_flag}, body.memory: {body.memory.is_episodic}')
    if episodic:
        normalized = []
        for epi in batch['states']:
            normalized.append(normalize_state(body, epi))
        batch['states'] = normalized
        normalized = []
        for epi in batch['next_states']:
            normalized.append(normalize_state(body, epi))
        batch['next_states'] = normalized
    else:
        batch['states'] = normalize_state(body, batch['states'])
        batch['next_states'] = normalize_state(body, batch['next_states'])
    return batch
