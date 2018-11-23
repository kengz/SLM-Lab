'''
Action policy methods to sampling actions
Algorithm provides a `calc_pdparam` which takes a state and do a forward pass through its net,
and the pdparam is used to construct an action probability distribution as appropriate per the action type as indicated by the body
Then the prob. dist. is used to sample action.

The default form looks like:
```
ActionPD, pdparam, body = init_action_pd(state, algorithm, body)
action, action_pd = sample_action_pd(ActionPD, pdparam, body)
```

We can also augment pdparam before sampling - as in the case of Boltzmann sampling,
or do epsilon-greedy to use pdparam-sampling or random sampling.
'''
from slm_lab.lib import logger, util
from torch import distributions
import numpy as np
import pydash as ps
import torch

logger = logger.get_logger(__name__)


# probability distributions constraints for different action types; the first in the list is the default
ACTION_PDS = {
    'continuous': ['Normal', 'Beta', 'Gumbel', 'LogNormal'],
    'multi_continuous': ['MultivariateNormal'],
    'discrete': ['Categorical', 'Argmax'],
    'multi_discrete': ['MultiCategorical'],
    'multi_binary': ['Bernoulli'],
}


class Argmax(distributions.Categorical):
    '''
    Special distribution class for argmax sampling, where probability is always 1 for the argmax.
    NOTE although argmax is not a sampling distribution, this implementation is for API consistency.
    '''

    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            new_probs = torch.zeros_like(probs, dtype=torch.float)
            new_probs[torch.argmax(probs, dim=0)] = 1.0
            probs = new_probs
        elif logits is not None:
            new_logits = torch.full_like(logits, -1e8, dtype=torch.float)
            max_idx = torch.argmax(logits, dim=0)
            new_logits[max_idx] = logits[max_idx]
            logits = new_logits

        super(Argmax, self).__init__(probs=probs, logits=logits, validate_args=validate_args)


class MultiCategorical(distributions.Categorical):
    '''MultiCategorical as collection of Categoricals'''

    def __init__(self, probs=None, logits=None, validate_args=None):
        self.categoricals = []
        if probs is None:
            probs = [None] * len(logits)
        elif logits is None:
            logits = [None] * len(probs)
        else:
            raise ValueError('Either probs or logits must be None')

        for sub_probs, sub_logits in zip(probs, logits):
            categorical = distributions.Categorical(probs=sub_probs, logits=sub_logits, validate_args=validate_args)
            self.categoricals.append(categorical)

    @property
    def logits(self):
        return [cat.logits for cat in self.categoricals]

    @property
    def probs(self):
        return [cat.probs for cat in self.categoricals]

    @property
    def param_shape(self):
        return [cat.param_shape for cat in self.categoricals]

    @property
    def mean(self):
        return torch.stack([cat.mean for cat in self.categoricals])

    @property
    def variance(self):
        return torch.stack([cat.variance for cat in self.categoricals])

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([cat.sample(sample_shape=sample_shape) for cat in self.categoricals])

    def log_prob(self, value):
        return torch.stack([cat.log_prob(value[idx]) for idx, cat in enumerate(self.categoricals)])

    def entropy(self):
        return torch.stack([cat.entropy() for cat in self.categoricals])

    def enumerate_support(self):
        return [cat.enumerate_support() for cat in self.categoricals]


setattr(distributions, 'Argmax', Argmax)
setattr(distributions, 'MultiCategorical', MultiCategorical)


# base methods


def try_preprocess(state, algorithm, body, append=True):
    '''Try calling preprocess as implemented in body's memory to use for net input'''
    if hasattr(body.memory, 'preprocess_state'):
        state = body.memory.preprocess_state(state, append=append)
    # as float, and always as minibatch for net input
    state = torch.from_numpy(state).float().unsqueeze_(dim=0)
    return state


def cond_squeeze(out):
    '''Helper to squeeze output depending if it is tensor (discrete pdparam) or list of tensors (continuous pdparam of loc and scale)'''
    if isinstance(out, list):
        for out_t in out:
            out_t.squeeze_(dim=0)
    else:
        out.squeeze_(dim=0)
    return out


def init_action_pd(state, algorithm, body, append=True):
    '''
    Build the proper action prob. dist. to use for action sampling.
    state is passed through algorithm's net via calc_pdparam, which the algorithm must implement using its proper net.
    This will return body, ActionPD and pdparam to allow augmentation, e.g. applying temperature tau to pdparam for boltzmann.
    Then, output must be called with sample_action_pd(body, ActionPD, pdparam) to sample action.
    @returns {cls, tensor, *} ActionPD, pdparam, body
    '''
    pdtypes = ACTION_PDS[body.action_type]
    assert body.action_pdtype in pdtypes, f'Pdtype {body.action_pdtype} is not compatible/supported with action_type {body.action_type}. Options are: {ACTION_PDS[body.action_type]}'
    ActionPD = getattr(distributions, body.action_pdtype)

    state = try_preprocess(state, algorithm, body, append=append)
    state = state.to(algorithm.net.device)
    pdparam = algorithm.calc_pdparam(state, evaluate=False)
    return ActionPD, pdparam, body


def sample_action_pd(ActionPD, pdparam, body):
    '''
    This uses the outputs from init_action_pd and an optionally augmented pdparam to construct a action_pd for sampling action
    @returns {tensor, distribution} action, action_pd A sampled action, and the prob. dist. used for sampling to enable calculations like kl, entropy, etc. later.
    '''
    pdparam = cond_squeeze(pdparam)
    if body.is_discrete:
        action_pd = ActionPD(logits=pdparam)
    else:  # continuous outputs a list, loc and scale
        assert len(pdparam) == 2, pdparam
        # scale (stdev) must be >0, use softplus
        if pdparam[1] < 5:
            pdparam[1] = torch.log(1 + torch.exp(pdparam[1])) + 1e-8
        action_pd = ActionPD(*pdparam)
    action = action_pd.sample()
    return action, action_pd


# interface action sampling methods


def default(state, algorithm, body):
    '''Plain policy by direct sampling using outputs of net as logits and constructing ActionPD as appropriate'''
    ActionPD, pdparam, body = init_action_pd(state, algorithm, body)
    action, action_pd = sample_action_pd(ActionPD, pdparam, body)
    return action, action_pd


def random(state, algorithm, body):
    '''Random action sampling that returns the same data format as default(), but without forward pass. Uses gym.space.sample()'''
    state = try_preprocess(state, algorithm, body, append=True)  # for consistency with init_action_pd inner logic
    if body.action_type == 'discrete':
        action_pd = distributions.Categorical(logits=torch.ones(body.action_space.high, device=algorithm.net.device))
    elif body.action_type == 'continuous':
        # Possibly this should this have a 'device' set
        action_pd = distributions.Uniform(low=torch.tensor(body.action_space.low).float(),
                                          high=torch.tensor(body.action_space.high).float())
    elif body.action_type == 'multi_discrete':
        action_pd = distributions.Categorical(
            logits=torch.ones(body.action_space.high.size, body.action_space.high[0], device=algorithm.net.device))
    elif body.action_type == 'multi_continuous':
        raise NotImplementedError
    elif body.action_type == 'multi_binary':
        raise NotImplementedError
    else:
        raise NotImplementedError
    sample = body.action_space.sample()
    action = torch.tensor(sample, device=algorithm.net.device)
    return action, action_pd


def epsilon_greedy(state, algorithm, body):
    '''Epsilon-greedy policy: with probability epsilon, do random action, otherwise do default sampling.'''
    if util.get_lab_mode() == 'enjoy':
        return default(state, algorithm, body)
    epsilon = body.explore_var
    if epsilon > np.random.rand():
        return random(state, algorithm, body)
    else:
        return default(state, algorithm, body)


def boltzmann(state, algorithm, body):
    '''
    Boltzmann policy: adjust pdparam with temperature tau; the higher the more randomness/noise in action.
    '''
    if util.get_lab_mode() == 'enjoy':
        return default(state, algorithm, body)
    tau = body.explore_var
    ActionPD, pdparam, body = init_action_pd(state, algorithm, body)
    pdparam /= tau
    action, action_pd = sample_action_pd(ActionPD, pdparam, body)
    return action, action_pd


# multi-body policy with a single forward pass to calc pdparam

def multi_default(states, algorithm, body_list, pdparam):
    '''
    Apply default policy body-wise
    Note, for efficiency, do a single forward pass to calculate pdparam, then call this policy like:
    @example

    pdparam = self.calc_pdparam(state, evaluate=False)
    action_a, action_pd_a = self.action_policy(pdparam, self, body_list)
    '''
    pdparam.squeeze_(dim=0)
    # assert pdparam has been chunked
    assert len(pdparam.shape) > 1 and len(pdparam) == len(body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
    action_list, action_pd_a = [], []
    for idx, sub_pdparam in enumerate(pdparam):
        body = body_list[idx]
        try_preprocess(states[idx], algorithm, body, append=True)  # for consistency with init_action_pd inner logic
        ActionPD = getattr(distributions, body.action_pdtype)
        action, action_pd = sample_action_pd(ActionPD, sub_pdparam, body)
        action_list.append(action)
        action_pd_a.append(action_pd)
    action_a = torch.tensor(action_list, device=algorithm.net.device).unsqueeze_(dim=1)
    return action_a, action_pd_a


def multi_random(states, algorithm, body_list, pdparam):
    '''Apply random policy body-wise.'''
    pdparam.squeeze_(dim=0)
    action_list, action_pd_a = [], []
    for idx, body in body_list:
        action, action_pd = random(states[idx], algorithm, body)
        action_list.append(action)
        action_pd_a.append(action_pd)
    action_a = torch.tensor(action_list, device=algorithm.net.device).unsqueeze_(dim=1)
    return action_a, action_pd_a


def multi_epsilon_greedy(states, algorithm, body_list, pdparam):
    '''Apply epsilon-greedy policy body-wise'''
    assert len(pdparam) > 1 and len(pdparam) == len(body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
    if util.get_lab_mode() == 'enjoy':
        return multi_default(states, algorithm, body_list, pdparam)
    action_list, action_pd_a = [], []
    for idx, sub_pdparam in enumerate(pdparam):
        body = body_list[idx]
        epsilon = body.explore_var
        if epsilon > np.random.rand():
            action, action_pd = random(states[idx], algorithm, body)
        else:
            try_preprocess(states[idx], algorithm, body, append=True)  # for consistency with init_action_pd inner logic
            ActionPD = getattr(distributions, body.action_pdtype)
            action, action_pd = sample_action_pd(ActionPD, sub_pdparam, body)
        action_list.append(action)
        action_pd_a.append(action_pd)
    action_a = torch.tensor(action_list, device=algorithm.net.device).unsqueeze_(dim=1)
    return action_a, action_pd_a


def multi_boltzmann(states, algorithm, body_list, pdparam):
    '''Apply Boltzmann policy body-wise'''
    # pdparam.squeeze_(dim=0)
    assert len(pdparam) > 1 and len(pdparam) == len(body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
    if util.get_lab_mode() == 'enjoy':
        return multi_default(states, algorithm, body_list, pdparam)
    action_list, action_pd_a = [], []
    for idx, sub_pdparam in enumerate(pdparam):
        body = body_list[idx]
        try_preprocess(states[idx], algorithm, body, append=True)  # for consistency with init_action_pd inner logic
        tau = body.explore_var
        sub_pdparam /= tau
        ActionPD = getattr(distributions, body.action_pdtype)
        action, action_pd = sample_action_pd(ActionPD, sub_pdparam, body)
        action_list.append(action)
        action_pd_a.append(action_pd)
    action_a = torch.tensor(action_list, device=algorithm.net.device).unsqueeze_(dim=1)
    return action_a, action_pd_a


# generic rate decay methods


def _linear_decay(start_val, end_val, start_step, end_step, step):
    '''Simple linear decay with annealing'''
    if step < start_step:
        return start_val
    slope = (end_val - start_val) / (end_step - start_step)
    val = max(slope * step + start_val, end_val)
    return val


def _rate_decay(start_val, end_val, start_step, end_step, step, decay_rate=0.9, frequency=20.):
    '''Compounding rate decay that anneals in 20 decay iterations until end_step'''
    if step < start_step:
        return start_val
    step_per_decay = (end_step - start_step) / frequency
    decay_step = step / step_per_decay
    val = max(np.power(decay_rate, decay_step) * start_val, end_val)
    return val


def _periodic_decay(start_val, end_val, start_step, end_step, step, frequency=60.):
    '''
    Linearly decaying sinusoid that decays in roughly 10 iterations until explore_anneal_epi
    Plot the equation below to see the pattern
    suppose sinusoidal decay, start_val = 1, end_val = 0.2, stop after 60 unscaled x steps
    then we get 0.2+0.5*(1-0.2)(1 + cos x)*(1-x/60)
    '''
    if step < start_step:
        return start_val
    x_freq = frequency
    step_per_decay = (end_step - start_step) / x_freq
    x = step / step_per_decay
    unit = start_val - end_val
    val = end_val * 0.5 * unit * (1 + np.cos(x) * (1 - x / x_freq))
    val = max(val, end_val)
    return val


# action policy update methods

def no_update(algorithm, body):
    '''No update, but exists for API consistency'''
    return body.explore_var


def fn_decay_explore_var(algorithm, body, fn):
    '''Apply a function to decay explore_var'''
    spec = algorithm.explore_var_spec
    step = body.env.clock.get(spec['clock_unit'])
    body.explore_var = fn(spec['start_val'], spec['end_val'], spec['start_step'], spec['end_step'], step)
    return body.explore_var


def linear_decay(algorithm, body):
    '''Apply linear decay to explore_var'''
    return fn_decay_explore_var(algorithm, body, _linear_decay)


def rate_decay(algorithm, body):
    '''Apply _rate_decay to explore_var'''
    return fn_decay_explore_var(algorithm, body, _rate_decay)


def periodic_decay(algorithm, body):
    '''Apply _periodic_decay to explore_var'''
    return fn_decay_explore_var(algorithm, body, _periodic_decay)

# entropy coefficient decay methods
# currently only linear decay supported


def entropy_linear_decay(algorithm, body):
    '''Apply a function to decay entropy_coef'''
    epi = body.env.clock.get('epi')
    # Offset the start of the decay
    if epi < algorithm.entropy_anneal_start_epi:
        return body.entropy_coef
    else:
        epi_offset = epi - algorithm.entropy_anneal_start_epi
        body.entropy_coef = _linear_decay(algorithm.entropy_coef_start,
                                          algorithm.entropy_coef_end,
                                          algorithm.entropy_anneal_epi,
                                          epi_offset)
        return body.entropy_coef

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
        _action, action_pd = sample_action_pd(ActionPD, pdparam, body)
        log_probs.append(action_pd.log_prob(actions[idx].float()).sum(dim=0))
    log_probs = torch.stack(log_probs)
    assert not torch.isnan(log_probs).any(), f'log_probs: {log_probs}, \npdparams: {pdparams} \nactions: {actions}'
    logger.debug(f'log_probs: {log_probs}')
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
    if ("Atari" in body.memory.__class__.__name__):
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
    if ('Atari' in body.memory.__class__.__name__):
        # never normalize atari, it has its own normalization step
        logger.debug('skipping normalizing for Atari, already handled by preprocess')
        return state
    elif ('Replay' in body.memory.__class__.__name__) and has_preprocess:
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
    logger.debug(f'state: {state}')
    update_online_stats(body, state)
    state = normalize_state(body, state)
    logger.debug(f'normalized state: {state}')
    return state


def normalize_states_and_next_states(body, batch, episodic_flag=None):
    '''
    Convenience function for normalizing the states and next states in a batch of data
    '''
    logger.debug(f'states: {batch["states"]}')
    logger.debug(f'next states: {batch["next_states"]}')
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
    logger.debug(f'normalized states: {batch["states"]}')
    logger.debug(f'normalized next states: {batch["next_states"]}')
    return batch
