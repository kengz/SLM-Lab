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
from slm_lab.lib import logger
from torch import distributions
import numpy as np
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
            new_prob[torch.argmax(probs, dim=0)] = 1.0
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
        return torch.tensor([cat.logits for cat in self.categoricals])

    @property
    def probs(self):
        return torch.tensor([cat.probs for cat in self.categoricals])

    @property
    def param_shape(self):
        return torch.tensor([cat.param_shape for cat in self.categoricals])

    @property
    def mean(self):
        return torch.tensor([cat.mean for cat in self.categoricals])

    @property
    def variance(self):
        return torch.tensor([cat.variance for cat in self.categoricals])

    def sample(self, sample_shape=torch.Size()):
        return torch.tensor([cat.sample(sample_shape=sample_shape) for cat in self.categoricals])

    def log_prob(self, value):
        return torch.tensor([cat.log_prob(value[idx]) for idx, cat in enumerate(self.categoricals)])

    def entropy(self):
        return torch.tensor([cat.entropy() for cat in self.categoricals])

    def enumerate_support(self):
        return torch.tensor([cat.enumerate_support() for cat in self.categoricals])


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
    if torch.cuda.is_available() and algorithm.net_spec['gpu']:
        state = state.cuda()
    pdparam = algorithm.calc_pdparam(state, evaluate=False).squeeze_(dim=0)
    return ActionPD, pdparam, body


def sample_action_pd(ActionPD, pdparam, body):
    '''
    This uses the outputs from init_action_pd and an optionally augmented pdparam to construct a action_pd for sampling action
    @returns {tensor, distribution} action, action_pd A sampled action, and the prob. dist. used for sampling to enable calculations like kl, entropy, etc. later.
    '''
    if body.is_discrete:
        action_pd = ActionPD(logits=pdparam)
    else:
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
    action_pd = distributions.Uniform(low=torch.from_numpy(np.array(body.action_space.low)).float(), high=torch.from_numpy(np.array(body.action_space.high)).float())
    sample = body.action_space.sample()
    action = torch.tensor(sample)
    return action, action_pd


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
    ActionPD, pdparam, body = init_action_pd(state, algorithm, body)
    pdparam /= tau
    action, action_pd = sample_action_pd(ActionPD, pdparam, body)
    return action, action_pd


# multi-body policy with a single forward pass to calc pdparam

def multi_default(pdparam, algorithm, body_list):
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
        ActionPD = getattr(distributions, body.action_pdtype)
        action, action_pd = sample_action_pd(ActionPD, sub_pdparam, body)
        action_list.append(action)
        action_pd_a.append(action_pd)
    action_a = torch.tensor(action_list).unsqueeze_(dim=1)
    return action_a, action_pd_a


def multi_random(pdparam, algorithm, body_list):
    '''Apply random policy body-wise.'''
    pdparam.squeeze_(dim=0)
    action_list, action_pd_a = [], []
    for idx, body in body_list:
        action, action_pd = random(None, algorithm, body)
        action_list.append(action)
        action_pd_a.append(action_pd)
    action_a = torch.tensor(action_list).unsqueeze_(dim=1)
    return action_a, action_pd_a


def multi_epsilon_greedy(pdparam, algorithm, body_list):
    '''Apply epsilon-greedy policy body-wise'''
    assert len(pdparam) > 1 and len(pdparam) == len(body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
    action_list, action_pd_a = [], []
    for idx, sub_pdparam in enumerate(pdparam):
        body = body_list[idx]
        epsilon = body.explore_var
        if epsilon > np.random.rand():
            action, action_pd = random(None, algorithm, body)
        else:
            ActionPD = getattr(distributions, body.action_pdtype)
            action, action_pd = sample_action_pd(ActionPD, sub_pdparam, body)
        action_list.append(action)
        action_pd_a.append(action_pd)
    action_a = torch.tensor(action_list).unsqueeze_(dim=1)
    return action_a, action_pd_a


def multi_boltzmann(pdparam, algorithm, body_list):
    '''Apply Boltzmann policy body-wise'''
    # pdparam.squeeze_(dim=0)
    assert len(pdparam) > 1 and len(pdparam) == len(body_list), f'pdparam shape: {pdparam.shape}, bodies: {len(body_list)}'
    action_list, action_pd_a = [], []
    for idx, sub_pdparam in enumerate(pdparam):
        body = body_list[idx]
        tau = body.explore_var
        sub_pdparam /= tau
        ActionPD = getattr(distributions, body.action_pdtype)
        action, action_pd = sample_action_pd(ActionPD, sub_pdparam, body)
        action_list.append(action)
        action_pd_a.append(action_pd)
    action_a = torch.tensor(action_list).unsqueeze_(dim=1)
    return action_a, action_pd_a


# generic rate decay methods


def _linear_decay(start_val, end_val, anneal_step, step):
    '''Simple linear decay with annealing'''
    rise = end_val - start_val
    slope = rise / anneal_step
    val = max(slope * (step - 1) + start_val, end_val)
    return val


def _rate_decay(start_val, end_val, anneal_step, step, decay_rate=0.9, frequency=20.):
    '''Compounding rate decay that anneals in 20 decay iterations until anneal_step'''
    step_per_decay = anneal_step / frequency
    decay_step = step / step_per_decay
    val = max(np.power(decay_rate, decay_step) * start_val, end_val)
    return val


def _periodic_decay(start_val, end_val, anneal_step, step, frequency=60.):
    '''
    Linearly decaying sinusoid that decays in roughly 10 iterations until explore_anneal_epi
    Plot the equation below to see the pattern
    suppose sinusoidal decay, start_val = 1, end_val = 0.2, stop after 60 unscaled x steps
    then we get 0.2+0.5*(1-0.2)(1 + cos x)*(1-x/60)
    '''
    x_freq = frequency
    step_per_decay = anneal_step / x_freq
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
    epi = body.env.clock.get('epi')
    body.explore_var = fn(algorithm.explore_var_start, algorithm.explore_var_end, algorithm.explore_anneal_epi, epi)
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
