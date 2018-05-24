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

# TODO rename, also mandate definition on all algo and network usage


# probability distributions constraints for different action types; the first in the list is the default
ACTION_PDS = {
    'continuous': ['Normal', 'Beta', 'Gumbel', 'LogNormal'],
    'multi_continuous': ['MultivariateNormal'],
    'discrete': ['Categorical'],
    # TODO create MultiCategorical class by extending
    # 'multi_discrete': ['MultiCategorical'],
    'multi_binary': ['Bernoulli'],
}


# base methods

def init_action_pd(state, algorithm, body):
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

    state = torch.from_numpy(state).float().unsqueeze_(dim=0)
    if torch.cuda.is_available() and algorithm.net_spec['gpu']:
        state = state.cuda()
    pdparam = algorithm.calc_pdparam(state).squeeze_(dim=0)
    return ActionPD, pdparam, body


def sample_action_pd(ActionPD, pdparam, body):
    '''
    This uses the outputs from init_action_pd and an optionally augmented pdparam to construct a action_pd for sampling action
    @returns {numpy_data, distribution} action, action_pd A sampled action, and the prob. dist. used for sampling to enable calculations like kl, entropy, etc. later.
    '''
    if body.is_discrete:
        action_pd = ActionPD(logits=pdparam)
    else:
        action_pd = ActionPD(*pdparam)
    action = action_pd.sample().numpy()
    return action, action_pd


# interface action sampling methods

def default(state, algorithm, body):
    '''Plain policy by direct sampling using outputs of net as logits and constructing ActionPD as appropriate'''
    ActionPD, pdparam, body = init_action_pd(state, algorithm, body)
    action, action_pd = sample_action_pd(ActionPD, pdparam, body)
    return action, action_pd


def epsilon_greedy(state, algorithm, body):
    '''Epsilon-greedy policy: with probability epsilon, do random action, otherwise do default sampling.'''
    epsilon = body.explore_var
    if epsilon > np.random.rand():
        action_pd = None
        return body.action_space.sample(), action_pd
    else:
        return default()


def boltzmann(state, algorithm, body):
    '''
    Boltzmann policy: adjust pdparam with temperature tau; the higher the more randomness/noise in action.
    '''
    tau = body.explore_var
    ActionPD, pdparam, body = init_action_pd(state, algorithm, body)
    pdparam /= tau
    action, action_pd = sample_action_pd(ActionPD, pdparam, body)
    return action, action_pd


# action policy update methods

def linear_decay(algoritm, body):
    '''Simple linear decay with annealing'''
    epi = body.env.clock.get('epi')
    rise = algorithm.explore_var_end - algorithm.explore_var_start
    slope = rise / float(algorithm.explore_anneal_epi)
    explore_var = max(slope * (epi - 1) + algorithm.explore_var_start, algorithm.explore_var_end)
    body.explore_var = explore_var
    return explore_var


def rate_decay(algoritm, body):
    epi = body.env.clock.get('epi')
    epi_per_decay = algorithm.explore_anneal_epi / 20.
    decay_step = epi / epi_per_decay
    explore_var = max(0.9 ^ decay_step * body.explore_var, algorithm.explore_var_end)
    body.explore_var = explore_var
    return explore_var


def periodic_decay(algorithm, body):
    '''
    linearly decaying sinusoid
    plot this graph to see the pattern
    suppose sinusoidal decay, explore_var_start =1, explore_var_end = 0.2, stop after 60 unscaled x steps
    then we get 0.2+0.5*(1-0.2)(1 + cos x)*(1-x/60)
    '''
    epi = body.env.clock.get('epi')
    unscaled_x = 60.
    epi_per_decay = algorithm.explore_anneal_epi / unscaled_x
    x = epi / epi_per_decay
    unit = (algorithm.explore_var_start - algorithm.explore_var_end)
    explore_var = algorithm.explore_var_end * 0.5 * unit * (1 + np.cos(x) * (1 - x / unscaled_x))
    explore_var = max(explore_var, algorithm.explore_var_end)
    body.explore_var = explore_var
    return explore_var
