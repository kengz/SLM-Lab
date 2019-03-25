'''
Calculations used by algorithms
All calculations for training shall have a standard API that takes in `batch` from algorithm.sample() method and return np array for calculation.
`batch` is a dict containing keys to any data type you wish, e.g. {rewards: np.array([...])}
'''
from slm_lab.lib import logger
import numpy as np
import torch

logger = logger.get_logger(__name__)


# Policy Gradient calc
# advantage functions

def calc_returns(rewards, dones, gamma):
    '''
    Calculate the simple returns (full rollout) for advantage
    i.e. sum discounted rewards up till termination
    '''
    is_tensor = torch.is_tensor(rewards)
    if is_tensor:
        assert not torch.isnan(rewards).any()
    else:
        assert not np.any(np.isnan(rewards))
    # handle epi-end, to not sum past current episode
    not_dones = 1 - dones
    T = len(rewards)
    if is_tensor:
        rets = torch.empty(T, dtype=torch.float32, device=rewards.device)
    else:
        rets = np.empty(T, dtype='float32')
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = rewards[t] + gamma * future_ret * not_dones[t]
        rets[t] = future_ret
    return rets


def calc_nstep_returns(rewards, dones, gamma, n, next_v_preds):
    '''
    Calculate the n-step returns for advantage
    see n-step return in: http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf
    i.e. for each timestep t:
        sum discounted rewards up till step n (0 to n-1 that is),
        then add v_pred for n as final term
    '''
    rets = rewards.clone()  # prevent mutation
    next_v_preds = next_v_preds.clone()  # prevent mutation
    nstep_rets = torch.zeros_like(rets) + rets
    cur_gamma = gamma
    not_dones = 1 - dones
    for i in range(1, n):
        # TODO shifting is expensive. rewrite
        # Shift returns by one and zero last element of each episode
        rets[:-1] = rets[1:]
        rets *= not_dones
        # Also shift V(s_t+1) so final terms use V(s_t+n)
        next_v_preds[:-1] = next_v_preds[1:]
        next_v_preds *= not_dones
        # Accumulate return
        nstep_rets += cur_gamma * rets
        # Update current gamma
        cur_gamma *= cur_gamma
    # Add final terms. Note no next state if epi is done
    final_terms = cur_gamma * next_v_preds * not_dones
    nstep_rets += final_terms
    return nstep_rets


def calc_gaes(rewards, dones, v_preds, gamma, lam):
    '''
    Calculate GAE from Schulman et al. https://arxiv.org/pdf/1506.02438.pdf
    v_preds are values predicted for current states, with one last element as the final next_state
    delta is defined as r + gamma * V(s') - V(s) in eqn 10
    GAE is defined in eqn 16
    This method computes in torch tensor to prevent unnecessary moves between devices (e.g. GPU tensor to CPU numpy)
    NOTE any standardization is done outside of this method
    '''
    T = len(rewards)
    assert not torch.isnan(rewards).any()
    assert T + 1 == len(v_preds)  # v_preds includes states and 1 last next_state
    gaes = torch.empty(T, dtype=torch.float32, device=v_preds.device)
    future_gae = 0.0  # this will autocast to tensor below
    # to multiply with not_dones to handle episode boundary (last state has no V(s'))
    not_dones = 1 - dones
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * v_preds[t + 1] * not_dones[t] - v_preds[t]
        gaes[t] = future_gae = delta + gamma * lam * not_dones[t] * future_gae
    assert not torch.isnan(gaes).any(), f'GAE has nan: {gaes}'
    return gaes


def calc_q_value_logits(state_value, raw_advantages):
    mean_adv = raw_advantages.mean(dim=-1).unsqueeze(dim=-1)
    return state_value + raw_advantages - mean_adv


def calc_shaped_rewards(rewards, dones, v_pred, gamma):
    '''
    Trick from OpenAI baselines to shape rewards for training
    https://github.com/openai/baselines/blob/master/baselines/a2c/runner.py#L60-L63
    This prevents near-instant policy collapse when training policy-methods on sparse rewards,
    in which the sparse signals cause quick saturation of network output policy to have extreme unchanging probabilities
    This method computes in torch tensor to prevent unnecessary moves between devices (e.g. GPU tensor to CPU numpy)
    '''
    T = len(rewards)
    assert not torch.isnan(rewards).any()
    shaped_rewards = torch.empty(T, dtype=torch.float32, device=rewards.device)
    # set bootstrapped reward to v_pred if not done, else 0
    shaped_reward = v_pred if dones[-1].item() == 0.0 else 0.0
    not_dones = 1 - dones
    for t in reversed(range(T)):
        shaped_reward = rewards[t] + gamma * shaped_reward * not_dones[t]
        shaped_rewards[t] = shaped_reward
    return shaped_rewards


def standardize(v):
    '''Method to standardize a rank-1 np array'''
    assert len(v) > 1, 'Cannot standardize vector of size 1'
    v_std = (v - v.mean()) / (v.std() + 1e-08)
    return v_std


def normalize(v):
    '''Method to normalize a rank-1 np array'''
    v_min = v.min()
    v_max = v.max()
    v_range = v_max - v_min
    v_range += 1e-08  # division guard
    v_norm = (v - v_min) / v_range
    return v_norm


# generic variable decay methods

def no_decay(start_val, end_val, start_step, end_step, step):
    '''dummy method for API consistency'''
    return start_val


def linear_decay(start_val, end_val, start_step, end_step, step):
    '''Simple linear decay with annealing'''
    if step < start_step:
        return start_val
    slope = (end_val - start_val) / (end_step - start_step)
    val = max(slope * (step - start_step) + start_val, end_val)
    return val


def rate_decay(start_val, end_val, start_step, end_step, step, decay_rate=0.9, frequency=20.):
    '''Compounding rate decay that anneals in 20 decay iterations until end_step'''
    if step < start_step:
        return start_val
    if step >= end_step:
        return end_val
    step_per_decay = (end_step - start_step) / frequency
    decay_step = (step - start_step) / step_per_decay
    val = max(np.power(decay_rate, decay_step) * start_val, end_val)
    return val


def periodic_decay(start_val, end_val, start_step, end_step, step, frequency=60.):
    '''
    Linearly decaying sinusoid that decays in roughly 10 iterations until explore_anneal_epi
    Plot the equation below to see the pattern
    suppose sinusoidal decay, start_val = 1, end_val = 0.2, stop after 60 unscaled x steps
    then we get 0.2+0.5*(1-0.2)(1 + cos x)*(1-x/60)
    '''
    if step < start_step:
        return start_val
    if step >= end_step:
        return end_val
    x_freq = frequency
    step_per_decay = (end_step - start_step) / x_freq
    x = (step - start_step) / step_per_decay
    unit = start_val - end_val
    val = end_val * 0.5 * unit * (1 + np.cos(x) * (1 - x / x_freq))
    val = max(val, end_val)
    return val


# misc math methods

def is_outlier(points, thres=3.5):
    '''
    Detects outliers using MAD modified_z_score method, generalized to work on points.
    From https://stackoverflow.com/a/22357811/3865298
    @example

    is_outlier([1, 1, 1])
    # => array([False, False, False], dtype=bool)
    is_outlier([1, 1, 2])
    # => array([False, False,  True], dtype=bool)
    is_outlier([[1, 1], [1, 1], [1, 2]])
    # => array([False, False,  True], dtype=bool)
    '''
    points = np.array(points)
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    with np.errstate(divide='ignore', invalid='ignore'):
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thres


def to_one_hot(data, max_val):
    '''Convert an int list of data into one-hot vectors'''
    return np.eye(max_val)[np.array(data)]
