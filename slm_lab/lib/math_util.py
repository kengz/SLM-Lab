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

def calc_returns(batch, gamma):
    '''
    Calculate the simple returns (full rollout) for advantage
    i.e. sum discounted rewards up till termination
    '''
    rewards = batch['rewards']
    is_tensor = torch.is_tensor(rewards)
    if is_tensor:
        assert not torch.isnan(rewards).any()
    else:
        assert not np.any(np.isnan(rewards))
    # handle epi-end, to not sum past current episode
    not_dones = 1 - batch['dones']
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


def calc_gammas(batch, gamma):
    '''Calculate the gammas to the right power for multiplication with rewards'''
    dones = batch['dones']
    news = torch.cat([torch.ones((1,), device=dones.device), dones[:-1]])
    gammas = torch.empty_like(news)
    cur_gamma = 1.0
    for t, new in enumerate(news):
        cur_gamma = new * 1.0 + (1 - new) * cur_gamma * gamma
        gammas[t] = cur_gamma
    return gammas


def calc_nstep_returns(batch, gamma, n, next_v_preds):
    '''
    Calculate the n-step returns for advantage
    see n-step return in: http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf
    i.e. for each timestep t:
        sum discounted rewards up till step n (0 to n-1 that is),
        then add v_pred for n as final term
    '''
    rets = batch['rewards'].clone()  # prevent mutation
    next_v_preds = next_v_preds.clone()  # prevent mutation
    nstep_rets = torch.zeros_like(rets) + rets
    cur_gamma = gamma
    for i in range(1, n):
        # Shift returns by one and zero last element of each episode
        rets[:-1] = rets[1:]
        rets *= (1 - batch['dones'])
        # Also shift V(s_t+1) so final terms use V(s_t+n)
        next_v_preds[:-1] = next_v_preds[1:]
        next_v_preds *= (1 - batch['dones'])
        # Accumulate return
        nstep_rets += cur_gamma * rets
        # Update current gamma
        cur_gamma *= cur_gamma
    # Add final terms. Note no next state if epi is done
    final_terms = cur_gamma * next_v_preds * (1 - batch['dones'])
    nstep_rets += final_terms
    return nstep_rets


def calc_gaes(rewards, v_preds, next_v_preds, gamma, lam):
    '''
    Calculate GAE
    See http://www.breloff.com/DeepRL-OnlineGAE/ for clear example.
    v_preds are values predicted for current states
    next_v_preds are values predicted for next states
    NOTE for standardization trick, do it out of here
    '''
    T = len(rewards)
    assert not torch.isnan(rewards).any()
    assert T == len(v_preds)
    gaes = torch.empty(T, dtype=torch.float32, device=v_preds.device)
    future_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v_preds[t] - v_preds[t]
        gaes[t] = future_gae = delta + gamma * lam * future_gae
    assert not torch.isnan(gaes).any(), f'GAE has nan: {gaes}'
    return gaes


def calc_q_value_logits(state_value, raw_advantages):
    mean_adv = raw_advantages.mean(dim=-1).unsqueeze_(dim=-1)
    return state_value + raw_advantages - mean_adv


def standardize(v):
    '''Method to standardize a rank-1 np array'''
    v_stdev = v.std()
    # guard nan std by setting to 0 and add small const
    v_stdev[v_stdev != v_stdev] = 0  # nan guard
    v_stdev += 1e-08  # division guard
    v_std = (v - v.mean()) / v_stdev
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
