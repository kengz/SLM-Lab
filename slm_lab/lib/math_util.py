'''
Calculations used by algorithms
All calculations for training shall have a standard API that takes in `batch` from algorithm.sample() method and return np array for calculation.
`batch` is a dict containing keys to any data type you wish, e.g. {rewards: np.array([...])}
'''
from slm_lab.lib import logger
import numpy as np
import torch

logger = logger.get_logger(__name__)


# general math methods


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


def nan_add(a1, a2):
    '''Add np arrays and reset any nan to 0. Used for adding total_reward'''
    a1_isnan = np.isnan(a1)
    if a1_isnan.all():
        return a2
    else:
        if a1_isnan.any():  # reset nan to 0 pre-sum
            a1 = np.nan_to_num(a1)
        a12 = a1 + a2
        if np.isnan(a12).any():  # reset nan to 0 post-sum
            a12 = np.nan_to_num(a12)
        return a12


def normalize(v):
    '''Method to normalize a rank-1 np array'''
    v_min = v.min()
    v_max = v.max()
    v_range = v_max - v_min
    v_range += 1e-08  # division guard
    v_norm = (v - v_min) / v_range
    return v_norm


def standardize(v):
    '''Method to standardize a rank-1 np array'''
    assert len(v) > 1, 'Cannot standardize vector of size 1'
    v_std = (v - v.mean()) / (v.std() + 1e-08)
    return v_std


def to_one_hot(data, max_val):
    '''Convert an int list of data into one-hot vectors'''
    return np.eye(max_val)[np.array(data)]


def venv_pack(batch_tensor, num_envs):
    '''Apply the reverse of venv_unpack to pack a batch tensor from (b*num_envs, *shape) to (b, num_envs, *shape)'''
    shape = list(batch_tensor.shape)
    if len(shape) < 2:  # scalar data (b, num_envs,)
        return batch_tensor.view(-1, num_envs)
    else:  # non-scalar data (b, num_envs, *shape)
        pack_shape = [-1, num_envs] + shape[1:]
        return batch_tensor.view(pack_shape)


def venv_unpack(batch_tensor):
    '''
    Unpack a sampled vec env batch tensor
    e.g. for a state with original shape (4, ), vec env should return vec state with shape (num_envs, 4) to store in memory
    When sampled with batch_size b, we should get shape (b, num_envs, 4). But we need to unpack the num_envs dimension to get (b * num_envs, 4) for passing to a network. This method does that.
    '''
    shape = list(batch_tensor.shape)
    if len(shape) < 3:  # scalar data (b, num_envs,)
        return batch_tensor.view(-1)
    else:  # non-scalar data (b, num_envs, *shape)
        unpack_shape = [-1] + shape[2:]
        return batch_tensor.view(unpack_shape)


# Policy Gradient calc
# advantage functions

def calc_returns(rewards, dones, gamma):
    '''
    Calculate the simple returns (full rollout) i.e. sum discounted rewards up till termination
    '''
    T = len(rewards)
    rets = torch.zeros_like(rewards)
    future_ret = torch.tensor(0.0, dtype=rewards.dtype)
    not_dones = 1 - dones
    for t in reversed(range(T)):
        rets[t] = future_ret = rewards[t] + gamma * future_ret * not_dones[t]
    return rets


def calc_nstep_returns(rewards, dones, next_v_pred, gamma, n):
    '''
    Calculate the n-step returns for advantage. Ref: http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf
    Also see Algorithm S3 from A3C paper https://arxiv.org/pdf/1602.01783.pdf for the calculation used below
    R^(n)_t = r_{t} + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1} + gamma^(n) V(s_{t+n})
    '''
    rets = torch.zeros_like(rewards)
    future_ret = next_v_pred
    not_dones = 1 - dones
    for t in reversed(range(n)):
        rets[t] = future_ret = rewards[t] + gamma * future_ret * not_dones[t]
    return rets


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
    assert T + 1 == len(v_preds)  # v_preds includes states and 1 last next_state
    gaes = torch.zeros_like(rewards)
    future_gae = torch.tensor(0.0, dtype=rewards.dtype)
    # to multiply with not_dones to handle episode boundary (last state has no V(s'))
    not_dones = 1 - dones
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * v_preds[t + 1] * not_dones[t] - v_preds[t]
        gaes[t] = future_gae = delta + gamma * lam * not_dones[t] * future_gae
    return gaes


def calc_q_value_logits(state_value, raw_advantages):
    mean_adv = raw_advantages.mean(dim=-1).unsqueeze(dim=-1)
    return state_value + raw_advantages - mean_adv


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
