'''
Calculations used by algorithms
All calculations for training shall have a standard API that takes in `batch` from algorithm.sample() method and return np array for calculation.
`batch` is a dict containing keys to any data type you wish, e.g. {rewards: np.array([...])}
'''
from slm_lab.lib import logger
import numpy as np
import torch
import pydash as ps

# Policy Gradient calc

# TODO standardize arg with full sampled info


def is_episodic(batch):
    '''
    Check if batch is episodic or is plain
    episodic: {k: [[*data_epi1], [*data_epi2], ...]}
    plain: {k: [*data]}
    '''
    dones = batch['dones']  # the most reliable, scalar
    # if depth > 1, is nested, then is episodic
    return len(np.shape(dones)) > 1


def calc_batch_adv(batch, gamma):
    '''Calculate the advantage for a batch of data containing list of epi_rewards'''
    batch_rewards = batch['rewards']
    if is_episodic(batch):
        batch_advs = [calc_adv(epi_rewards, gamma) for epi_rewards in batch_rewards]
        batch_advs = np.concatenate(batch_advs)
    else:
        batch_advs = calc_adv(batch_rewards, gamma)
    return batch_advs


def calc_adv(rewards, gamma):
    '''Base method to calculate plain advantage with simple reward baseline'''
    T = len(rewards)
    assert not np.any(np.isnan(rewards))
    advs = np.empty(T, 'float32')
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = rewards[t] + gamma * future_ret
        advs[t] = future_ret
    return advs


def calc_gae_and_v_tar(batch, gamma, lam):
    '''Calculate GAE and v_targets given a batch of trajectory. See http://www.breloff.com/DeepRL-OnlineGAE/ for clear example.'''
    # TODO ensure proper horizon we do don't need not_done. sample till one step before done
    rewards = batch['rewards']
    v_preds = batch['v_preds']
    T = len(rewards)
    assert not np.any(np.isnan(rewards))
    assert T == len(v_preds)
    v_preds = np.append(v_preds, batch['next_v_pred'])
    advs = np.empty(T, 'float32')
    future_adv = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * v_preds[t + 1] - v_preds[t]
        advs[t] = future_adv = delta + gamma * lam * future_adv
    assert not np.isnan(advs).any(), f'GAE has nan: {gaes}'
    batch['v_targets'] = advs + batch['v_preds']
    seg['advs'] = advs
    return batch


# Q-learning calc
