'''
Calculations used by algorithms
All calculations for training shall have a standard API that takes in `batch` from algorithm.sample() method and return np array for calculation.
`batch` is a dict containing keys to any data type you wish, e.g. {rewards: np.array([...])}
'''
from slm_lab.lib import logger, util
import numpy as np
import torch
import pydash as ps

# Policy Gradient calc

# TODO standardize arg with full sampled info


def calc_advs(batch, gamma):
    '''
    Base method to calculate plain advantage with simple reward baseline
    NOTE for standardization trick, do it out of here
    '''
    rewards = batch['rewards']
    T = len(rewards)
    assert not np.any(np.isnan(rewards))
    advs = np.empty(T, 'float32')
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = rewards[t] + gamma * future_ret
        advs[t] = future_ret
    advs = torch.from_numpy(advs).float()
    return advs


def calc_gaes_v_targets(rewards, v_preds, next_v_preds, gamma, lam):
    '''
    Calculate GAE and v_targets.
    See http://www.breloff.com/DeepRL-OnlineGAE/ for clear example.
    v_preds are values predicted for current states
    next_v_preds are values predicted for next states
    NOTE for standardization trick, do it out of here
    '''
    T = len(rewards)
    assert not np.any(np.isnan(rewards))
    assert T == len(v_preds)
    gaes = np.empty(T, 'float32')
    future_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v_preds[t] - v_preds[t]
        gaes[t] = future_gae = delta + gamma * lam * future_gae
    assert not np.isnan(gaes).any(), f'GAE has nan: {gaes}'
    gaes = torch.from_numpy(gaes).float()
    v_targets = gaes + v_preds
    return gaes, v_targets


# Q-learning calc
