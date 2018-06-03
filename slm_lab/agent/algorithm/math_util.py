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

# advantage functions


def calc_returns(batch, gamma):
    '''
    Calculate the simple returns (full rollout) for advantage
    i.e. sum discounted rewards up till termination
    '''
    rewards = batch['rewards']
    # handle epi-end, to not sum past current episode
    not_dones = 1 - batch['dones']
    T = len(rewards)
    assert not np.any(np.isnan(rewards))
    rets = np.empty(T, 'float32')
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = rewards[t] + gamma * future_ret * not_dones[t]
        rets[t] = future_ret
    rets = torch.from_numpy(rets).float()
    return rets


def calc_nstep_returns(batch, gamma, n):
    '''
    Calculate the n-step returns for advantage
    i.e. sum discounted rewards up till step n (0 to n-1 that is) for each timestep t
    '''
    rets = calc_returns(batch, gamma)
    # subtract by offsetting n-steps
    tail_rets = torch.cat([rets[n:], torch.zeros((n,))])
    nstep_rets = rets - tail_rets
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
    assert not np.any(np.isnan(rewards))
    assert T == len(v_preds)
    gaes = np.empty(T, 'float32')
    future_gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v_preds[t] - v_preds[t]
        gaes[t] = future_gae = delta + gamma * lam * future_gae
    assert not np.isnan(gaes).any(), f'GAE has nan: {gaes}'
    gaes = torch.from_numpy(gaes).float()
    return gaes


# Q-learning calc
