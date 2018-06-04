'''
Calculations used by algorithms
All calculations for training shall have a standard API that takes in `batch` from algorithm.sample() method and return np array for calculation.
`batch` is a dict containing keys to any data type you wish, e.g. {rewards: np.array([...])}
'''
from slm_lab.lib import logger, util
import numpy as np
import torch
import pydash as ps

logger = logger.get_logger(__name__)

# Policy Gradient calc
# advantage functions


def calc_returns(batch, gamma):
    '''
    Calculate the simple returns (full rollout) for advantage
    i.e. sum discounted rewards up till termination
    '''
    rewards = batch['rewards']
    assert not np.any(np.isnan(rewards))
    # handle epi-end, to not sum past current episode
    not_dones = 1 - batch['dones']
    T = len(rewards)
    rets = np.empty(T, 'float32')
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = rewards[t] + gamma * future_ret * not_dones[t]
        rets[t] = future_ret
    rets = torch.from_numpy(rets).float()
    return rets


def calc_gammas(batch, gamma):
    '''Calculate the gammas to the right power for multiplication with rewards'''
    news = torch.cat([torch.ones((1,)), batch['dones'][:-1]])
    gammas = torch.empty_like(news)
    cur_gamma = 1.0
    for t, new in enumerate(news):
        cur_gamma = new * 1.0 + (1 - new) * cur_gamma * gamma
        gammas[t] = cur_gamma
    return gammas


def calc_nstep_returns(batch, gamma, n, v_preds):
    '''
    Calculate the n-step returns for advantage
    see n-step return in: http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf
    i.e. for each timestep t:
        sum discounted rewards up till step n (0 to n-1 that is),
        then add v_pred for n as final term
    '''
    rets = calc_returns(batch, gamma)
    rets_len = len(rets)
    # to subtract by offsetting n-steps
    tail_rets = torch.cat([rets[n:], torch.zeros((n,))])[:rets_len]

    # to add back the subtracted with v_pred at n
    gammas = calc_gammas(batch, gamma)
    final_terms = gammas * v_preds
    final_terms = torch.cat([final_terms[n:], torch.zeros((n,))])[:rets_len]

    nstep_rets = rets - tail_rets + final_terms
    assert not np.isnan(nstep_rets).any(), f'N-step returns has nan: {nstep_rets}'
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
