'''
Calculations used by algorithms
All calculations for training shall have a standard API that takes in `batch` from algorithm.sample() method.
`batch` is a dict containing keys to any data type you wish, e.g. {rewards: np.array([...])}
'''
from slm_lab.lib import logger
import numpy as np
import torch
import pydash as ps

# Policy Gradient calc

# TODO standardize arg with full sampled info


def calc_batch_adv(batch, gamma):
    '''Calculate the advantage for a batch of data containing list of epi_rewards'''
    batch_rewards = batch['rewards']
    batch_advs = [calc_adv(epi_rewards, gamma) for epi_rewards in batch_rewards]
    batch_advs = torch.cat(batch_advs)
    return batch_advs


def calc_adv(epi_rewards, gamma):
    '''Base method to calculate plain advantage with simple reward baseline'''
    T = len(epi_rewards)
    advs = np.empty(T, 'float32')
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = epi_rewards[t] + gamma * future_ret
        advs[t] = future_ret
    advs = (advs - advs.mean()) / (advs.std() + 1e-08)
    advs = torch.from_numpy(advs)
    return advs


def calc_gae_and_v_tar(segment, gamma, lam):
    '''Calculate GAE and v_targets given a segment of trajectory. See http://www.breloff.com/DeepRL-OnlineGAE/ for clear example.'''
    # TODO ensure proper horizon we do don't need not_done. sample till one step before done
    rewards = segment['rewards']
    v_preds = segment['v_preds']
    T = len(rewards)
    assert not np.any(np.isnan(rewards))
    assert T == len(v_preds)
    v_preds = np.append(v_preds, segment['next_v_pred'])
    advs = np.empty(T, 'float32')
    future_adv = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * v_preds[t + 1] - v_preds[t]
        advs[t] = future_adv = delta + gamma * lam * future_adv
    assert not np.isnan(advs).any(), f'GAE has nan: {gaes}'
    segment['v_targets'] = advs + segment['v_preds']
    seg['advs'] = advs
    return segment


# Q-learning calc
