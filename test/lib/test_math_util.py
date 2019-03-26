from slm_lab.lib import math_util
import numpy as np
import pytest
import torch


def calc_nstep_returns_slow(rewards, dones, v_preds, gamma, n):
    '''
    Slower method to check the correctness of calc_nstep_returns
    Calculate the n-step returns for advantage. Ref: http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf
    R^(n)_t = r_{t+1} + gamma r_{t+2} + ... + gamma^(n-1) r_{t+n} + gamma^(n) V(s_{t+n})
    For edge case where there is no r term, substitute with V and end the sum,
    e.g. for max t = 5, R^(3)_4 = r_5 + gamma V(s_5)
    '''
    T = len(rewards)
    assert not torch.isnan(rewards).any()
    rets = torch.zeros(T, dtype=torch.float32, device=v_preds.device)
    # to multiply with not_dones to handle episode boundary (last state has no V(s'))
    # not_dones = 1 - dones
    for t in range(T):
        ret = 0.0
        cur_gamma = 1.0
        for idx in range(n):
            i = idx + 1
            # short circuit if this reward does not exist
            if t + i >= T or dones[t + i]:
                i -= 1  # set it back to index of last valid reward
                break
            ret += cur_gamma * rewards[t + i]
            cur_gamma *= gamma
        ret += cur_gamma * v_preds[t + i]
        rets[t] = ret
    return rets


def test_calc_gaes():
    rewards = torch.tensor([1., 0., 1., 1., 0., 1., 1., 1.])
    dones = torch.tensor([0., 0., 1., 1., 0., 0., 0., 0.])
    v_preds = torch.tensor([1.1, 0.1, 1.1, 1.1, 0.1, 1.1, 1.1, 1.1, 1.1])
    assert len(v_preds) == len(rewards) + 1  # includes last state
    gamma = 0.99
    lam = 0.95
    gaes = math_util.calc_gaes(rewards, dones, v_preds, gamma, lam)
    res = torch.tensor([0.84070045, 0.89495, -0.1, -0.1, 3.616724, 2.7939649, 1.9191545, 0.989])
    # use allclose instead of equal to account for atol
    assert torch.allclose(gaes, res)


def test_calc_nstep_returns():
    rewards = torch.tensor([0., 1., 2., 3., 4., 5., 6., ])
    v_preds = torch.tensor([1., 2., 3., 4., 5., 6., 7.])
    gamma = 0.99
    n = 3

    dones = torch.tensor([0., 0., 0., 0., 0., 0., 0., ])
    nstep_rets = math_util.calc_nstep_returns(rewards, dones, v_preds, gamma, n)
    res = calc_nstep_returns_slow(rewards, dones, v_preds, gamma, n)
    # use allclose instead of equal to account for atol
    assert torch.allclose(nstep_rets, res)

    dones = torch.tensor([0., 0., 0., 0., 1., 0., 0., ])
    nstep_rets = math_util.calc_nstep_returns(rewards, dones, v_preds, gamma, n)
    res = calc_nstep_returns_slow(rewards, dones, v_preds, gamma, n)
    # use allclose instead of equal to account for atol
    assert torch.allclose(nstep_rets, res)

    dones = torch.tensor([0., 0., 0., 0., 0., 0., 1., ])
    nstep_rets = math_util.calc_nstep_returns(rewards, dones, v_preds, gamma, n)
    res = calc_nstep_returns_slow(rewards, dones, v_preds, gamma, n)
    # use allclose instead of equal to account for atol
    assert torch.allclose(nstep_rets, res)


def test_calc_shaped_rewards():
    rewards = torch.tensor([1., 0., 1., 1., 0., 1., 1., 1.])
    dones = torch.tensor([0., 0., 1., 0., 0., 0., 0., 0.])
    v_pred = torch.tensor(2.1)
    gamma = 0.99
    shaped_rewards = math_util.calc_shaped_rewards(rewards, dones, v_pred, gamma)
    # when last is not done, doesn't use valye bootstrap
    res = torch.tensor([1.9801, 0.99, 1.0, 5.90807411479, 4.957650621, 5.0077279, 4.04821, 3.079])
    # use allclose instead of equal to account for atol
    assert torch.allclose(shaped_rewards, res)

    # when last is done, doesn't use valye bootstrap
    dones = torch.tensor([0., 0., 1., 0., 0., 0., 0., 1.])
    shaped_rewards = math_util.calc_shaped_rewards(rewards, dones, v_pred, gamma)
    res = torch.tensor([1.9801, 0.99, 1.0, 3.9109950099999997, 2.9403989999999998, 2.9701, 1.99, 1.0])
    assert torch.allclose(shaped_rewards, res)


@pytest.mark.parametrize('vec,res', [
    ([1, 1, 1], [False, False, False]),
    ([1, 1, 2], [False, False, True]),
    ([[1, 1], [1, 1], [1, 2]], [False, False, True]),
])
def test_is_outlier(vec, res):
    assert np.array_equal(math_util.is_outlier(vec), res)


@pytest.mark.parametrize('start_val, end_val, start_step, end_step, step, correct', [
    (0.1, 0.0, 0, 100, 0, 0.1),
    (0.1, 0.0, 0, 100, 50, 0.05),
    (0.1, 0.0, 0, 100, 100, 0.0),
    (0.1, 0.0, 0, 100, 150, 0.0),
    (0.1, 0.0, 100, 200, 50, 0.1),
    (0.1, 0.0, 100, 200, 100, 0.1),
    (0.1, 0.0, 100, 200, 150, 0.05),
    (0.1, 0.0, 100, 200, 200, 0.0),
    (0.1, 0.0, 100, 200, 250, 0.0),
])
def test_linear_decay(start_val, end_val, start_step, end_step, step, correct):
    assert math_util.linear_decay(start_val, end_val, start_step, end_step, step) == correct


@pytest.mark.parametrize('start_val, end_val, start_step, end_step, step, correct', [
    (1.0, 0.0, 0, 100, 0, 1.0),
    (1.0, 0.0, 0, 100, 5, 0.9),
    (1.0, 0.0, 0, 100, 10, 0.81),
    (1.0, 0.0, 0, 100, 25, 0.59049),
    (1.0, 0.0, 0, 100, 50, 0.3486784401),
    (1.0, 0.0, 0, 100, 100, 0.0),
    (1.0, 0.0, 0, 100, 150, 0.0),
    (1.0, 0.0, 100, 200, 0, 1.0),
    (1.0, 0.0, 100, 200, 50, 1.0),
    (1.0, 0.0, 100, 200, 100, 1.0),
    (1.0, 0.0, 100, 200, 105, 0.9),
    (1.0, 0.0, 100, 200, 125, 0.59049),
    (1.0, 0.0, 100, 200, 200, 0.0),
    (1.0, 0.0, 100, 200, 250, 0.0),
])
def test_rate_decay(start_val, end_val, start_step, end_step, step, correct):
    np.testing.assert_almost_equal(math_util.rate_decay(start_val, end_val, start_step, end_step, step), correct)
