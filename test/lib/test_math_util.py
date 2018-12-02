from slm_lab.lib import math_util
import numpy as np
import pytest


@pytest.mark.parametrize('i,res', [
    (2, np.array([[0], [0], [1], [2]])),
    (3, np.array([[0], [1], [2], [3]])),
    (4, np.array([[4], [4], [4], [4]])),
    (5, np.array([[4], [4], [4], [5]])),
])
def test_get_backstack(i, res):
    dones = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    states = np.arange(8)
    states = np.expand_dims(states, -1)
    stack_len = 4

    stack_states = math_util.get_backstack(dones, states, stack_len, i)
    assert np.array_equal(stack_states, res)
