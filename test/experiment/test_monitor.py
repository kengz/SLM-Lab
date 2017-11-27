import numpy as np
import pytest
from slm_lab.experiment.monitor import AEBSpace


def test_aeb_data_space_dual_data_proj(test_spec):
    aeb_space = AEBSpace(test_spec)
    # override for test
    aeb_space.a_eb_proj = [
        [(0, 0)],
        [(0, 1)]
    ]
    aeb_space.init_data_spaces()

    a0_action = [1, 2]
    a1_action = [3, 4]
    data_proj = [[a0_action], [a1_action]]
    dual_data_proj = [[[1, 2], [3, 4]]]

    # test add and dual_data_proj creation
    aeb_space.add('action', data_proj)
    action_space = aeb_space.data_spaces['action']
    assert np.array_equal(action_space.dual_data_proj, dual_data_proj)
    # test get
    assert np.array_equal(action_space.get(e=0), [a0_action, a1_action])
    assert np.array_equal(action_space.get(a=0), [a0_action])

    # symmetric test
    aeb_space.add('state', dual_data_proj)
    state_space = aeb_space.data_spaces['state']
    state_space.dual_data_proj
    assert np.array_equal(state_space.dual_data_proj, data_proj)


@pytest.mark.parametrize('a_eb_proj,res_e_ab_proj,res_e_ab_dual_map', [
    ([
        [(0, 0), (1, 0)],
    ], [
        [(0, 0)],
        [(0, 0)]
    ], [
        [(0, 0)],
        [(0, 1)]
    ]),
    ([
        [(0, 0)],
        [(0, 0), (0, 1)],
    ], [
        [(0, 0), (1, 0), (1, 1)],
    ], [
        [(0, 0), (1, 0), (1, 1)]
    ]),
    ([
        [(0, 0)],
        [(0, 0), (1, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (1, 3)]
    ], [
        [(0, 0), (1, 0)],
        [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3)],
    ], [
        [(0, 0), (1, 0)],
        [(1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3)]
    ]),
    ([
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0), (1, 1)]
    ], [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1)]
    ], [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 2), (1, 3)]
    ]),
])
def test_compute_dual_map(test_spec, a_eb_proj, res_e_ab_proj, res_e_ab_dual_map):
    aeb_space = AEBSpace(test_spec)
    e_ab_dual_map, e_ab_proj = aeb_space.compute_dual_map(a_eb_proj)
    assert np.array_equal(e_ab_dual_map, res_e_ab_dual_map)
    assert np.array_equal(e_ab_proj, res_e_ab_proj)
    a_eb_dual_map, check_a_eb_proj = aeb_space.compute_dual_map(
        e_ab_proj)
    assert np.array_equal(a_eb_proj, check_a_eb_proj)
