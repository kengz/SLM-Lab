from slm_lab.experiment.control import make_agent_env
from slm_lab.lib import util
from slm_lab.spec import spec_util
from xvfbwrapper import Xvfb
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='session', autouse=True)
def test_xvfb():
    '''provide xvfb in test environment'''
    vdisplay = Xvfb()
    try:  # guard for multiprocessing dist test
        vdisplay.start()
        yield vdisplay
        vdisplay.stop()
    except Exception as e:
        yield vdisplay


@pytest.fixture(scope='session')
def test_spec():
    spec = spec_util.get('base.json', 'base_case_openai')
    spec = spec_util.override_test_spec(spec)
    return spec


@pytest.fixture
def test_df():
    data = pd.DataFrame({
        'integer': [1, 2, 3],
        'square': [1, 4, 9],
        'letter': ['a', 'b', 'c'],
    })
    assert isinstance(data, pd.DataFrame)
    return data


@pytest.fixture
def test_dict():
    data = {
        'a': 1,
        'b': 2,
        'c': 3,
    }
    assert isinstance(data, dict)
    return data


@pytest.fixture
def test_list():
    data = [1, 2, 3]
    assert isinstance(data, list)
    return data


@pytest.fixture
def test_obj():
    class Foo:
        bar = 'bar'
    return Foo()


@pytest.fixture
def test_str():
    data = 'lorem ipsum dolor'
    assert isinstance(data, str)
    return data


@pytest.fixture(scope='session', params=[
    (
        2,
        [
            [np.asarray([1, 1, 1, 1]), 1, 1, np.asarray([2, 2, 2, 2]), 1],
            [np.asarray([2, 2, 2, 2]), 1, 2, np.asarray([3, 3, 3, 3]), 2],
            [np.asarray([3, 3, 3, 3]), 1, 3, np.asarray([4, 4, 4, 4]), 3],
            [np.asarray([4, 4, 4, 4]), 1, 4, np.asarray([5, 5, 5, 5]), 4],
            [np.asarray([5, 5, 5, 5]), 1, 5, np.asarray([6, 6, 6, 6]), 5],
            [np.asarray([6, 6, 6, 6]), 1, 6, np.asarray([7, 7, 7, 7]), 6],
            [np.asarray([7, 7, 7, 7]), 1, 7, np.asarray([8, 8, 8, 8]), 7],
            [np.asarray([8, 8, 8, 8]), 1, 8, np.asarray([9, 9, 9, 9]), 8],
        ]
    ),
])
def test_memory(request):
    spec = spec_util.get('base.json', 'base_memory')
    agent, env = make_agent_env(spec)
    res = (agent.body.memory, ) + request.param
    return res


@pytest.fixture(scope='session', params=[
    (
        2,
        [
            [np.asarray([1, 1, 1, 1]), 1, 1, np.asarray([2, 2, 2, 2]), 0],
            [np.asarray([2, 2, 2, 2]), 1, 2, np.asarray([3, 3, 3, 3]), 0],
            [np.asarray([3, 3, 3, 3]), 1, 3, np.asarray([4, 4, 4, 4]), 0],
            [np.asarray([4, 4, 4, 4]), 1, 4, np.asarray([5, 5, 5, 5]), 0],
            [np.asarray([5, 5, 5, 5]), 1, 5, np.asarray([6, 6, 6, 6]), 0],
            [np.asarray([6, 6, 6, 6]), 1, 6, np.asarray([7, 7, 7, 7]), 0],
            [np.asarray([7, 7, 7, 7]), 1, 7, np.asarray([8, 8, 8, 8]), 0],
            [np.asarray([8, 8, 8, 8]), 1, 8, np.asarray([9, 9, 9, 9]), 1],
        ]
    ),
])
def test_on_policy_episodic_memory(request):
    spec = spec_util.get('base.json', 'base_on_policy_memory')
    agent, env = make_agent_env(spec)
    res = (agent.body.memory, ) + request.param
    return res


@pytest.fixture(scope='session', params=[
    (
        4,
        [
            [np.asarray([1, 1, 1, 1]), 1, 1, np.asarray([2, 2, 2, 2]), 0],
            [np.asarray([2, 2, 2, 2]), 1, 2, np.asarray([3, 3, 3, 3]), 0],
            [np.asarray([3, 3, 3, 3]), 1, 3, np.asarray([4, 4, 4, 4]), 0],
            [np.asarray([4, 4, 4, 4]), 1, 4, np.asarray([5, 5, 5, 5]), 0],
            [np.asarray([5, 5, 5, 5]), 1, 5, np.asarray([6, 6, 6, 6]), 0],
            [np.asarray([6, 6, 6, 6]), 1, 6, np.asarray([7, 7, 7, 7]), 0],
            [np.asarray([7, 7, 7, 7]), 1, 7, np.asarray([8, 8, 8, 8]), 0],
            [np.asarray([8, 8, 8, 8]), 1, 8, np.asarray([9, 9, 9, 9]), 1],
        ]
    ),
])
def test_on_policy_batch_memory(request):
    spec = spec_util.get('base.json', 'base_on_policy_batch_memory')
    agent, env = make_agent_env(spec)
    res = (agent.body.memory, ) + request.param
    return res


@pytest.fixture(scope='session', params=[
    (
        4,
        [
            [np.asarray([1, 1, 1, 1]), 1, 1, np.asarray([2, 2, 2, 2]), 0, 1000],
            [np.asarray([2, 2, 2, 2]), 1, 2, np.asarray([3, 3, 3, 3]), 0, 0],
            [np.asarray([3, 3, 3, 3]), 1, 3, np.asarray([4, 4, 4, 4]), 0, 0],
            [np.asarray([4, 4, 4, 4]), 1, 4, np.asarray([5, 5, 5, 5]), 0, 0],
            [np.asarray([5, 5, 5, 5]), 1, 5, np.asarray([6, 6, 6, 6]), 0, 1000],
            [np.asarray([6, 6, 6, 6]), 1, 6, np.asarray([7, 7, 7, 7]), 0, 0],
            [np.asarray([7, 7, 7, 7]), 1, 7, np.asarray([8, 8, 8, 8]), 0, 0],
            [np.asarray([8, 8, 8, 8]), 1, 8, np.asarray([9, 9, 9, 9]), 1, 1000],
        ]
    ),
])
def test_prioritized_replay_memory(request):
    spec = spec_util.get('base.json', 'base_prioritized_replay_memory')
    agent, env = make_agent_env(spec)
    res = (agent.body.memory, ) + request.param
    return res
