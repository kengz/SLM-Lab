from slm_lab.experiment.control import make_agent_env
from slm_lab.spec import spec_util
import numpy as np
import pandas as pd
import pytest


def make_exp(state, action, reward, next_state, done, terminated=False, truncated=False, error=None):
    """Create experience dict for add_experience()."""
    exp = {
        'state': np.asarray(state),
        'action': action,
        'reward': reward,
        'next_state': np.asarray(next_state),
        'done': done,
        'terminated': terminated,
        'truncated': truncated,
    }
    if error is not None:
        exp['error'] = error
    return exp


# Standard experiences for replay memory tests
STANDARD_EXPERIENCES = [
    make_exp([1, 1, 1, 1], 1, 1, [2, 2, 2, 2], False),
    make_exp([2, 2, 2, 2], 1, 2, [3, 3, 3, 3], False),
    make_exp([3, 3, 3, 3], 1, 3, [4, 4, 4, 4], False),
    make_exp([4, 4, 4, 4], 1, 4, [5, 5, 5, 5], False),
    make_exp([5, 5, 5, 5], 1, 5, [6, 6, 6, 6], False),
    make_exp([6, 6, 6, 6], 1, 6, [7, 7, 7, 7], False),
    make_exp([7, 7, 7, 7], 1, 7, [8, 8, 8, 8], False),
    make_exp([8, 8, 8, 8], 1, 8, [9, 9, 9, 9], True, terminated=True),
]

# PER experiences with error field
PER_EXPERIENCES = [
    make_exp([1, 1, 1, 1], 1, 1, [2, 2, 2, 2], False, error=1000),
    make_exp([2, 2, 2, 2], 1, 2, [3, 3, 3, 3], False, error=0),
    make_exp([3, 3, 3, 3], 1, 3, [4, 4, 4, 4], False, error=0),
    make_exp([4, 4, 4, 4], 1, 4, [5, 5, 5, 5], False, error=0),
    make_exp([5, 5, 5, 5], 1, 5, [6, 6, 6, 6], False, error=1000),
    make_exp([6, 6, 6, 6], 1, 6, [7, 7, 7, 7], False, error=0),
    make_exp([7, 7, 7, 7], 1, 7, [8, 8, 8, 8], False, error=0),
    make_exp([8, 8, 8, 8], 1, 8, [9, 9, 9, 9], True, terminated=True, error=1000),
]


@pytest.fixture(scope='session')
def test_spec():
    spec = spec_util.get('experimental/misc/base.json', 'base_case_gymnasium')
    spec_util.tick(spec, 'trial')
    spec = spec_util.override_spec(spec, 'test')
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


@pytest.fixture(scope='session', params=[(2, STANDARD_EXPERIENCES)])
def test_memory(request):
    spec = spec_util.get('experimental/misc/base.json', 'base_replay_memory')
    spec_util.tick(spec, 'trial')
    agent, env = make_agent_env(spec)
    res = (agent.memory,) + request.param
    return res


@pytest.fixture(scope='session', params=[(2, STANDARD_EXPERIENCES)])
def test_on_policy_episodic_memory(request):
    spec = spec_util.get('experimental/misc/base.json', 'base_onpolicy_memory')
    spec_util.tick(spec, 'trial')
    agent, env = make_agent_env(spec)
    res = (agent.memory,) + request.param
    return res


@pytest.fixture(scope='session', params=[(4, STANDARD_EXPERIENCES)])
def test_on_policy_batch_memory(request):
    spec = spec_util.get('experimental/misc/base.json', 'base_onpolicy_batch_memory')
    spec_util.tick(spec, 'trial')
    agent, env = make_agent_env(spec)
    res = (agent.memory,) + request.param
    return res


@pytest.fixture(scope='session', params=[(4, PER_EXPERIENCES)])
def test_prioritized_replay_memory(request):
    spec = spec_util.get('experimental/misc/base.json', 'base_prioritized_replay_memory')
    spec_util.tick(spec, 'trial')
    agent, env = make_agent_env(spec)
    res = (agent.memory,) + request.param
    return res
