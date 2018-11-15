from slm_lab.agent import AgentSpace
from slm_lab.agent.net.conv import ConvNet
from slm_lab.agent.net.mlp import MLPNet, HydraMLPNet
from slm_lab.agent.net.recurrent import RecurrentNet
from slm_lab.env import EnvSpace
from slm_lab.experiment.monitor import AEBSpace, InfoSpace
from slm_lab.lib import util
from slm_lab.spec import spec_util
import numpy as np
import pandas as pd
import pytest


spec = None
aeb_space = None
agent = None
env = None


@pytest.fixture(scope='session')
def test_spec():
    global spec
    spec = spec_util.get('base.json', 'base_case_openai')
    spec = util.override_test_spec(spec)
    return spec


@pytest.fixture(scope='session')
def test_info_space():
    return InfoSpace()


@pytest.fixture(scope='session')
def test_aeb_space(test_spec):
    global aeb_space
    if aeb_space is None:
        aeb_space = AEBSpace(test_spec, InfoSpace())
        env_space = EnvSpace(test_spec, aeb_space)
        aeb_space.init_body_space()
        agent_space = AgentSpace(test_spec, aeb_space)
    return aeb_space


@pytest.fixture(scope='session')
def test_agent(test_aeb_space):
    agent = test_aeb_space.agent_space.agents[0]
    return agent


@pytest.fixture(scope='session')
def test_env(test_aeb_space):
    env = test_aeb_space.env_space.envs[0]
    return env


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
    memspec = spec_util.get('base.json', 'base_memory')
    memspec = util.override_test_spec(memspec)
    aeb_mem_space = AEBSpace(memspec, InfoSpace())
    env_space = EnvSpace(memspec, aeb_mem_space)
    aeb_mem_space.init_body_space()
    agent_space = AgentSpace(memspec, aeb_mem_space)
    agent = agent_space.agents[0]
    body = agent.nanflat_body_a[0]
    res = (body.memory, ) + request.param
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
    memspec = spec_util.get('base.json', 'base_on_policy_memory')
    memspec = util.override_test_spec(memspec)
    aeb_mem_space = AEBSpace(memspec, InfoSpace())
    env_space = EnvSpace(memspec, aeb_mem_space)
    aeb_mem_space.init_body_space()
    agent_space = AgentSpace(memspec, aeb_mem_space)
    agent = agent_space.agents[0]
    body = agent.nanflat_body_a[0]
    res = (body.memory, ) + request.param
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
    memspec = spec_util.get('base.json', 'base_on_policy_batch_memory')
    memspec = util.override_test_spec(memspec)
    aeb_mem_space = AEBSpace(memspec, InfoSpace())
    env_space = EnvSpace(memspec, aeb_mem_space)
    aeb_mem_space.init_body_space()
    agent_space = AgentSpace(memspec, aeb_mem_space)
    agent = agent_space.agents[0]
    body = agent.nanflat_body_a[0]
    res = (body.memory, ) + request.param
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
def test_prioritized_replay_memory(request):
    memspec = spec_util.get('base.json', 'base_prioritized_replay_memory')
    memspec = util.override_test_spec(memspec)
    aeb_mem_space = AEBSpace(memspec, InfoSpace())
    env_space = EnvSpace(memspec, aeb_mem_space)
    aeb_mem_space.init_body_space()
    agent_space = AgentSpace(memspec, aeb_mem_space)
    agent = agent_space.agents[0]
    body = agent.nanflat_body_a[0]
    res = (body.memory, ) + request.param
    return res
