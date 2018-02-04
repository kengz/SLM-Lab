from slm_lab.agent import AgentSpace, Body
from slm_lab.agent.memory import Replay
from slm_lab.agent.net.convnet import ConvNet
from slm_lab.agent.net.recurrent import RecurrentNet
from slm_lab.agent.net.feedforward import MLPNet, MultiMLPNet, MLPHeterogenousHeads
from slm_lab.env import EnvSpace
# from slm_lab.experiment.control import Trial
from slm_lab.experiment.monitor import AEBSpace, InfoSpace
from slm_lab.lib import util
from slm_lab.spec import spec_util
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


spec = None
aeb_space = None
agent = None
env = None


@pytest.fixture(scope='session')
def test_spec():
    global spec
    spec = spec_util.get('base.json', 'base_case')
    spec = util.override_test_spec(spec)
    return spec


# TODO properly use in tests
# @pytest.fixture(scope='session')
# def test_session(test_spec):
#     trial = Trial(test_spec)
#     session = trial.init_session()
#     yield session
#     session.close()


@pytest.fixture(scope='session')
def test_aeb_space(test_spec):
    global aeb_space
    if aeb_space is None:
        aeb_space = AEBSpace(test_spec, InfoSpace())
        env_space = EnvSpace(test_spec, aeb_space)
        agent_space = AgentSpace(test_spec, aeb_space)
        aeb_space.init_body_space()
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


@pytest.fixture
def test_multiline_str():
    data = '''
        lorem ipsum dolor
        sit amet

        consectetur adipiscing elit
        '''
    assert isinstance(data, str)
    return data


@pytest.fixture(scope="class", params=[
    (
        MLPNet,
        {
            'in_dim': 10, 'hid_dim': [5, 3],
            'out_dim':2,
            'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPNet,
        {
            'in_dim': 20, 'hid_dim': [10, 50, 5],
            'out_dim':2, 'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPNet,
        {
            'in_dim': 10, 'hid_dim': [],
            'out_dim':5, 'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPHeterogenousHeads,
        {
            'in_dim': 10, 'hid_dim': [5, 3],
            'out_dim':[2],
            'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPHeterogenousHeads,
        {
            'in_dim': 10, 'hid_dim': [5, 3],
            'out_dim':[2, 1],
            'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPHeterogenousHeads,
        {
            'in_dim': 10, 'hid_dim': [5, 3],
            'out_dim':[2, 5, 1],
            'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPHeterogenousHeads,
        {
            'in_dim': 10, 'hid_dim': [10, 50, 5],
            'out_dim':[2, 5, 1],
            'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPHeterogenousHeads,
        {
            'in_dim': 10, 'hid_dim': [],
            'out_dim':[5], 'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPHeterogenousHeads,
        {
            'in_dim': 10, 'hid_dim': [],
            'out_dim':[5, 2], 'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        MLPHeterogenousHeads,
        {
            'in_dim': 10, 'hid_dim': [],
            'out_dim':[5, 2, 1], 'hid_layers_activation': 'tanh',
        },
        None,
        2
    ), (
        ConvNet,
        {
            'in_dim': (3, 32, 32),
            'hid_layers': ([],
                           []),
            'out_dim': 10,
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
            'clamp_grad': False,
            'batch_norm': False,
        },
        None,
        2
    ), (
        ConvNet,
        {
            'in_dim': (3, 32, 32),
            'hid_layers': ([[3, 16, (5, 5), 2, 0, 1],
                            [16, 32, (5, 5), 2, 0, 1]],
                           [100]),
            'out_dim': 10,
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
            'clamp_grad': False,
            'batch_norm': False,
        },
        None,
        2
    ), (
        ConvNet,
        {
            'in_dim': (3, 32, 32),
            'hid_layers': ([[3, 16, (5, 5), 2, 0, 1],
                            [16, 32, (5, 5), 2, 0, 1]],
                           [100, 50]),
            'out_dim': 10,
            'optim_param': {'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
            'clamp_grad': False,
            'batch_norm': True,
        },
        None,
        2
    ), (
        ConvNet,
        {
            'in_dim': (3, 32, 32),
            'hid_layers': ([[3, 16, (5, 5), 2, 0, 1],
                            [16, 32, (5, 5), 1, 0, 1],
                            [32, 64, (5, 5), 1, 0, 2]],
                           [100]),
            'out_dim': 10,
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
            'clamp_grad': True,
            'batch_norm': False,
        },
        None,
        2
    ), (
        ConvNet,
        {
            'in_dim': (3, 32, 32),
            'hid_layers': ([[3, 16, (5, 5), 2, 0, 1],
                            [16, 32, (5, 5), 1, 0, 1],
                            [32, 64, (5, 5), 1, 0, 2]],
                           [100]),
            'out_dim': 10,
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
            'clamp_grad': True,
            'batch_norm': True,
        },
        None,
        2
    ), (
        ConvNet,
        {
            'in_dim': (3, 32, 32),
            'hid_layers': ([[3, 16, (7, 7), 1, 0, 1],
                            [16, 32, (5, 5), 1, 0, 1],
                            [32, 64, (3, 3), 1, 0, 1]],
                           [100, 50]),
            'out_dim': 10,
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
            'clamp_grad': False,
            'batch_norm': False,
        },
        None,
        2
    ), (
        ConvNet,
        {
            'in_dim': (3, 32, 32),
            'hid_layers': ([[3, 16, (7, 7), 1, 0, 1],
                            [16, 32, (5, 5), 1, 0, 1],
                            [32, 64, (3, 3), 1, 0, 1]],
                           [100, 50]),
            'out_dim': 10,
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
            'clamp_grad': False,
            'batch_norm': True,
        },
        None,
        2
    ), (
        MultiMLPNet,
        {
            'in_dim': [[5, 10], [8, 16]],
            'hid_dim': [64],
            'out_dim': [[3], [2]],
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        MultiMLPNet,
        {
            'in_dim': [[5, 10], [8, 16]],
            'hid_dim': [],
            'out_dim': [[3], [2]],
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        MultiMLPNet,
        {
            'in_dim': [[5, 10], [8, 16]],
            'hid_dim': [],
            'out_dim': [[5, 3], [8, 2]],
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        MultiMLPNet,
        {
            'in_dim': [[5, 10, 15], [8, 16]],
            'hid_dim': [],
            'out_dim': [[5, 3], [12, 8, 2]],
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        MultiMLPNet,
        {
            'in_dim': [[5, 10, 15], [8, 16]],
            'hid_dim': [32, 64],
            'out_dim': [[5, 3], [12, 8, 2]],
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        MultiMLPNet,
        {
            'in_dim': [[5, 10], [8, 16, 24]],
            'hid_dim': [32, 64],
            'out_dim': [[9, 6, 3], [2]],
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 4,
            'hid_dim': [64, 50],
            'out_dim': 10,
            'sequence_length': 8,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 4,
            'hid_dim': [50],
            'out_dim': 10,
            'sequence_length': 8,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 4,
            'hid_dim': [64, 32, 50],
            'out_dim': 10,
            'sequence_length': 8,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 4,
            'hid_dim': [64, 32, 100],
            'out_dim': 10,
            'sequence_length': 8,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 6,
            'hid_dim': [64, 32, 100],
            'out_dim': 10,
            'sequence_length': 8,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 6,
            'hid_dim': [64, 32, 100],
            'out_dim': 10,
            'sequence_length': 16,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 6,
            'hid_dim': [64, 32, 100],
            'out_dim': 20,
            'sequence_length': 16,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 6,
            'hid_dim': [64, 32, 16],
            'out_dim': [20],
            'sequence_length': 16,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 6,
            'hid_dim': [64, 32, 16],
            'out_dim': [20, 10],
            'sequence_length': 16,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ), (
        RecurrentNet,
        {
            'in_dim': 6,
            'hid_dim': [64, 32, 16],
            'out_dim': [20, 10, 1],
            'sequence_length': 16,
            'hid_layers_activation': 'tanh',
            'optim_param':{'name': 'Adam'},
            'loss_param': {'name': 'mse_loss'},
        },
        None,
        2
    ),
])
def test_nets(request):
    net = request.param[0](**request.param[1])
    res = (net,) + request.param[2:]
    return res


@pytest.fixture(scope="class", params=[(None, None)])
def test_data_gen(request):
    return request.param


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
    agent_space = AgentSpace(memspec, aeb_mem_space)
    aeb_mem_space.init_body_space()
    aeb_mem_space.post_body_init()
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
    agent_space = AgentSpace(memspec, aeb_mem_space)
    aeb_mem_space.init_body_space()
    aeb_mem_space.post_body_init()
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
    agent_space = AgentSpace(memspec, aeb_mem_space)
    aeb_mem_space.init_body_space()
    aeb_mem_space.post_body_init()
    agent = agent_space.agents[0]
    body = agent.nanflat_body_a[0]
    res = (body.memory, ) + request.param
    return res
