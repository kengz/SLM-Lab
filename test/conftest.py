from slm_lab.agent import AgentSpace
from slm_lab.agent.memory import Replay
from slm_lab.agent.net.convnet import ConvNet
from slm_lab.agent.net.feedforward import MLPNet, MultiMLPNet
from slm_lab.env import EnvSpace
from slm_lab.experiment.control import Trial
from slm_lab.experiment.monitor import AEBSpace, InfoSpace
from slm_lab.lib import util
from slm_lab.spec import spec_util
from torch.autograd import Variable
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
    spec['meta']['train_mode'] = True
    return spec


# TODO properly use in tests
@pytest.fixture(scope='session')
def test_session(test_spec):
    trial = Trial(test_spec)
    session = trial.init_session()
    yield session
    session.close()


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
    ),
])
def test_nets(request):
    net = request.param[0](**request.param[1])
    res = (net,) + request.param[2:]
    return res


@pytest.fixture(scope="class", params=[(None, None)])
def test_data_gen(request):
    return request.param


@pytest.fixture(scope="session", params=[
    ((5, 1, 1),
     2,
     [[1, 1, 1, 2, 0], [2, 2, 2, 3, 0], [3, 3, 3, 4, 0], [4, 4, 4, 5, 0],
      [5, 5, 5, 6, 0], [6, 6, 6, 7, 0], [7, 7, 7, 8, 0], [8, 8, 8, 9, 0],
      [9, 9, 9, 10, 0], [10, 10, 10, 11, 0], [11, 11, 11, 0, 1]]),
    ((8, 3, 2),
     3,
     [[[1, 1, 1], [1, 1], 1, [2, 2, 2], 0],
      [[2, 2, 2], [2, 2], 2, [3, 3, 3], 0],
      [[3, 3, 3], [3, 3], 3, [4, 4, 4], 0],
      [[4, 4, 4], [4, 4], 4, [5, 5, 5], 0],
      [[5, 5, 5], [5, 5], 5, [6, 6, 6], 0],
      [[6, 6, 6], [6, 6], 6, [7, 7, 7], 0],
      [[7, 7, 7], [7, 7], 7, [8, 8, 8], 0],
      [[8, 8, 8], [8, 8], 8, [9, 9, 9], 0],
      [[9, 9, 9], [9, 9], 9, [10, 10, 10], 0],
      [[10, 10, 10], [10, 10], 10, [11, 11, 11], 0],
      [[11, 11, 11], [11, 11], 11, [0, 0, 0], 1]])])
def test_memory(request, test_agent):
    max_size, state_dim, action_dim = request.param[0]
    batch_size = request.param[1]
    experiences = request.param[2]
    body = test_agent.bodies[0]
    body.max_size = max_size
    body.state_dim = state_dim
    body.action_dim = action_dim
    memory = Replay(test_agent)
    memory.post_body_init()
    return [memory, batch_size, experiences]
