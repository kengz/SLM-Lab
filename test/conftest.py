import pandas as pd
import torch
from torch.autograd import Variable
from unity_lab.agent.net.feedforward import MLPNet
import pytest


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
(MLPNet(10, [5, 3], 2),
Variable(torch.ones((2, 10))),
Variable(torch.zeros((2, 2))),
None,
2),
(MLPNet(20, [10, 50, 5], 2),
Variable(torch.ones((2, 20))),
Variable(torch.zeros((2, 2))),
None,
2),
(MLPNet(10, [], 5),
Variable(torch.ones((2, 10))),
Variable(torch.zeros((2, 5))),
None,
2)])
def test_nets(request):
    return request.param
