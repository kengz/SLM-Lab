from copy import deepcopy
from slm_lab.env.base import Clock
from slm_lab.agent.net import net_util
from slm_lab.agent.net.recurrent import RecurrentNet
import pytest
import torch
import torch.nn as nn

net_spec = {
    "type": "RecurrentNet",
    "shared": True,
    "cell_type": "GRU",
    "fc_hid_layers": [10],
    "hid_layers_activation": "relu",
    "rnn_hidden_size": 64,
    "rnn_num_layers": 2,
    "bidirectional": False,
    "seq_len": 4,
    "init_fn": "xavier_uniform_",
    "clip_grad_val": 1.0,
    "loss_spec": {
        "name": "SmoothL1Loss"
    },
    "optim_spec": {
        "name": "Adam",
        "lr": 0.02
    },
    "lr_scheduler_spec": {
        "name": "StepLR",
        "step_size": 30,
        "gamma": 0.1
    },
    "gpu": True
}
state_dim = 10
out_dim = 3
batch_size = 16
seq_len = net_spec['seq_len']
in_dim = (seq_len, state_dim)
net = RecurrentNet(net_spec, in_dim, out_dim)
# init net optimizer and its lr scheduler
optim = net_util.get_optim(net, net.optim_spec)
lr_scheduler = net_util.get_lr_scheduler(optim, net.lr_scheduler_spec)
x = torch.rand((batch_size, seq_len, state_dim))


def test_init():
    net = RecurrentNet(net_spec, in_dim, out_dim)
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'fc_model')
    assert hasattr(net, 'rnn_model')
    assert hasattr(net, 'model_tail')
    assert not hasattr(net, 'model_tails')
    assert net.rnn_model.bidirectional == False


def test_forward():
    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_training_step():
    y = torch.rand((batch_size, out_dim))
    clock = Clock(100, 'total_t', 1)
    loss = net.loss_fn(net.forward(x), y)
    net.training_step(loss, optim, lr_scheduler, lr_clock=clock)
    assert loss != 0.0


@pytest.mark.parametrize('bidirectional', (False, True))
@pytest.mark.parametrize('cell_type', ('RNN', 'LSTM', 'GRU'))
def test_variant(bidirectional, cell_type):
    var_net_spec = deepcopy(net_spec)
    var_net_spec['bidirectional'] = bidirectional
    var_net_spec['cell_type'] = cell_type
    net = RecurrentNet(var_net_spec, in_dim, out_dim)
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'fc_model')
    assert hasattr(net, 'rnn_model')
    assert hasattr(net, 'model_tail')
    assert not hasattr(net, 'model_tails')
    assert net.rnn_model.bidirectional == bidirectional

    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_no_fc():
    no_fc_net_spec = deepcopy(net_spec)
    no_fc_net_spec['fc_hid_layers'] = []
    net = RecurrentNet(no_fc_net_spec, in_dim, out_dim)
    assert isinstance(net, nn.Module)
    assert not hasattr(net, 'fc_model')
    assert hasattr(net, 'rnn_model')
    assert hasattr(net, 'model_tail')
    assert not hasattr(net, 'model_tails')

    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_multitails():
    net = RecurrentNet(net_spec, in_dim, [3, 4])
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'fc_model')
    assert hasattr(net, 'rnn_model')
    assert not hasattr(net, 'model_tail')
    assert hasattr(net, 'model_tails')
    assert len(net.model_tails) == 2

    y = net.forward(x)
    assert len(y) == 2
    assert y[0].shape == (batch_size, 3)
    assert y[1].shape == (batch_size, 4)
