from copy import deepcopy
from slm_lab.env.base import Clock
from slm_lab.agent.net import net_util
from slm_lab.agent.net.conv import ConvNet
import torch
import torch.nn as nn

net_spec = {
    "type": "ConvNet",
    "shared": True,
    "conv_hid_layers": [
        [32, 8, 4, 0, 1],
        [64, 4, 2, 0, 1],
        [64, 3, 1, 0, 1]
    ],
    "fc_hid_layers": [512],
    "hid_layers_activation": "relu",
    "init_fn": "xavier_uniform_",
    "batch_norm": False,
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
in_dim = (4, 84, 84)
out_dim = 3
batch_size = 16
net = ConvNet(net_spec, in_dim, out_dim)
# init net optimizer and its lr scheduler
optim = net_util.get_optim(net, net.optim_spec)
lr_scheduler = net_util.get_lr_scheduler(optim, net.lr_scheduler_spec)
x = torch.rand((batch_size,) + in_dim)


def test_init():
    net = ConvNet(net_spec, in_dim, out_dim)
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'conv_model')
    assert hasattr(net, 'fc_model')
    assert hasattr(net, 'model_tail')
    assert not hasattr(net, 'model_tails')


def test_forward():
    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_train_step():
    y = torch.rand((batch_size, out_dim))
    clock = Clock(100, 'total_t', 1)
    loss = net.loss_fn(net.forward(x), y)
    net.train_step(loss, optim, lr_scheduler, lr_clock=clock)
    assert loss != 0.0


def test_no_fc():
    no_fc_net_spec = deepcopy(net_spec)
    no_fc_net_spec['fc_hid_layers'] = []
    net = ConvNet(no_fc_net_spec, in_dim, out_dim)
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'conv_model')
    assert not hasattr(net, 'fc_model')
    assert hasattr(net, 'model_tail')
    assert not hasattr(net, 'model_tails')

    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_multitails():
    net = ConvNet(net_spec, in_dim, [3, 4])
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'conv_model')
    assert hasattr(net, 'fc_model')
    assert not hasattr(net, 'model_tail')
    assert hasattr(net, 'model_tails')
    assert len(net.model_tails) == 2

    y = net.forward(x)
    assert len(y) == 2
    assert y[0].shape == (batch_size, 3)
    assert y[1].shape == (batch_size, 4)
