from copy import deepcopy
from slm_lab.agent.net import net_util
from slm_lab.agent.net.mlp import MLPNet
import torch
import torch.nn as nn

net_spec = {
    "type": "MLPNet",
    "shared": True,
    "hid_layers": [32],
    "hid_layers_activation": "relu",
    "init_fn": "xavier_uniform_",
    "clip_grad_val": 1.0,
    "loss_spec": {
        "name": "MSELoss"
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
    "update_type": "replace",
    "update_frequency": 1,
    "polyak_coef": 0.9,
    "gpu": True
}
in_dim = 10
out_dim = 3
batch_size = 16
net = MLPNet(net_spec, in_dim, out_dim)
# init net optimizer and its lr scheduler
optim = net_util.get_optim(net, net.optim_spec)
lr_scheduler = net_util.get_lr_scheduler(optim, net.lr_scheduler_spec)
x = torch.rand((batch_size, in_dim))


def test_init():
    net = MLPNet(net_spec, in_dim, out_dim)
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'model')
    assert hasattr(net, 'model_tail')
    assert not hasattr(net, 'model_tails')


def test_forward():
    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_train_step():
    y = torch.rand((batch_size, out_dim))
    loss = net.loss_fn(net.forward(x), y)
    net.train_step(loss, optim, lr_scheduler)
    assert loss != 0.0


def test_no_lr_scheduler():
    nopo_lrs_net_spec = deepcopy(net_spec)
    nopo_lrs_net_spec['lr_scheduler_spec'] = None
    net = MLPNet(nopo_lrs_net_spec, in_dim, out_dim)
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'model')
    assert hasattr(net, 'model_tail')
    assert not hasattr(net, 'model_tails')

    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_multitails():
    net = MLPNet(net_spec, in_dim, [3, 4])
    assert isinstance(net, nn.Module)
    assert hasattr(net, 'model')
    assert not hasattr(net, 'model_tail')
    assert hasattr(net, 'model_tails')
    assert len(net.model_tails) == 2

    y = net.forward(x)
    assert len(y) == 2
    assert y[0].shape == (batch_size, 3)
    assert y[1].shape == (batch_size, 4)


def test_layer_norm():
    '''Test MLPNet with layer normalization enabled'''
    ln_net_spec = deepcopy(net_spec)
    ln_net_spec['layer_norm'] = True
    ln_net_spec['hid_layers'] = [32, 32]
    net = MLPNet(ln_net_spec, in_dim, out_dim)

    assert isinstance(net, nn.Module)
    # Check that LayerNorm layers are present in the model
    layer_norm_count = sum(1 for m in net.model.modules() if isinstance(m, nn.LayerNorm))
    assert layer_norm_count == 2, f'Expected 2 LayerNorm layers, got {layer_norm_count}'

    # Test forward pass works
    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)

    # Test training step works
    target = torch.rand((batch_size, out_dim))
    loss = net.loss_fn(net.forward(x), target)
    optim = net_util.get_optim(net, net.optim_spec)
    net.train_step(loss, optim, None)
    assert loss != 0.0
