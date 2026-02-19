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
    "loss_spec": {"name": "MSELoss"},
    "optim_spec": {"name": "Adam", "lr": 0.02},
    "lr_scheduler_spec": {"name": "StepLR", "step_size": 30, "gamma": 0.1},
    "update_type": "replace",
    "update_frequency": 1,
    "polyak_coef": 0.9,
    "gpu": True,
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
    assert hasattr(net, "model")
    assert hasattr(net, "tails")
    assert not isinstance(net.tails, nn.ModuleList)


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
    nopo_lrs_net_spec["lr_scheduler_spec"] = None
    net = MLPNet(nopo_lrs_net_spec, in_dim, out_dim)
    assert isinstance(net, nn.Module)
    assert hasattr(net, "model")
    assert hasattr(net, "tails")
    assert not isinstance(net.tails, nn.ModuleList)

    y = net.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_multitails():
    net = MLPNet(net_spec, in_dim, [3, 4])
    assert isinstance(net, nn.Module)
    assert hasattr(net, "model")
    assert hasattr(net, "tails")
    assert isinstance(net.tails, nn.ModuleList)
    assert len(net.tails) == 2

    y = net.forward(x)
    assert len(y) == 2
    assert y[0].shape == (batch_size, 3)
    assert y[1].shape == (batch_size, 4)


# layer_norm tests


def test_layer_norm_false_no_layernorm():
    """layer_norm=False (default) should not include LayerNorm modules"""
    spec = {**net_spec, "layer_norm": False}
    mlp = MLPNet(spec, in_dim, out_dim)
    has_ln = any(isinstance(m, nn.LayerNorm) for m in mlp.model.modules())
    assert not has_ln


def test_layer_norm_true_has_layernorm():
    """layer_norm=True should add LayerNorm layers in model"""
    spec = {**net_spec, "layer_norm": True}
    mlp = MLPNet(spec, in_dim, out_dim)
    ln_layers = [m for m in mlp.model.modules() if isinstance(m, nn.LayerNorm)]
    assert len(ln_layers) > 0


def test_layer_norm_forward_shape_unchanged():
    """Output shape should be the same regardless of layer_norm setting"""
    spec = {**net_spec, "layer_norm": True}
    mlp = MLPNet(spec, in_dim, out_dim)
    y = mlp.forward(x)
    assert y.shape == (batch_size, out_dim)


def test_build_fc_model_layer_norm_layer_count():
    """build_fc_model with layer_norm=True should have more layers (Linear + LayerNorm + activation)"""
    from slm_lab.agent.net.net_util import build_fc_model

    model_no_ln = build_fc_model([in_dim, 32, 16], "relu", layer_norm=False)
    model_ln = build_fc_model([in_dim, 32, 16], "relu", layer_norm=True)
    # With layer_norm, each dim pair gets: Linear + LayerNorm + Activation = 3 layers
    # Without: Linear + Activation = 2 layers
    assert len(model_ln) > len(model_no_ln)
