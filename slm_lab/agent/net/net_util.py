from functools import partial
import pydash as _
import torch
import torch.nn.functional as F


def flatten_params(net):
    '''Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2'''
    return torch.cat([param.data.view(-1) for param in net.parameters()], 0)


def load_params(net, flattened):
    '''Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2'''
    offset = 0
    for param in net.parameters():
        param.data.copy_(
            flattened[offset:offset + param.nelement()]).view(param.size())
        offset += param.nelement()
    return net


def set_optim(cls, optim_param):
    '''Helper to parse optim param and construct optim for net'''
    optim_param = optim_param or {}
    OptimClass = getattr(torch.optim, _.get(optim_param, 'name', 'Adam'))
    optim_param.pop('name', None)
    optim = OptimClass(cls.parameters(), **optim_param)
    return optim


def set_loss_fn(cls, loss_param):
    '''Helper to parse loss param and construct loss_fn for net'''
    loss_param = loss_param or {}
    loss_fn = getattr(F, _.get(loss_param, 'name', 'mse_loss'))
    loss_param.pop('name', None)
    if not _.is_empty(loss_param):
        loss_fn = partial(loss_fn, **loss_param)
    return loss_fn


def set_activation_fn(cls, activation_param):
    '''Helper to parse activation function param and construct activation_fn for net'''
    activation_fn = getattr(F, activation_param, 'relu')
    return activation_fn
