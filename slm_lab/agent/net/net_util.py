from functools import partial
import pydash as _
import torch
import torch.nn as nn
import torch.nn.functional as F
from slm_lab.lib import logger


def flatten_params(net):
    '''Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2'''
    return torch.cat([param.data.view(-1) for param in net.parameters()], 0)


def get_activation_fn(activation):
    '''Helper to generate activation function layers for net'''
    layer = None
    if activation == 'sigmoid':
        layer = nn.Sigmoid()
    elif activation == 'lrelu':
        layer = nn.LeakyReLU(negative_slope=0.05)
    elif activation == 'tanh':
        layer = nn.Tanh()
    elif activation == 'relu':
        layer = nn.ReLU()
    else:
        logger.debug("No activation fn or unrecognised activation fn")
        layer = nn.ReLU()
    return layer


def get_loss_fn(cls, loss_param):
    '''Helper to parse loss param and construct loss_fn for net'''
    loss_param = loss_param or {}
    loss_fn = getattr(F, _.get(loss_param, 'name', 'mse_loss'))
    loss_param = _.omit(loss_param, 'name')
    if not _.is_empty(loss_param):
        loss_fn = partial(loss_fn, **loss_param)
    return loss_fn


def get_optim(cls, optim_param):
    '''Helper to parse optim param and construct optim for net'''
    optim_param = optim_param or {}
    OptimClass = getattr(torch.optim, _.get(optim_param, 'name', 'Adam'))
    optim_param = _.omit(optim_param, 'name')
    optim = OptimClass(cls.parameters(), **optim_param)
    return optim


def get_optim_multinet(params, optim_param):
    '''Helper to parse optim param and construct optim for net'''
    optim_param = optim_param or {}
    OptimClass = getattr(torch.optim, _.get(optim_param, 'name', 'Adam'))
    optim_param.pop('name', None)
    optim = OptimClass(params, **optim_param)
    return optim


def load_params(net, flattened):
    '''Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2'''
    offset = 0
    for param in net.parameters():
        param.data.copy_(
            flattened[offset:offset + param.nelement()]).view(param.size())
        offset += param.nelement()
    return net
