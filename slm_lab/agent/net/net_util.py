from functools import partial
import pydash as _
import torch
import torch.nn as nn
import torch.nn.functional as F
from slm_lab.lib import logger


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


def flatten_params(net):
    '''Flattens all of the parameters in a net
    Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2'''
    return torch.cat([param.data.view(-1) for param in net.parameters()], 0)


def load_params(net, flattened):
    '''Loads flattened parameters into a net
    Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2'''
    offset = 0
    for param in net.parameters():
        param.data.copy_(
            flattened[offset:offset + param.nelement()]).view(param.size())
        offset += param.nelement()
    return net


def init_gru_layer(gru):
    '''Initializes a GRU layer in with xavier_uniform initialization and 0 biases'''
    for layer_p in gru._all_weights:
        for p in layer_p:
            if 'weight' in p:
                torch.nn.init.xavier_uniform(gru.__getattr__(p))
            elif 'bias' in p:
                torch.nn.init.constant(gru.__getattr__(p), 0.0)


def init_layers(layers, layer_type):
    '''
    Initializes all of the layers of type 'Linear', 'Conv', or GRU, using xavier uniform initialization for the weights and 0.01 for the biases, 0.0 for the biases of the GRU.
    Initializes all layers of type 'BatchNorm' using uniform initialization for the weights and the same as above for the biases
    '''
    biasinit = 0.01
    for layer in layers:
        classname = layer.__class__.__name__
        if classname.find(layer_type) != -1:
            if layer_type == 'BatchNorm':
                torch.nn.init.uniform(layer.weight.data)
                torch.nn.init.constant(layer.bias.data, biasinit)
            elif layer_type == 'GRU':
                init_gru_layer(layer)
            else:
                torch.nn.init.xavier_uniform(layer.weight.data)
                torch.nn.init.constant(layer.bias.data, biasinit)
