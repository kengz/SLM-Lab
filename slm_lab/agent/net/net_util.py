import pydash as ps
import torch
import torch.nn as nn
import torch.nn.functional as F
from slm_lab.lib import logger


NN_LOWCASE_LOOKUP = {nn_name.lower(): nn_name for nn_name in nn.__dict__}
logger = logger.get_logger(__name__)


def build_sequential(dims, activation):
    '''Build the Sequential model by interleaving nn.Linear and activation_fn'''
    dim_pairs = list(zip(dims[:-1], dims[1:]))
    layers = []
    for in_d, out_d in dim_pairs:
        layers.append(nn.Linear(in_d, out_d))
        layers.append(get_activation_fn(activation))
    model = nn.Sequential(*layers)
    return model


def get_activation_fn(activation):
    '''Helper to generate activation function layers for net'''
    nn_name = NN_LOWCASE_LOOKUP.get(activation) or NN_LOWCASE_LOOKUP['relu']
    ActivationClass = getattr(nn, nn_name)
    return ActivationClass()


def get_loss_fn(cls, loss_spec):
    '''Helper to parse loss param and construct loss_fn for net'''
    LossClass = getattr(nn, loss_spec['name'])
    loss_spec = ps.omit(loss_spec, 'name')
    loss_fn = LossClass(**loss_spec)
    return loss_fn


def get_optim(cls, optim_spec):
    '''Helper to parse optim param and construct optim for net'''
    OptimClass = getattr(torch.optim, optim_spec['name'])
    optim_spec = ps.omit(optim_spec, 'name')
    optim = OptimClass(cls.parameters(), **optim_spec)
    return optim


def init_gru_layer(layer):
    '''Initializes a GRU layer in with xavier_uniform initialization and 0 biases'''
    for layer_p in layer._all_weights:
        for p in layer_p:
            if 'weight' in p:
                torch.nn.init.xavier_uniform_(layer.__getattr__(p))
            elif 'bias' in p:
                torch.nn.init.constant_(layer.__getattr__(p), 0.0)


def init_layers(layers):
    '''
    Initializes all of the layers of type 'Linear', 'Conv', or GRU, using xavier uniform initialization for the weights and 0.01 for the biases, 0.0 for the biases of the GRU.
    Initializes all layers of type 'BatchNorm' using uniform initialization for the weights and the same as above for the biases
    '''
    bias_init = 0.01
    for layer in layers:
        classname = layer.__class__.__name__
        if 'BatchNorm' in classname:
            torch.nn.init.uniform_(layer.weight.data)
            torch.nn.init.constant_(layer.bias.data, bias_init)
        elif 'GRU' in classname:
            init_gru_layer(layer)
        elif 'Linear' in classname:
            torch.nn.init.xavier_uniform_(layer.weight.data)
            torch.nn.init.constant_(layer.bias.data, bias_init)
        else:
            pass


def save(net, model_path):
    '''Save model weights to path'''
    torch.save(net.state_dict(), model_path)


def load(net, model_path):
    '''Save model weights from a path into a net module'''
    net.load_state_dict(torch.load(model_path))


# params methods

def copy_trainable_params(net):
    return [param.clone() for param in net.parameters()]


def copy_fixed_params(net):
    return None


def get_grad_norms(net):
    '''Returns a list of the norm of the gradients for all parameters'''
    norms = []
    for i, param in enumerate(net.parameters()):
        grad_norm = torch.norm(param.grad.data)
        if grad_norm is None:
            logger.info(f'Param with None grad: {param}, layer: {i}')
        norms.append(grad_norm)
    return norms


def flatten_params(net):
    '''Flattens all of the parameters in a net
    Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2'''
    return torch.cat([param.data.view(-1) for param in net.parameters()], 0)


def load_params(net, flattened):
    '''Loads flattened parameters into a net
    Source: https://discuss.pytorch.org/t/running-average-of-parameters/902/2'''
    offset = 0
    for param in net.parameters():
        param.data.copy_(flattened[offset:offset + param.nelement()]).view(param.size())
        offset += param.nelement()
    return net


def polyak_update(src_net, tar_net, beta=0.5):
    '''Polyak weight update to update a target tar_net'''
    tar_params = tar_net.named_parameters()
    src_params = src_net.named_parameters()
    src_dict_params = dict(src_params)

    for name, tar_param in tar_params:
        if name in src_dict_params:
            src_dict_params[name].data.copy_(beta * tar_param.data + (1 - beta) * src_dict_params[name].data)

    tar_net.load_state_dict(src_dict_params)
