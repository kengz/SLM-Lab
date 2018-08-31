from slm_lab.agent.algorithm import policy_util
from slm_lab.lib import logger, util
import os
import pydash as ps
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def get_policy_out_dim(body):
    '''Helper method to construct the policy network out_dim for a body according to is_discrete, action_type'''
    if body.is_discrete:
        if body.action_type == 'multi_discrete':
            assert ps.is_list(body.action_dim), body.action_dim
            policy_out_dim = body.action_dim
        else:
            assert ps.is_integer(body.action_dim), body.action_dim
            policy_out_dim = body.action_dim
    else:
        if body.action_type == 'multi_continuous':
            assert ps.is_list(body.action_dim), body.action_dim
            raise NotImplementedError('multi_continuous not supported yet')
        else:
            assert ps.is_integer(body.action_dim), body.action_dim
            if body.action_dim == 1:
                policy_out_dim = 2  # singleton stay as int
            else:
                policy_out_dim = body.action_dim * [2]
    return policy_out_dim


def get_out_dim(body, add_critic=False):
    '''Construct the NetClass out_dim for a body according to is_discrete, action_type, and whether to add a critic unit'''
    policy_out_dim = get_policy_out_dim(body)
    if add_critic:
        if ps.is_list(policy_out_dim):
            out_dim = policy_out_dim + [1]
        else:
            out_dim = [policy_out_dim, 1]
    else:
        out_dim = policy_out_dim
    return out_dim


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
    Initializes all of the layers of type 'Linear', 'Conv', or 'GRU', using xavier uniform initialization for the weights and 0.01 for the biases, 0.0 for the biases of the GRU.
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


# lr decay methods


def no_decay(net, clock):
    '''No update'''
    return net.optim_spec['lr']


def fn_decay_lr(net, clock, fn):
    '''
    Decay learning rate for net module, only returns the new lr for user to set to appropriate nets
    In the future, might add more flexible lr adjustment, like boosting and decaying on need.
    '''
    total_t = clock.get('total_t')
    start_val, end_val = net.optim_spec['lr'], 1e-6
    anneal_total_t = net.lr_anneal_timestep or max(10e6, 60 * net.lr_decay_frequency)

    if total_t >= net.lr_decay_min_timestep and total_t % net.lr_decay_frequency == 0:
        logger.debug(f'anneal_total_t: {anneal_total_t}, total_t: {total_t}')
        new_lr = fn(start_val, end_val, anneal_total_t, total_t)
        return new_lr
    else:
        return no_decay(net, clock)


def linear_decay(net, clock):
    '''Apply _linear_decay to lr'''
    return fn_decay_lr(net, clock, policy_util._linear_decay)


def rate_decay(net, clock):
    '''Apply _rate_decay to lr'''
    return fn_decay_lr(net, clock, policy_util._rate_decay)


def periodic_decay(net, clock):
    '''Apply _periodic_decay to lr'''
    return fn_decay_lr(net, clock, policy_util._periodic_decay)


# params methods


def save(net, model_path):
    '''Save model weights to path'''
    torch.save(net.state_dict(), model_path)
    logger.info(f'Saved model to {model_path}')


def save_algorithm(algorithm, epi=None):
    '''Save all the nets for an algorithm'''
    agent = algorithm.agent
    net_names = algorithm.net_names
    prepath = util.get_prepath(agent.spec, agent.info_space, unit='session')
    if epi is not None:
        prepath = f'{prepath}_epi{epi}'
    logger.info(f'Saving algorithm {util.get_class_name(algorithm)} nets {net_names}')
    for net_name in net_names:
        net = getattr(algorithm, net_name)
        model_path = f'{prepath}_model_{net_name}.pth'
        save(net, model_path)
        optim_path = f'{prepath}_optim_{net_name}.pth'
        save(net.optim, optim_path)


def load(net, model_path):
    '''Save model weights from a path into a net module'''
    net.load_state_dict(torch.load(model_path))
    logger.info(f'Loaded model from {model_path}')


def load_algorithm(algorithm):
    '''Save all the nets for an algorithm'''
    agent = algorithm.agent
    net_names = algorithm.net_names
    prepath = util.get_prepath(agent.spec, agent.info_space, unit='session')
    logger.info(f'Loading algorithm {util.get_class_name(algorithm)} nets {net_names}')
    for net_name in net_names:
        net = getattr(algorithm, net_name)
        model_path = f'{prepath}_model_{net_name}.pth'
        load(net, model_path)
        optim_path = f'{prepath}_optim_{net_name}.pth'
        load(net.optim, optim_path)


def copy(src_net, tar_net):
    '''Copy model weights from src to target'''
    tar_net.load_state_dict(src_net.state_dict())


def polyak_update(src_net, tar_net, beta=0.5):
    '''Polyak weight update to update a target tar_net'''
    tar_params = tar_net.named_parameters()
    src_params = src_net.named_parameters()
    src_dict_params = dict(src_params)

    for name, tar_param in tar_params:
        if name in src_dict_params:
            src_dict_params[name].data.copy_(beta * tar_param.data + (1 - beta) * src_dict_params[name].data)

    tar_net.load_state_dict(src_dict_params)


def to_assert_trained():
    '''Condition for running assert_trained'''
    return os.environ.get('PY_ENV') == 'test' or util.get_lab_mode() == 'dev'


def gen_assert_trained(pre_model):
    '''
    Generate assert_trained function used to check weight updates
    @example

    assert_trained = gen_assert_trained(model)
    # ...
    loss.backward()
    optim.step()
    assert_trained(model)
    '''
    pre_weights = [param.clone() for param in pre_model.parameters()]

    def assert_trained(post_model):
        post_weights = [param.clone() for param in post_model.parameters()]
        assert not all(torch.equal(w1, w2) for w1, w2 in zip(pre_weights, post_weights)), 'Model parameter is not updated in training_step(), check if your tensor is detached from graph.'
        assert all(1e-3 < param.grad.norm() < 1e4 for param in post_model.parameters()), 'Gradient norm has extreme value < 1e-3 or > 1e4, which is bad. Check your network and loss computation. Consider using the "clip_grad" and "clip_grad_val" net parameter'
        logger.debug('Passed network weight update assertation in dev lab_mode.')
    return assert_trained


def push_global_grad(local_net, global_net):
    '''Push local gradient to global for distributed training'''
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad


def pull_global_param(local_net, global_net):
    '''Pull global param to local network for distributed training'''
    copy(local_net, global_net)
