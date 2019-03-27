from functools import partial
from slm_lab import ROOT_DIR
from slm_lab.lib import logger, util
import os
import pydash as ps
import torch
import torch.nn as nn


NN_LOWCASE_LOOKUP = {nn_name.lower(): nn_name for nn_name in nn.__dict__}
logger = logger.get_logger(__name__)


class NoOpLRScheduler:
    '''Symbolic LRScheduler class for API consistency'''

    def __init__(self, optim):
        self.optim = optim

    def step(self, epoch=None):
        pass

    def get_lr(self):
        return self.optim.defaults['lr']


def build_sequential(dims, activation):
    '''Build the Sequential model by interleaving nn.Linear and activation_fn'''
    assert len(dims) >= 2, 'dims need to at least contain input, output'
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


def get_lr_scheduler(cls, lr_scheduler_spec):
    '''Helper to parse lr_scheduler param and construct Pytorch optim.lr_scheduler'''
    if ps.is_empty(lr_scheduler_spec):
        lr_scheduler = NoOpLRScheduler(cls.optim)
    else:
        LRSchedulerClass = getattr(torch.optim.lr_scheduler, lr_scheduler_spec['name'])
        lr_scheduler_spec = ps.omit(lr_scheduler_spec, 'name')
        lr_scheduler = LRSchedulerClass(cls.optim, **lr_scheduler_spec)
    return lr_scheduler


def get_policy_out_dim(body):
    '''Helper method to construct the policy network out_dim for a body according to is_discrete, action_type'''
    action_dim = body.action_dim
    if body.is_discrete:
        if body.action_type == 'multi_discrete':
            assert ps.is_list(action_dim), action_dim
            policy_out_dim = action_dim
        else:
            assert ps.is_integer(action_dim), action_dim
            policy_out_dim = action_dim
    else:
        if body.action_type == 'multi_continuous':
            assert ps.is_list(action_dim), action_dim
            raise NotImplementedError('multi_continuous not supported yet')
        else:
            assert ps.is_integer(action_dim), action_dim
            if action_dim == 1:
                policy_out_dim = 2  # singleton stay as int
            else:
                policy_out_dim = action_dim * [2]
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


def init_layers(net, init_fn):
    if init_fn is None:
        return
    if init_fn == 'xavier_uniform_':
        try:
            gain = nn.init.calculate_gain(net.hid_layers_activation)
        except ValueError:
            gain = 1
        init_fn = partial(nn.init.xavier_uniform_, gain=gain)
    elif 'kaiming' in init_fn:
        assert net.hid_layers_activation in ['relu', 'leaky_relu'], f'Kaiming initialization not supported for {net.hid_layers_activation}'
        init_fn = nn.init.__dict__[init_fn]
        init_fn = partial(init_fn, nonlinearity=net.hid_layers_activation)
    else:
        init_fn = nn.init.__dict__[init_fn]
    net.apply(partial(init_parameters, init_fn=init_fn))


def init_parameters(module, init_fn):
    '''
    Initializes module's weights using init_fn, which is the name of function from from nn.init
    Initializes module's biases to either 0.01 or 0.0, depending on module
    The only exception is BatchNorm layers, for which we use uniform initialization
    '''
    bias_init = 0.01
    classname = module.__class__.__name__
    if 'BatchNorm' in classname:
        init_fn(module.weight)
        nn.init.constant_(module.bias, bias_init)
    elif 'GRU' in classname:
        for name, param in module.named_parameters():
            if 'weight' in name:
                init_fn(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    elif 'Linear' in classname or ('Conv' in classname and 'Net' not in classname):
        init_fn(module.weight)
        nn.init.constant_(module.bias, bias_init)


# params methods


def save(net, model_path):
    '''Save model weights to path'''
    torch.save(net.state_dict(), util.smart_path(model_path))
    logger.info(f'Saved model to {model_path}')


def save_algorithm(algorithm, ckpt=None):
    '''Save all the nets for an algorithm'''
    agent = algorithm.agent
    net_names = algorithm.net_names
    prepath = util.get_prepath(agent.spec, agent.info_space, unit='session')
    if ckpt is not None:
        prepath = f'{prepath}_ckpt-{ckpt}'
    logger.info(f'Saving algorithm {util.get_class_name(algorithm)} nets {net_names}')
    for net_name in net_names:
        net = getattr(algorithm, net_name)
        model_path = f'{prepath}_{net_name}_model.pth'
        save(net, model_path)
        optim_path = f'{prepath}_{net_name}_optim.pth'
        save(net.optim, optim_path)


def load(net, model_path):
    '''Save model weights from a path into a net module'''
    device = None if torch.cuda.is_available() else 'cpu'
    net.load_state_dict(torch.load(util.smart_path(model_path), map_location=device))
    logger.info(f'Loaded model from {model_path}')


def load_algorithm(algorithm):
    '''Save all the nets for an algorithm'''
    agent = algorithm.agent
    net_names = algorithm.net_names
    if util.in_eval_lab_modes():
        # load specific model in eval mode
        prepath = agent.info_space.eval_model_prepath
    else:
        prepath = util.get_prepath(agent.spec, agent.info_space, unit='session')
    logger.info(f'Loading algorithm {util.get_class_name(algorithm)} nets {net_names}')
    for net_name in net_names:
        net = getattr(algorithm, net_name)
        model_path = f'{prepath}_{net_name}_model.pth'
        load(net, model_path)
        optim_path = f'{prepath}_{net_name}_optim.pth'
        load(net.optim, optim_path)


def copy(src_net, tar_net):
    '''Copy model weights from src to target'''
    tar_net.load_state_dict(src_net.state_dict())


def polyak_update(src_net, tar_net, retain_ratio=0.5):
    '''
    Polyak weight update to update a target tar_net, retain old weights by its ratio, i.e.
    target <- retain_ratio * source + (1 - retain_ratio) * target
    '''
    for src_param, tar_param in zip(src_net.parameters(), tar_net.named_parameters()):
        tar_param.data.copy_(retain_ratio * src_param.data + (1.0 - retain_ratio) * tar_param.data)


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
    assert_trained(model, loss)
    '''
    pre_weights = [param.clone() for param in pre_model.parameters()]

    def assert_trained(post_model, loss):
        post_weights = [param.clone() for param in post_model.parameters()]
        if loss == 0:
            # TODO if without momentum, weights should not change too
            for p_name, param in post_model.named_parameters():
                assert param.grad.norm() == 0
        else:
            assert not all(torch.equal(w1, w2) for w1, w2 in zip(pre_weights, post_weights)), f'Model parameter is not updated in training_step(), check if your tensor is detached from graph. loss: {loss}'
            min_norm = 0
            max_norm = 1e5
            for p_name, param in post_model.named_parameters():
                try:
                    assert min_norm < param.grad.norm() < max_norm, f'Gradient norm fails the extreme value check {min_norm} < {p_name}:{param.grad.norm()} < {max_norm}, which is bad. Loss: {loss}. Check your network and loss computation. Consider using the "clip_grad_val" net parameter.'
                except Exception as e:
                    logger.warn(e)
        logger.debug('Passed network weight update assertation in dev lab_mode.')
    return assert_trained


def get_grad_norms(algorithm):
    '''Gather all the net's grad norms of an algorithm for debugging'''
    grad_norms = []
    for net_name in algorithm.net_names:
        net = getattr(algorithm, net_name)
        if net.grad_norms is not None:
            grad_norms.extend(net.grad_norms)
    return grad_norms
