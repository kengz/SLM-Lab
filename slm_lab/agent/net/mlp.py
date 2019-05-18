from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import logger, math_util, util
import numpy as np
import pydash as ps
import torch
import torch.nn as nn

logger = logger.get_logger(__name__)


class MLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network
    If more than 1 output tensors, will create a self.model_tails instead of making last layer part of self.model

    e.g. net_spec
    "net": {
        "type": "MLPNet",
        "shared": true,
        "hid_layers": [32],
        "hid_layers_activation": "relu",
        "out_layer_activation": null,
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
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        hid_layers: list containing dimensions of the hidden layers
        hid_layers_activation: activation function for the hidden layers
        out_layer_activation: activation function for the output layer, same shape as out_dim
        init_fn: weight initialization function
        clip_grad_val: clip gradient norm if value is not None
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_scheduler_spec: Pytorch optim.lr_scheduler
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        nn.Module.__init__(self)
        super().__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            out_layer_activation=None,
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'shared',
            'hid_layers',
            'hid_layers_activation',
            'out_layer_activation',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        dims = [self.in_dim] + self.hid_layers
        self.model = net_util.build_fc_model(dims, self.hid_layers_activation)
        # add last layer with no activation
        # tails. avoid list for single-tail for compute speed
        if ps.is_integer(self.out_dim):
            self.model_tail = net_util.build_fc_model([dims[-1], self.out_dim], self.out_layer_activation)
        else:
            if not ps.is_list(self.out_layer_activation):
                self.out_layer_activation = [self.out_layer_activation] * len(out_dim)
            assert len(self.out_layer_activation) == len(self.out_dim)
            tails = []
            for out_d, out_activ in zip(self.out_dim, self.out_layer_activation):
                tail = net_util.build_fc_model([dims[-1], out_d], out_activ)
                tails.append(tail)
            self.model_tails = nn.ModuleList(tails)

        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)

    def forward(self, x):
        '''The feedforward step'''
        x = self.model(x)
        if hasattr(self, 'model_tails'):
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(x))
            return outs
        else:
            return self.model_tail(x)

    @net_util.dev_check_training_step
    def training_step(self, loss, optim, lr_scheduler, lr_clock=None):
        '''Train a network given a computed loss'''
        lr_scheduler.step(epoch=ps.get(lr_clock, 'total_t'))
        optim.zero_grad()
        loss.backward()
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        if hasattr(self, 'global_net'):
            net_util.push_global_grads(self, self.global_net)
        optim.step()
        if hasattr(self, 'global_net'):
            net_util.copy(self.global_net, self)
        lr_clock.tick('grad_step')
        return loss


class HydraMLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network with multiple state and action heads, and a single shared body.

    e.g. net_spec
    "net": {
        "type": "HydraMLPNet",
        "shared": true,
        "hid_layers": [
            [[32],[32]], # 2 heads with hidden layers
            [64], # body
            [] # tail, no hidden layers
        ],
        "hid_layers_activation": "relu",
        "out_layer_activation": null,
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
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        Multi state processing heads, single shared body, and multi action tails.
        There is one state and action head per body/environment
        Example:

          env 1 state       env 2 state
         _______|______    _______|______
        |    head 1    |  |    head 2    |
        |______________|  |______________|
                |                  |
                |__________________|
         ________________|_______________
        |          Shared body           |
        |________________________________|
                         |
                 ________|_______
                |                |
         _______|______    ______|_______
        |    tail 1    |  |    tail 2    |
        |______________|  |______________|
                |                |
           env 1 action      env 2 action
        '''
        nn.Module.__init__(self)
        super().__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            out_layer_activation=None,
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'out_layer_activation',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])
        assert len(self.hid_layers) == 3, 'Your hidden layers must specify [*heads], [body], [*tails]. If not, use MLPNet'
        assert isinstance(self.in_dim, list), 'Hydra network needs in_dim as list'
        assert isinstance(self.out_dim, list), 'Hydra network needs out_dim as list'
        self.head_hid_layers = self.hid_layers[0]
        self.body_hid_layers = self.hid_layers[1]
        self.tail_hid_layers = self.hid_layers[2]
        if len(self.head_hid_layers) == 1:
            self.head_hid_layers = self.head_hid_layers * len(self.in_dim)
        if len(self.tail_hid_layers) == 1:
            self.tail_hid_layers = self.tail_hid_layers * len(self.out_dim)

        self.model_heads = self.build_model_heads(in_dim)
        heads_out_dim = np.sum([head_hid_layers[-1] for head_hid_layers in self.head_hid_layers])
        dims = [heads_out_dim] + self.body_hid_layers
        self.model_body = net_util.build_fc_model(dims, self.hid_layers_activation)
        self.model_tails = self.build_model_tails(self.out_dim, self.out_layer_activation)

        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)

    def build_model_heads(self, in_dim):
        '''Build each model_head. These are stored as Sequential models in model_heads'''
        assert len(self.head_hid_layers) == len(in_dim), 'Hydra head hid_params inconsistent with number in dims'
        model_heads = nn.ModuleList()
        for in_d, hid_layers in zip(in_dim, self.head_hid_layers):
            dims = [in_d] + hid_layers
            model_head = net_util.build_fc_model(dims, self.hid_layers_activation)
            model_heads.append(model_head)
        return model_heads

    def build_model_tails(self, out_dim, out_layer_activation):
        '''Build each model_tail. These are stored as Sequential models in model_tails'''
        if not ps.is_list(out_layer_activation):
            out_layer_activation = [out_layer_activation] * len(out_dim)
        model_tails = nn.ModuleList()
        if ps.is_empty(self.tail_hid_layers):
            for out_d, out_activ in zip(out_dim, out_layer_activation):
                tail = net_util.build_fc_model([self.body_hid_layers[-1], out_d], out_activ)
                model_tails.append(tail)
        else:
            assert len(self.tail_hid_layers) == len(out_dim), 'Hydra tail hid_params inconsistent with number out dims'
            for out_d, out_activ, hid_layers in zip(out_dim, out_layer_activation, self.tail_hid_layers):
                dims = hid_layers
                model_tail = net_util.build_fc_model(dims, self.hid_layers_activation)
                tail_out = net_util.build_fc_model([dims[-1], out_d], out_activ)
                model_tail.add_module(str(len(model_tail)), tail_out)
                model_tails.append(model_tail)
        return model_tails

    def forward(self, xs):
        '''The feedforward step'''
        head_xs = []
        for model_head, x in zip(self.model_heads, xs):
            head_xs.append(model_head(x))
        head_xs = torch.cat(head_xs, dim=-1)
        body_x = self.model_body(head_xs)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(body_x))
        return outs

    @net_util.dev_check_training_step
    def training_step(self, loss, optim, lr_scheduler, lr_clock=None):
        lr_scheduler.step(epoch=ps.get(lr_clock, 'total_t'))
        optim.zero_grad()
        loss.backward()
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        if hasattr(self, 'global_net'):
            net_util.push_global_grads(self, self.global_net)
        optim.step()
        if hasattr(self, 'global_net'):
            net_util.copy(self.global_net, self)
        lr_clock.tick('grad_step')
        return loss


class DuelingMLPNet(MLPNet):
    '''
    Class for generating arbitrary sized feedforward neural network, with dueling heads. Intended for Q-Learning algorithms only.
    Implementation based on "Dueling Network Architectures for Deep Reinforcement Learning" http://proceedings.mlr.press/v48/wangf16.pdf

    e.g. net_spec
    "net": {
        "type": "DuelingMLPNet",
        "shared": true,
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
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'shared',
            'hid_layers',
            'hid_layers_activation',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        # Guard against inappropriate algorithms and environments
        # Build model body
        dims = [self.in_dim] + self.hid_layers
        self.model_body = net_util.build_fc_model(dims, self.hid_layers_activation)
        # output layers
        self.v = nn.Linear(dims[-1], 1)  # state value
        self.adv = nn.Linear(dims[-1], out_dim)  # action dependent raw advantage
        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)

    def forward(self, x):
        '''The feedforward step'''
        x = self.model_body(x)
        state_value = self.v(x)
        raw_advantages = self.adv(x)
        out = math_util.calc_q_value_logits(state_value, raw_advantages)
        return out
