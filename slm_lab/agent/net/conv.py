from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import logger, math_util, util
import numpy as np
import pydash as ps
import torch
import torch.nn as nn

logger = logger.get_logger(__name__)


class ConvNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized convolutional neural network,
    with optional batch normalization

    Assumes that a single input example is organized into a 3D tensor.
    The entire model consists of three parts:
        1. self.conv_model
        2. self.fc_model
        3. self.model_tails

    e.g. net_spec
    "net": {
        "type": "ConvNet",
        "shared": true,
        "conv_hid_layers": [
            [32, 8, 4, 0, 1],
            [64, 4, 2, 0, 1],
            [64, 3, 1, 0, 1]
        ],
        "fc_hid_layers": [512],
        "hid_layers_activation": "relu",
        "out_layer_activation": "tanh",
        "init_fn": null,
        "normalize": false,
        "batch_norm": false,
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
        "update_type": "replace",
        "update_frequency": 10000,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        conv_hid_layers: list containing dimensions of the convolutional hidden layers, each is a list representing hid_layer = out_d, kernel, stride, padding, dilation.
            Asssumed to all come before the flat layers.
            Note: a convolutional layer should specify the in_channel, out_channels, kernel_size, stride (of kernel steps), padding, and dilation (spacing between kernel points) E.g. [3, 16, (5, 5), 1, 0, (2, 2)]
            For more details, see http://pytorch.org/docs/master/nn.html#conv2d and https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        fc_hid_layers: list of fc layers following the convolutional layers
        hid_layers_activation: activation function for the hidden layers
        out_layer_activation: activation function for the output layer, same shape as out_dim
        init_fn: weight initialization function
        normalize: whether to divide by 255.0 to normalize image input
        batch_norm: whether to add batch normalization after each convolutional layer, excluding the input layer.
        clip_grad_val: clip gradient norm if value is not None
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_scheduler_spec: Pytorch optim.lr_scheduler
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        assert len(in_dim) == 3  # image shape (c,w,h)
        nn.Module.__init__(self)
        super().__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            out_layer_activation=None,
            init_fn=None,
            normalize=False,
            batch_norm=True,
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
            'conv_hid_layers',
            'fc_hid_layers',
            'hid_layers_activation',
            'out_layer_activation',
            'init_fn',
            'normalize',
            'batch_norm',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        # conv body
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        self.conv_out_dim = self.get_conv_output_size()

        # fc body
        if ps.is_empty(self.fc_hid_layers):
            tail_in_dim = self.conv_out_dim
        else:
            # fc body from flattened conv
            self.fc_model = net_util.build_fc_model([self.conv_out_dim] + self.fc_hid_layers, self.hid_layers_activation)
            tail_in_dim = self.fc_hid_layers[-1]

        # tails. avoid list for single-tail for compute speed
        if ps.is_integer(self.out_dim):
            self.model_tail = net_util.build_fc_model([tail_in_dim, self.out_dim], self.out_layer_activation)
        else:
            if not ps.is_list(self.out_layer_activation):
                self.out_layer_activation = [self.out_layer_activation] * len(out_dim)
            assert len(self.out_layer_activation) == len(self.out_dim)
            tails = []
            for out_d, out_activ in zip(self.out_dim, self.out_layer_activation):
                tail = net_util.build_fc_model([tail_in_dim, out_d], out_activ)
                tails.append(tail)
            self.model_tails = nn.ModuleList(tails)

        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def get_conv_output_size(self):
        '''Helper function to calculate the size of the flattened features after the final convolutional layer'''
        with torch.no_grad():
            x = torch.ones(1, *self.in_dim)
            x = self.conv_model(x)
            return x.numel()

    def build_conv_layers(self, conv_hid_layers):
        '''
        Builds all of the convolutional layers in the network and store in a Sequential model
        '''
        conv_layers = []
        in_d = self.in_dim[0]  # input channel
        for i, hid_layer in enumerate(conv_hid_layers):
            hid_layer = [tuple(e) if ps.is_list(e) else e for e in hid_layer]  # guard list-to-tuple
            # hid_layer = out_d, kernel, stride, padding, dilation
            conv_layers.append(nn.Conv2d(in_d, *hid_layer))
            if self.hid_layers_activation is not None:
                conv_layers.append(net_util.get_activation_fn(self.hid_layers_activation))
            # Don't include batch norm in the first layer
            if self.batch_norm and i != 0:
                conv_layers.append(nn.BatchNorm2d(in_d))
            in_d = hid_layer[0]  # update to out_d
        conv_model = nn.Sequential(*conv_layers)
        return conv_model

    def forward(self, x):
        '''
        The feedforward step
        Note that PyTorch takes (c,h,w) but gym provides (h,w,c), so preprocessing must be done before passing to network
        '''
        if self.normalize:
            x = x / 255.0
        x = self.conv_model(x)
        x = x.view(x.size(0), -1)  # to (batch_size, -1)
        if hasattr(self, 'fc_model'):
            x = self.fc_model(x)
        # return tensor if single tail, else list of tail tensors
        if hasattr(self, 'model_tails'):
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(x))
            return outs
        else:
            return self.model_tail(x)

    @net_util.dev_check_train_step
    def train_step(self, loss, optim, lr_scheduler, lr_clock=None, global_net=None):
        lr_scheduler.step(epoch=ps.get(lr_clock, 'total_t'))
        optim.zero_grad()
        loss.backward()
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        if global_net is not None:
            net_util.push_global_grads(self, global_net)
        optim.step()
        if global_net is not None:
            net_util.copy(global_net, self)
        lr_clock.tick('grad_step')
        return loss


class DuelingConvNet(ConvNet):
    '''
    Class for generating arbitrary sized convolutional neural network,
    with optional batch normalization, and with dueling heads. Intended for Q-Learning algorithms only.
    Implementation based on "Dueling Network Architectures for Deep Reinforcement Learning" http://proceedings.mlr.press/v48/wangf16.pdf

    Assumes that a single input example is organized into a 3D tensor.
    The entire model consists of three parts:
        1. self.conv_model
        2. self.fc_model
        3. self.model_tails

    e.g. net_spec
    "net": {
        "type": "DuelingConvNet",
        "shared": true,
        "conv_hid_layers": [
            [32, 8, 4, 0, 1],
            [64, 4, 2, 0, 1],
            [64, 3, 1, 0, 1]
        ],
        "fc_hid_layers": [512],
        "hid_layers_activation": "relu",
        "init_fn": "xavier_uniform_",
        "normalize": false,
        "batch_norm": false,
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
        "update_type": "replace",
        "update_frequency": 10000,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        assert len(in_dim) == 3  # image shape (c,w,h)
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            init_fn=None,
            normalize=False,
            batch_norm=False,
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
            'conv_hid_layers',
            'fc_hid_layers',
            'hid_layers_activation',
            'init_fn',
            'normalize',
            'batch_norm',
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
        assert isinstance(out_dim, int)

        # conv body
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        self.conv_out_dim = self.get_conv_output_size()

        # fc body
        if ps.is_empty(self.fc_hid_layers):
            tail_in_dim = self.conv_out_dim
        else:
            # fc layer from flattened conv
            self.fc_model = net_util.build_fc_model([self.conv_out_dim] + self.fc_hid_layers, self.hid_layers_activation)
            tail_in_dim = self.fc_hid_layers[-1]

        # tails. avoid list for single-tail for compute speed
        self.v = nn.Linear(tail_in_dim, 1)  # state value
        self.adv = nn.Linear(tail_in_dim, out_dim)  # action dependent raw advantage
        self.model_tails = nn.ModuleList(self.v, self.adv)

        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def forward(self, x):
        '''The feedforward step'''
        if self.normalize:
            x = x / 255.0
        x = self.conv_model(x)
        x = x.view(x.size(0), -1)  # to (batch_size, -1)
        if hasattr(self, 'fc_model'):
            x = self.fc_model(x)
        state_value = self.v(x)
        raw_advantages = self.adv(x)
        out = math_util.calc_q_value_logits(state_value, raw_advantages)
        return out
