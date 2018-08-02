from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import logger, util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logger.get_logger(__name__)


class ConvNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized convolutional neural network,
    with optional batch normalization

    Assumes that a single input example is organized into a 3D tensor.
    The entire model consists of three parts:
        1. self.conv_model
        2. self.dense_model
        3. self.model_tails

    e.g. net_spec
    "net": {
        "type": "ConvNet",
        "hid_layers": [
          [
            [4, 32, [8, 8], 4, 0, [1, 1]],
            [32, 64, [4, 4], 2, 0, [1, 1]],
            [64, 64, [3, 3], 1, 0, [1, 1]]
          ],
          [512]
        ],
        "hid_layers_activation": "relu",
        "batch_norm": false,
        "clip_grad": false,
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "SmoothL1Loss"
        },
        "optim_spec": {
          "name": "RMSprop",
          "lr": 0.00025,
          "alpha": 0.95,
          "eps": 0.01,
          "momentum": 0.0,
          "centered": true
        },
        "lr_decay": "no_decay",
        "lr_decay_frequency": 400,
        "lr_decay_min_timestep": 1400,
        "lr_anneal_timestep": 1000000,
        "update_type": "replace",
        "update_frequency": 10000,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        hid_layers: list with tuple consisting of two elements. (conv_hid, flat_hid)
                    Note: tuple must contain two elements, use empty list if no such layers.
            1. conv_hid: list containing dimensions of the convolutional hidden layers. Asssumed to all come before the flat layers.
                Note: a convolutional layer should specify the in_channel, out_channels, kernel_size, stride (of kernel steps), padding, and dilation (spacing between kernel points) E.g. [3, 16, (5, 5), 1, 0, (2, 2)]
                For more details, see http://pytorch.org/docs/master/nn.html#conv2d and https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

            2. flat_hid: list of dense layers following the convolutional layers
        hid_layers_activation: activation function for the hidden layers
        batch_norm: whether to add batch normalization after each convolutional layer, excluding the input layer.
        clip_grad: whether to clip the gradient
        clip_grad_val: the clip value
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_decay: function to decay learning rate
        lr_decay_frequency: how many total timesteps per decay
        lr_decay_min_timestep: minimum amount of total timesteps before starting decay
        lr_anneal_timestep: timestep to anneal lr decay
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        # OpenAI gym provides images as W x H x C, pyTorch expects C x W x H
        in_dim = np.roll(in_dim, 1)
        # use generic multi-output for Convnet
        out_dim = np.reshape(out_dim, -1).tolist()
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            batch_norm=True,
            clip_grad=False,
            clip_grad_val=1.0,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_decay='no_decay',
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'batch_norm',
            'clip_grad',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_decay',
            'lr_decay_frequency',
            'lr_decay_min_timestep',
            'lr_anneal_timestep',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        self.conv_hid_layers = self.hid_layers[0]
        self.dense_hid_layers = self.hid_layers[1]
        # conv layer
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        # fc layer from flattened conv
        self.dense_model = self.build_dense_layers(self.dense_hid_layers)
        # tails
        tail_in_dim = self.dense_hid_layers[-1] if len(self.dense_hid_layers) > 0 else self.conv_out_dim
        self.model_tails = nn.ModuleList([nn.Linear(tail_in_dim, out_d) for out_d in self.out_dim])

        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_decay = getattr(net_util, self.lr_decay)

    def __str__(self):
        return super(ConvNet, self).__str__() + f'\noptim: {self.optim}'

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
        for i, hid_layer in enumerate(conv_hid_layers):
            conv_layers.append(nn.Conv2d(
                hid_layer[0],  # in chnl
                hid_layer[1],  # out chnl
                tuple(hid_layer[2]),  # kernel
                stride=hid_layer[3],
                padding=hid_layer[4],
                dilation=tuple(hid_layer[5])))
            conv_layers.append(net_util.get_activation_fn(self.hid_layers_activation))
            # Don't include batch norm in the first layer
            if self.batch_norm and i != 0:
                conv_layers.append(nn.BatchNorm2d(hid_layer[1]))
        conv_model = nn.Sequential(*conv_layers)
        return conv_model

    def build_dense_layers(self, dense_hid_layers):
        '''
        Builds all of the dense layers in the network and store in a Sequential model
        '''
        self.conv_out_dim = self.get_conv_output_size()
        dims = [self.conv_out_dim] + dense_hid_layers
        dense_model = net_util.build_sequential(dims, self.hid_layers_activation)
        return dense_model

    def forward(self, x):
        '''The feedforward step'''
        if x.dim() == 3:
            x = x.permute(2, 0, 1).clone()
            x.unsqueeze_(dim=0)
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        x = self.conv_model(x)
        x = x.view(-1, self.conv_out_dim)
        x = self.dense_model(x)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(x))
        # return tensor if single tail, else list of tail tensors
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def training_step(self, x=None, y=None, loss=None, retain_graph=False):
        '''Takes a single training step: one forward and one backwards pass'''
        self.train()
        self.zero_grad()
        self.optim.zero_grad()
        if loss is None:
            out = self(x)
            loss = self.loss_fn(out, y)
        assert not torch.isnan(loss).any()
        if net_util.to_assert_trained():
            assert_trained = net_util.gen_assert_trained(self.conv_model)
        loss.backward(retain_graph=retain_graph)
        if self.clip_grad:
            logger.debug(f'Clipping gradient')
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad_val)
        self.optim.step()
        if net_util.to_assert_trained():
            assert_trained(self.conv_model)
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.eval()
        return self(x)

    def update_lr(self, clock):
        assert 'lr' in self.optim_spec
        old_lr = self.optim_spec['lr']
        new_lr = self.lr_decay(self, clock)
        if new_lr == old_lr:
            return
        self.optim_spec['lr'] = new_lr
        logger.info(f'Learning rate decayed from {old_lr:.6f} to {self.optim_spec["lr"]:.6f}')
        self.optim = net_util.get_optim(self, self.optim_spec)


class DuelingConvNet(ConvNet):
    '''
    Class for generating arbitrary sized convolutional neural network,
    with optional batch normalization, and with dueling heads. Intended for Q-Learning algorithms only.
    Implementation based on "Dueling Network Architectures for Deep Reinforcement Learning" http://proceedings.mlr.press/v48/wangf16.pdf

    Assumes that a single input example is organized into a 3D tensor.
    The entire model consists of three parts:
        1. self.conv_model
        2. self.dense_model
        3. self.model_tails

    e.g. net_spec
    "net": {
        "type": "DuelingConvNet",
        "hid_layers": [
          [
            [4, 32, [8, 8], 4, 0, [1, 1]],
            [32, 64, [4, 4], 2, 0, [1, 1]],
            [64, 64, [3, 3], 1, 0, [1, 1]]
          ],
          [512]
        ],
        "hid_layers_activation": "relu",
        "batch_norm": false,
        "clip_grad": false,
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "SmoothL1Loss"
        },
        "optim_spec": {
          "name": "RMSprop",
          "lr": 0.00025,
          "alpha": 0.95,
          "eps": 0.01,
          "momentum": 0.0,
          "centered": true
        },
        "lr_decay": "no_decay",
        "lr_decay_frequency": 400,
        "lr_decay_min_timestep": 1400,
        "lr_anneal_timestep": 1000000,
        "update_type": "replace",
        "update_frequency": 10000,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        hid_layers: list with tuple consisting of two elements. (conv_hid, flat_hid)
                    Note: tuple must contain two elements, use empty list if no such layers.
            1. conv_hid: list containing dimensions of the convolutional hidden layers. Asssumed to all come before the flat layers.
                Note: a convolutional layer should specify the in_channel, out_channels, kernel_size, stride (of kernel steps), padding, and dilation (spacing between kernel points) E.g. [3, 16, (5, 5), 1, 0, (2, 2)]
                For more details, see http://pytorch.org/docs/master/nn.html#conv2d and https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

            2. flat_hid: list of dense layers following the convolutional layers
        hid_layers_activation: activation function for the hidden layers
        batch_norm: whether to add batch normalization after each convolutional layer, excluding the input layer.
        clip_grad: whether to clip the gradient
        clip_grad_val: the clip value
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_decay: function to decay learning rate
        lr_decay_frequency: how many total timesteps per decay
        lr_decay_min_timestep: minimum amount of total timesteps before starting decay
        lr_anneal_timestep: timestep to anneal lr decay
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        # OpenAI gym provides images as W x H x C, pyTorch expects C x W x H
        in_dim = np.roll(in_dim, 1)
        # use generic multi-output for Convnet
        out_dim = np.reshape(out_dim, -1).tolist()
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            batch_norm=True,
            clip_grad=False,
            clip_grad_val=1.0,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_decay='no_decay',
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'batch_norm',
            'clip_grad',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_decay',
            'lr_decay_frequency',
            'lr_decay_min_timestep',
            'lr_anneal_timestep',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        # Guard against inappropriate algorithms and environments
        assert len(out_dim) == 1
        # Build model
        self.conv_hid_layers = self.hid_layers[0]
        self.dense_hid_layers = self.hid_layers[1]
        # conv layer
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        # fc layer from flattened conv
        self.dense_model = self.build_dense_layers(self.dense_hid_layers)
        # tails
        tail_in_dim = self.dense_hid_layers[-1] if len(self.dense_hid_layers) > 0 else self.conv_out_dim
        # output layers
        self.v = nn.Linear(tail_in_dim, 1)  # state value
        self.adv = nn.Linear(tail_in_dim, out_dim[0])  # action dependent raw advantage

        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_decay = getattr(net_util, self.lr_decay)

    def forward(self, x):
        '''The feedforward step'''
        if x.dim() == 3:
            x = x.permute(2, 0, 1).clone()
            x.unsqueeze_(dim=0)
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        x = self.conv_model(x)
        x = x.view(-1, self.conv_out_dim)
        x = self.dense_model(x)
        state_value = self.v(x)
        raw_advantages = self.adv(x)
        out = net_util.calc_q_value_logits(state_value, raw_advantages)
        return out
