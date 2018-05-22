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
         3. self.out_layers
    '''

    def __init__(self, net_spec, algorithm, in_dim, out_dim):
        '''
        net_spec:
        hid_layers: list with tuple consisting of two elements. (conv_hid, flat_hid)
                    Note: tuple must contain two elements, use empty list if no such layers.
            1. conv_hid: list containing dimensions of the convolutional hidden layers. Asssumed to all come before the flat layers.
                Note: a convolutional layer should specify the in_channel, out_channels, kernel_size, stride (of kernel steps), padding, and dilation (spacing between kernel points) E.g. [3, 16, (5, 5), 1, 0, (2, 2)]
                For more details, see http://pytorch.org/docs/master/nn.html#conv2d and https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

            2. flat_hid: list of dense layers following the convolutional layers
        hid_layers_activation: activation function for the hidden layers
        optim_spec: parameters for initializing the optimizer
        loss_spec: measure of error between model predictions and correct outputs
        batch_norm: whether to add batch normalization after each convolutional layer, excluding the input layer.
        clip_grad: whether to clip the gradient
        clip_grad_val: the clip value
        decay_lr: whether to decay learning rate
        decay_lr_factor: the multiplicative decay factor
        decay_lr_frequency: how many total timesteps per decay
        decay_lr_min_timestep: minimum amount of total timesteps before starting decay
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_weight: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        # OpenAI gym provides images as W x H x C, pyTorch expects C x W x H
        in_dim = np.roll(in_dim, 1)
        # use generic multi-output for Convnet
        out_dim = np.reshape(out_dim, -1).tolist()
        nn.Module.__init__(self)
        super(ConvNet, self).__init__(net_spec, algorithm, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'MSELoss'},
            batch_norm=True,
            clip_grad=False,
            clip_grad_val=1.0,
            decay_lr_factor=0.9,
            update_type='replace',
            update_frequency=1,
            polyak_weight=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'optim_spec',
            'loss_spec',
            'batch_norm',
            'clip_grad',
            'clip_grad_val',
            'decay_lr',
            'decay_lr_factor',
            'decay_lr_frequency',
            'decay_lr_min_timestep',
            'update_type',
            'update_frequency',
            'polyak_weight',
            'gpu',
        ])

        self.conv_hid_layers = self.hid_layers[0]
        self.dense_hid_layers = self.hid_layers[1]
        # conv layer
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        # fc layer from flattened conv
        self.dense_model = self.build_dense_layers(self.dense_hid_layers)
        # tails
        tail_in_dim = self.dense_hid_layers[-1] if len(self.dense_hid_layers) > 1 else self.conv_out_dim
        self.model_tails = nn.ModuleList([nn.Linear(tail_in_dim, out_d) for out_d in self.out_dim])

        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)

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
            logger.debug(f'x: {x.size()}')
        x = self.conv_model(x)
        x = x.view(-1, self.conv_out_dim)
        x = self.dense_model(x)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(x))
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def training_step(self, x=None, y=None, loss=None):
        '''Takes a single training step: one forward and one backwards pass'''
        self.train()
        self.zero_grad()
        self.optim.zero_grad()
        if loss is None:
            out = self(x)
            loss = self.loss_fn(out, y)
        loss.backward()
        if self.clip_grad:
            logger.debug(f'Clipping gradient')
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clip_grad_val)
        self.optim.step()
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.eval()
        return self(x)

    def update_lr(self):
        assert 'lr' in self.optim_spec
        old_lr = self.optim_spec['lr']
        self.optim_spec['lr'] = old_lr * self.decay_lr_factor
        logger.info(f'Learning rate decayed from {old_lr:.6f} to {self.optim_spec["lr"]:.6f}')
        self.optim = net_util.get_optim(self, self.optim_spec)
