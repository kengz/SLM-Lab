from slm_lab.agent.net import net_util
from slm_lab.agent.net.feedforward import MLPNet
from torch.autograd import Variable
from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(MLPNet):
    '''
    Class for generating arbitrary sized convolutional neural network,
    with ReLU activations, and optional batch normalization

    Assumed that a single input example is organized into a 3D tensor
    '''

    def __init__(self,
                 in_dim,
                 conv_hid,
                 flat_hid,
                 out_dim,
                 hid_layers_activation=None,
                 optim_param=None,
                 loss_param=None,
                 clamp_grad=False,
                 batch_norm=True):
        '''
        in_dim: dimension of the inputs
        conv_hid: list containing dimensions of the convolutional hidden layers. Asssumed to all come before the flat layers.
            Note: a convolutional layer should specify the in_channel, out_channels, kernel_size, stride (of kernel steps), padding, and dilation (spacing between kernel points) E.g. [3, 16, (5, 5), 1, 0, (2, 2)]
            For more details, see http://pytorch.org/docs/master/nn.html#conv2d and https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

        flat_hid: list of dense layers following the convolutional layers
        out_dim: dimension of the ouputs
        optim: optimizer
        loss_param: measure of error between model
        predictions and correct outputs
        hid_layers_activation: activation function for the hidden layers
        out_activation_param: activation function for the last layer
        clamp_grad: whether to clamp the gradient to + / - 1
        batch_norm: whether to add batch normalization after each convolutional layer, excluding the input layer.
        @example:
        net = ConvNet(
                (3, 32, 32),
                [[3, 36, (5, 5), 1, 0, (2, 2)],
                [36, 128, (5, 5), 1, 0, (2, 2)]],
                [100],
                10,
                optim_param={'name': 'Adam'},
                loss_param={'name': 'mse_loss'})
        '''
        # Calling super on greatgrandfather class
        Module.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_norm = batch_norm
        self.conv_layers = []
        self.batch_norms = []
        self.flat_layers = []
        self.build_conv_layers(conv_hid)
        self.build_flat_layers(flat_hid, out_dim)
        self.num_hid_layers = len(self.conv_layers) + len(self.flat_layers)

        self.hid_layers_activation_fn = net_util.get_activation_fn(
            self, hid_layers_activation)
        self.optim = net_util.get_optim(self, optim_param)
        self.loss_fn = net_util.get_loss_fn(self, loss_param)
        self.clamp_grad = clamp_grad
        self.init_params()

    def get_conv_output_size(self):
        x = Variable(torch.ones(1, *self.in_dim))
        bn_flag = len(self.batch_norms) > 0
        for i, layer in enumerate(self.conv_layers):
            # Don't need to pass through batch norms to calculate size
            x = layer(x)
        return x.numel()

    def build_conv_layers(self, conv_hid):
        for i, layer in enumerate(conv_hid):
            lin = nn.Conv2d(
                conv_hid[i][0],
                conv_hid[i][1],
                conv_hid[i][2],
                stride=conv_hid[i][3],
                padding=conv_hid[i][4],
                dilation=conv_hid[i][5])
            setattr(self, 'conv_' + str(i), lin)
            self.conv_layers.append(lin)
            # Don't include batch norm in the first layer
            if self.batch_norm and i != 0:
                b = nn.BatchNorm2d(conv_hid[i][1])
                setattr(self, 'bn_' + str(i), b)
                self.batch_norms.append(b)

    def build_flat_layers(self, flat_hid, out_dim):
        self.flat_dim = self.get_conv_output_size()
        for i, layer in enumerate(flat_hid):
            in_D = self.flat_dim if i == 0 else flat_hid[i - 1]
            out_D = flat_hid[i]
            lin = nn.Linear(in_D, out_D)
            setattr(self, 'linear_' + str(i), lin)
            self.flat_layers.append(lin)
        in_D = flat_hid[-1] if len(flat_hid) > 0 else self.flat_dim
        self.out_layer = nn.Linear(in_D, out_dim)

    def forward(self, x):
        '''The feedforward step'''
        bn_flag = len(self.batch_norms) > 0
        for i, layer in enumerate(self.conv_layers):
            if bn_flag and i != 0:
                bn = self.batch_norms[i - 1]
                x = self.hid_layers_activation_fn(bn(layer(x)))
            else:
                x = self.hid_layers_activation_fn(layer(x))
        x = x.view(-1, self.flat_dim)
        for layer in self.flat_layers:
            x = self.hid_layers_activation_fn(layer(x))
        x = self.out_layer(x)
        return x

    def training_step(self, x, y):
        '''
        Takes a single training step: one forwards and one backwards pass
        '''
        return super(ConvNet, self).training_step(x, y)

    def eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model
        returns: network output given input x
        '''
        return super(ConvNet, self).eval(x)

    def init_params(self):
        '''
        Initializes all of the model's parameters using uniform initialization.
        Note: Ideally it should be xavier initialization, but there appears to be unreproduceable behaviours in pyTorch.
        Sometimes the trainable params tests pass (see nn_test.py), other times they dont.
        Biases are all set to 0.01
        '''
        initrange = 0.1
        biasinit = 0.01
        bninitrange = 1.0
        bnbiasinit = 0.1
        layers = self.conv_layers + self.flat_layers \
            + list([self.out_layer])
        for layer in layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(biasinit)
        for bn in self.batch_norms:
            bn.weight.data.uniform_(-bninitrange, bninitrange)
            bn.bias.data.fill_(bnbiasinit)

    def gather_trainable_params(self):
        '''
        Gathers parameters that should be trained into a list returns: copy of a list of fixed params
        '''
        return super(ConvNet, self).gather_trainable_params()

    def gather_fixed_params(self):
        '''
        Gathers parameters that should be fixed into a list returns: copy of a list of fixed params
        '''
        return super(ConvNet, self).gather_fixed_params()
