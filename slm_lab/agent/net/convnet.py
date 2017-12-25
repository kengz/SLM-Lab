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
                 clamp_grad_val=1.0,
                 batch_norm=True):
        '''
        in_dim: dimension of the inputs
        conv_hid: list containing dimensions of the convolutional hidden layers. Asssumed to all come before the flat layers.
            Note: a convolutional layer should specify the in_channel, out_channels, kernel_size, stride (of kernel steps), padding, and dilation (spacing between kernel points) E.g. [3, 16, (5, 5), 1, 0, (2, 2)]
            For more details, see http://pytorch.org/docs/master/nn.html#conv2d and https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

        flat_hid: list of dense layers following the convolutional layers
        out_dim: dimension of the ouputs
        optim_param: parameters for initializing the optimizer
        loss_param: measure of error between model
        predictions and correct outputs
        hid_layers_activation: activation function for the hidden layers
        out_activation_param: activation function for the last layer
        clamp_grad: whether to clamp the gradient
        batch_norm: whether to add batch normalization after each convolutional layer, excluding the input layer.
        @example:
        net = ConvNet(
                (3, 32, 32),
                [[3, 36, (5, 5), 1, 0, (2, 2)],
                [36, 128, (5, 5), 1, 0, (2, 2)]],
                [100],
                10,
                hid_layers_activation='relu',
                optim_param={'name': 'Adam'},
                loss_param={'name': 'mse_loss'},
                clamp_grad=False,
                batch_norm=True)
        '''
        Module.__init__(self)
        # Create net and initialize params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_norm = batch_norm
        self.conv_layers = []
        self.conv_model = self.build_conv_layers(conv_hid, hid_layers_activation)
        self.flat_layers = []
        self.dense_model = self.build_flat_layers(flat_hid, out_dim, hid_layers_activation)
        self.num_hid_layers = len(self.conv_layers) + len(self.flat_layers) - 1
        self.init_params()
        # Init other net variables
        self.optim = net_util.get_optim(self, optim_param)
        self.loss_fn = net_util.get_loss_fn(self, loss_param)
        self.clamp_grad = clamp_grad
        self.clamp_grad_val = clamp_grad_val

    def get_conv_output_size(self):
        x = Variable(torch.ones(1, *self.in_dim))
        x = self.conv_model(x)
        return x.numel()

    def build_conv_layers(self, conv_hid, hid_layers_activation):
        for i, layer in enumerate(conv_hid):
            self.conv_layers += [nn.Conv2d(
                conv_hid[i][0],
                conv_hid[i][1],
                conv_hid[i][2],
                stride=conv_hid[i][3],
                padding=conv_hid[i][4],
                dilation=conv_hid[i][5])]
            self.conv_layers += [net_util.get_activation_fn(hid_layers_activation)]
            # Don't include batch norm in the first layer
            if self.batch_norm and i != 0:
                self.conv_layers += [nn.BatchNorm2d(conv_hid[i][1])]
        return nn.Sequential(*self.conv_layers)

    def build_flat_layers(self, flat_hid, out_dim, hid_layers_activation):
        self.flat_dim = self.get_conv_output_size()
        for i, layer in enumerate(flat_hid):
            in_D = self.flat_dim if i == 0 else flat_hid[i - 1]
            out_D = flat_hid[i]
            self.flat_layers += [nn.Linear(in_D, out_D)]
            self.flat_layers += [net_util.get_activation_fn(hid_layers_activation)]
        in_D = flat_hid[-1] if len(flat_hid) > 0 else self.flat_dim
        self.flat_layers += [nn.Linear(in_D, out_dim)]
        return nn.Sequential(*self.flat_layers)

    def forward(self, x):
        '''The feedforward step'''
        x = self.conv_model(x)
        x = x.view(-1, self.flat_dim)
        x = self.dense_model(x)
        return x

    def training_step(self, x, y):
        '''
        Takes a single training step: one forwards and one backwards pass
        '''
        return super(ConvNet, self).training_step(x, y)

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model
        returns: network output given input x
        '''
        return super(ConvNet, self).wrap_eval(x)

    def init_params(self):
        '''
        Initializes all of the model's parameters using xavier uniform initialization.
        Biases are all set to 0.01
        '''
        biasinit = 0.01
        layers = self.conv_layers + self.flat_layers
        for layer in layers:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1 or classname.find('Conv') != -1:
                torch.nn.init.xavier_uniform(layer.weight.data)
                layer.bias.data.fill_(biasinit)

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
