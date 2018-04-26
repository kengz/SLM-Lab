from slm_lab.agent.net import net_util
from slm_lab.lib import logger
from torch.autograd import Variable
from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logger.get_logger(__name__)


class ConvNet(nn.Module):
    '''
    Class for generating arbitrary sized convolutional neural network,
    with optional batch normalization

    Assumes that a single input example is organized into a 3D tensor.
    The entire model consists of three parts:
         1. self.conv_model
         2. self.dense_model
         3. self.out_layers
    '''

    def __init__(self,
                 in_dim,
                 hid_layers,
                 out_dim,
                 hid_layers_activation=None,
                 optim_param=None,
                 loss_param=None,
                 clamp_grad=False,
                 clamp_grad_val=1.0,
                 batch_norm=True,
                 gpu=False,
                 decay_lr=0.9):
        '''
        in_dim: dimension of the inputs
        hid_layers: tuple consisting of two elements. (conv_hid, flat_hid)
                    Note: tuple must contain two elements, use empty list if no such layers.
            1. conv_hid: list containing dimensions of the convolutional hidden layers. Asssumed to all come before the flat layers.
                Note: a convolutional layer should specify the in_channel, out_channels, kernel_size, stride (of kernel steps), padding, and dilation (spacing between kernel points) E.g. [3, 16, (5, 5), 1, 0, (2, 2)]
                For more details, see http://pytorch.org/docs/master/nn.html#conv2d and https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

            2. flat_hid: list of dense layers following the convolutional layers
        out_dim: dimension of the output for one output, otherwise a list containing the dimensions of the ouputs for a multi-headed network
        hid_layers_activation: activation function for the hidden layers
        optim_param: parameters for initializing the optimizer
        loss_param: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient
        batch_norm: whether to add batch normalization after each convolutional layer, excluding the input layer.
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        @example:
        net = ConvNet(
                (3, 32, 32),
                ([[3, 36, (5, 5), 1, 0, (2, 2)],[36, 128, (5, 5), 1, 0, (2, 2)]],[100]),
                10,
                hid_layers_activation='relu',
                optim_param={'name': 'Adam'},
                loss_param={'name': 'mse_loss'},
                clamp_grad=False,
                batch_norm=True,
                gpu=True,
                decay_lr=0.9)
        '''
        super(ConvNet, self).__init__()
        # Create net and initialize params
        # We need to transpose the dimensions for pytorch.
        # OpenAI gym provides images as W x H x C, pyTorch expects C x W x H
        self.in_dim = list(in_dim[:-1])
        self.in_dim.insert(0, in_dim[-1])
        # Handle multiple types of out_dim (single and multi-headed)
        if type(out_dim) is int:
            out_dim = [out_dim]
        self.out_dim = out_dim
        self.batch_norm = batch_norm
        self.conv_layers = []
        self.conv_model = self.build_conv_layers(
            hid_layers[0], hid_layers_activation)
        self.flat_layers = []
        self.dense_model = self.build_flat_layers(
            hid_layers[1], hid_layers_activation)
        self.out_layers = []
        in_D = hid_layers[1][-1] if len(hid_layers[1]) > 0 else self.flat_dim
        for dim in out_dim:
            self.out_layers += [nn.Linear(in_D, dim)]
        self.num_hid_layers = len(self.conv_layers) + len(self.flat_layers)
        self.init_params()
        if torch.cuda.is_available() and gpu:
            self.conv_model.cuda()
            self.dense_model.cuda()
            for l in self.out_layers:
                l.cuda()
        # Init other net variables
        self.params = list(self.conv_model.parameters()) + \
            list(self.dense_model.parameters())
        for layer in self.out_layers:
            self.params.extend(list(layer.parameters()))
        self.optim_param = optim_param
        self.optim = net_util.get_optim_multinet(self.params, self.optim_param)
        self.loss_fn = net_util.get_loss_fn(self, loss_param)
        self.clamp_grad = clamp_grad
        self.clamp_grad_val = clamp_grad_val
        self.decay_lr = decay_lr
        logger.info(f'loss fn: {self.loss_fn}')
        logger.info(f'optimizer: {self.optim}')
        logger.info(f'decay lr: {self.decay_lr}')

    def get_conv_output_size(self):
        '''Helper function to calculate the size of the
           flattened features after the final convolutional layer'''
        x = Variable(torch.ones(1, *self.in_dim))
        x = self.conv_model(x)
        return x.numel()

    def build_conv_layers(self, conv_hid, hid_layers_activation):
        '''Builds all of the convolutional layers in the network.
           These layers are turned into a Sequential model and stored
           in self.conv_model.
           The entire model consists of two parts:
                1. self.conv_model
                2. self.dense_model
                3. self.out_layers'''
        for i, layer in enumerate(conv_hid):
            self.conv_layers += [nn.Conv2d(
                conv_hid[i][0],
                conv_hid[i][1],
                tuple(conv_hid[i][2]),
                stride=conv_hid[i][3],
                padding=conv_hid[i][4],
                dilation=tuple(conv_hid[i][5]))]
            self.conv_layers += [
                net_util.get_activation_fn(hid_layers_activation)]
            # Don't include batch norm in the first layer
            if self.batch_norm and i != 0:
                self.conv_layers += [nn.BatchNorm2d(conv_hid[i][1])]
        return nn.Sequential(*self.conv_layers)

    def build_flat_layers(self, flat_hid, hid_layers_activation):
        '''Builds all of the dense layers in the network.
           These layers are turned into a Sequential model and stored
           in self.dense_model.
           The entire model consists of two parts:
                1. self.conv_model
                2. self.dense_model
                3. self.out_layers'''
        self.flat_dim = self.get_conv_output_size()
        for i, layer in enumerate(flat_hid):
            in_D = self.flat_dim if i == 0 else flat_hid[i - 1]
            out_D = flat_hid[i]
            self.flat_layers += [nn.Linear(in_D, out_D)]
            self.flat_layers += [
                net_util.get_activation_fn(hid_layers_activation)]
        return nn.Sequential(*self.flat_layers)

    def forward(self, x):
        '''The feedforward step'''
        if x.dim() == 3:
            x = x.permute(2, 0, 1).clone()
            x.unsqueeze_(dim=0)
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
            logger.debug(f'x: {x.size()}')
        x = self.conv_model(x)
        x = x.view(-1, self.flat_dim)
        x = self.dense_model(x)
        '''If only one head, return tensor, otherwise return list of outputs'''
        outs = []
        for layer in self.out_layers:
            out = layer(x)
            outs.append(out)
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def set_train_eval(self, train=True):
        '''Helper function to set model in training or evaluation mode'''
        nets = [self.conv_model] + [self.dense_model] + self.out_layers
        for net in nets:
            if train:
                net.train()
            else:
                net.eval()

    def training_step(self, x, y):
        '''
        Takes a single training step: one forward and one backwards pass
        '''
        self.set_train_eval()
        self.optim.zero_grad()
        out = self(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        if self.clamp_grad:
            logger.debug(f'Clipping gradient...')
            torch.nn.utils.clip_grad_norm(
                self.conv_model.parameters(), self.clamp_grad_val)
            torch.nn.utils.clip_grad_norm(
                self.dense_model.parameters(), self.clamp_grad_val)
        self.optim.step()
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.set_train_eval(train=False)
        outs = self(x)
        if type(outs) is list:
            outs = [o.data for o in outs]
        else:
            outs = outs.data
        return outs

    def init_params(self):
        '''
        Initializes all of the model's parameters using xavier uniform initialization.
        Biases are all set to 0.01
        '''
        layers = self.conv_layers + self.flat_layers + self.out_layers
        net_util.init_layers(layers, 'Linear')
        net_util.init_layers(layers, 'Conv')
        net_util.init_layers(layers, 'BatchNorm')

    def gather_trainable_params(self):
        '''
        Gathers parameters that should be trained into a list returns: copy of a list of fixed params
        '''
        return [param.clone() for param in self.params]

    def gather_fixed_params(self):
        '''
        Gathers parameters that should be fixed into a list returns: copy of a list of fixed params
        '''
        return None

    def get_grad_norms(self):
        '''Returns a list of the norm of the gradients for all parameters'''
        norms = []
        for i, param in enumerate(self.params):
            grad_norm = torch.norm(param.grad.data)
            if grad_norm is None:
                logger.info(f'Param with None grad: {param}, layer: {i}')
            norms.append(grad_norm)
        return norms

    def __str__(self):
        '''Overriding so that print() will print the whole network'''
        s = self.conv_model.__str__() + '\n' + self.dense_model.__str__()
        for layer in self.out_layers:
            s += '\n' + layer.__str__()
        return s

    def update_lr(self):
        assert 'lr' in self.optim_param
        old_lr = self.optim_param['lr']
        self.optim_param['lr'] = old_lr * self.decay_lr
        logger.debug(f'Learning rate decayed from {old_lr} to {self.optim_param["lr"]}')
        self.optim = net_util.get_optim_multinet(self.params, self.optim_param)
