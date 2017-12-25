from slm_lab.agent.net import net_util
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network
    '''

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 hid_layers_activation=None,
                 optim_param=None,
                 loss_param=None,
                 clamp_grad=False,
                 clamp_grad_val=1.0):
        '''
        in_dim: dimension of the inputs
        hid_dim: list containing dimensions of the hidden layers
        out_dim: dimension of the ouputs
        optim_param: parameters for initializing the optimizer
        hid_layers_activation: activation function for the hidden layers
        loss_param: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient
        @example:
        net = MLPNet(1000, [512, 256, 128], 10, 'relu', optim_param={'name': 'Adam'}, loss_param={'name': 'mse_loss'}, True, 2.0)
        '''
        super(MLPNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        for i, layer in enumerate(hid_dim):
            in_D = in_dim if i == 0 else hid_dim[i - 1]
            out_D = hid_dim[i]
            self.layers += [nn.Linear(in_D, out_D)]
            self.layers += [net_util.get_activation_fn(hid_layers_activation)]
        in_D = hid_dim[-1] if len(hid_dim) > 0 else in_dim
        self.layers += [nn.Linear(in_D, out_dim)]
        self.model = nn.Sequential(*self.layers)
        self.init_params()
        self.optim = net_util.get_optim(self, optim_param)
        self.loss_fn = net_util.get_loss_fn(self, loss_param)
        self.clamp_grad = clamp_grad
        self.clamp_grad_val = clamp_grad_val

    def forward(self, x):
        '''The feedforward step'''
        return self.model(x)

    def training_step(self, x, y):
        '''
        Takes a single training step: one forward and one backwards pass
        '''
        self.model.train()
        self.model.zero_grad()
        self.optim.zero_grad()
        out = self(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        if self.clamp_grad:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clamp_grad_val)
        self.optim.step()
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.eval()
        return self(x).data

    def init_params(self):
        '''
        Initializes all of the model's parameters using xavier uniform initialization.
        Note: There appears to be unreproduceable behaviours in pyTorch for xavier init
        Sometimes the trainable params tests pass (see nn_test.py), other times they dont.
        Biases are all set to 0.01
        '''
        biasinit = 0.01
        for layer in self.layers:
            classname = layer.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.xavier_uniform(layer.weight.data)
                layer.bias.data.fill_(biasinit)

    def gather_trainable_params(self):
        '''
        Gathers parameters that should be trained into a list returns: copy of a list of fixed params
        '''
        return [param.clone() for param in self.parameters()]

    def gather_fixed_params(self):
        '''
        Gathers parameters that should be fixed into a list returns: copy of a list of fixed params
        '''
        return None
