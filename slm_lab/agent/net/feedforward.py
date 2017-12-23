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
                 clamp_grad_val=1):
        '''
        in_dim: dimension of the inputs
        hid_dim: list containing dimensions of the hidden layers
        out_dim: dimension of the ouputs
        optim: optimizer
        loss_param: measure of error between model predictions and correct outputs
        hid_layers_activation: activation function for the hidden layers
        out_activation_param: activation function for the last layer
        clamp_grad: whether to clamp the gradient to + / - 1
        @example:
        net = MLPNet(1000, [512, 256, 128], 10, optim_param={'name': 'Adam'}, loss_param={'name': 'mse_loss'})
        '''
        # TODO see openai baselines, model.py,  _mlp is so clean
        super(MLPNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_layers = []
        # TODO parametrize the activation function choice
        for i, layer in enumerate(hid_dim):
            in_D = in_dim if i == 0 else hid_dim[i - 1]
            out_D = hid_dim[i]
            lin = nn.Linear(in_D, out_D)
            setattr(self, 'linear_' + str(i), lin)
            self.hid_layers.append(lin)
        # TODO parametrize output layer activation too
        in_D = hid_dim[-1] if len(hid_dim) > 0 else in_dim
        self.out_layer = nn.Linear(in_D, out_dim)
        self.num_hid_layers = len(self.hid_layers)

        # TODO cant we do like tf?: layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
        self.hid_layers_activation_fn = net_util.set_activation_fn(
            self, hid_layers_activation)
        self.optim = net_util.set_optim(self, optim_param)
        self.loss_fn = net_util.set_loss_fn(self, loss_param)
        print(self.hid_layers_activation_fn, self.optim, self.loss_fn)
        self.clamp_grad = clamp_grad
        self.clamp_grad_val = clamp_grad_val
        self.init_params()

    def forward(self, x):
        '''The feedforward step'''
        # TODO parametrize the activation function choice
        for i in range(self.num_hid_layers):
            x = self.hid_layers_activation_fn((self.hid_layers[i](x)))
        x = self.out_layer(x)
        return x

    def training_step(self, x, y):
        '''
        Takes a single training step: one forwards and one backwards pass
        '''
        # Sets model in training mode and zero the gradients
        self.train()
        self.zero_grad()
        self.optim.zero_grad()
        out = self(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        if self.clamp_grad:
            print("Clipping gradient")
            torch.nn.utils.clip_grad_norm(self.parameters(), self.clamp_grad_val)
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
        Initializes all of the model's parameters using uniform initialization.
        Note: Ideally it should be xavier initialization, but there appears to be unreproduceable behaviours in pyTorch.
        Sometimes the trainable params tests pass (see nn_test.py), other times they dont.
        Biases are all set to 0.01
        '''
        initrange = 0.1
        biasinit = 0.01
        lin_layers = self.hid_layers + list([self.out_layer])
        for layer in lin_layers:
            # layer.weight.data.uniform_(-initrange, initrange)
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

    def print_grad_norms(self):
        lin_layers = self.hid_layers + list([self.out_layer])
        for i, layer in enumerate(lin_layers):
            print("Gradient norm layer {}: {}".format(
                i, torch.norm(layer.weight.grad)))
