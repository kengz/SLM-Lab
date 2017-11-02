import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class MLPNet(nn.Module):
    '''
    Class for generating arbitrary sized feedforward
    neural network, with ReLU activations
    '''

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 optim=optim.Adam,
                 loss_fn=F.smooth_l1_loss,
                 clamp_grad=False):
        '''
        in_dim: dimension of the inputs
        hid_dim: list containing dimensions of the hidden layers
        out_dim: dimension of the ouputs
        optim: optimizer
        loss_fn: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient to + / - 1
        example:
        net = MLPNet(1000, [512, 256, 128], 10, optim.Adam, nn.SmoothL1Loss)
        '''
        super(MLPNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_layers = []
        self.batch_norms = []
        for i, layer in enumerate(hid_dim):
            if i == 0:
                in_D = in_dim
            else:
                in_D = hid_dim[i - 1]
            out_D = hid_dim[i]
            l = nn.Linear(in_D, out_D)
            setattr(self, 'linear_' + str(i), l)
            self.hid_layers.append(l)
        if len(hid_dim) > 0:
            self.out_layer = nn.Linear(hid_dim[-1], out_dim)
        else:
            self.out_layer = nn.Linear(in_dim, out_dim)
        self.num_hid_layers = len(self.hid_layers)
        self.optim = optim(self.parameters())
        self.loss_fn = loss_fn
        self.clamp_grad = clamp_grad
        self.init_params()

    def forward(self, x):
        ''' The feedforward step '''
        for i in range(self.num_hid_layers):
            x = F.relu((self.hid_layers[i](x)))
        x = self.out_layer(x)
        return x

    def training_step(self, x, y):
        '''
        Takes a single training step: one forwards and one backwards pass
        '''
        # Sets model in training mode and zero the gradients
        self.train()
        self.optim.zero_grad()
        out = self(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        if self.clamp_grad:
            for param in self.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optim.step()
        return loss

    def eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model
        returns: network output given input x
        '''
        self.eval()
        return self(x)

    def init_params(self):
        '''
        Initializes all of the model's parameters using uniform initialization.
        Note: Ideally it should be xavier initialization, but there appears
        to be unreproduceable behaviours in pyTorch.
        Sometimes the trainable params tests pass (see nn_test.py), other times
        they dont.
        Biases are all set to 0.01
        '''
        initrange = 0.2
        biasinit = 0.01
        lin_layers = self.hid_layers + list([self.out_layer])
        for layer in lin_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(biasinit)
