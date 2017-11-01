import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class MLPNet(nn.Module):
    '''
    Class for generating arbitrary sized feedforward
    neural network, with ReLU activations and batch normalization
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
                in_D = hid_dim[i-1]
            out_D = hid_dim[i]
            l = nn.Linear(in_D, out_D)
            setattr(self, 'linear_' + str(i), l)
            self.hid_layers.append(l)
            b = nn.BatchNorm1d(out_D)
            setattr(self, 'bn_' + str(i), b)
            self.batch_norms.append(b)
        assert len(self.batch_norms) == len(self.hid_layers)
        self.out_layer = nn.Linear(hid_dim[-1], out_dim)
        self.num_hid_layers = len(self.hid_layers)
        self.optim = optim(self.parameters())
        self.loss_fn = loss_fn
        self.clamp_grad = clamp_grad
        self.init_params()

    def forward(self, x):
        '''
        The feedforward step
        '''
        print(x)
        for i in range(self.num_hid_layers):
            x = F.relu(self.batch_norms[i](self.hid_layers[i](x)))
            print(x)
        x = self.out_layer(x)
        return x

    def training_step(self, x, y):
        '''
        Takes a single training step: one forwards and one backwards pass
        '''

        '''
        Set model in training mode and zero the gradients
        Should be set to train() when training and eval() during inference
        '''
        self.train()
        self.zero_grad()
        self.optim.zero_grad()

        out = self(x)
        print(out)
        loss = self.loss_fn(out, y)
        print(loss)
        loss.backward()
        if self.clamp_grad:
            for param in self.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optim.step()
        return loss

    def init_params(self):
        '''
        Initializes all of the model's parameters using uniform initialization.
        Biases are all set to 0
        '''
        # TODO: change to Xavier init
        initrange = 0.2
        lin_layers = self.hid_layers + list([self.out_layer])
        for layer in lin_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(0)
        for layer in self.batch_norms:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(0)
