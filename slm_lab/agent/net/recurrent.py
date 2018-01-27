from slm_lab.agent.net import net_util
from torch.autograd import Variable
from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentNet(nn.Module):
    '''
    Class for generating arbitrary sized recurrent neural networks which take a sequence of
    states as input.

    Assumes that a single input example is organized into a 3D tensor
    batch_size x sequence_length x state_dim
    The entire model consists of three parts:
         1. self.state_proc_model
         2. self.rnn
         3. self.fc_out
    '''

    def __init__(self,
                 in_dim,
                 sequence_length,
                 state_processing_layers,
                 hid_dim,
                 out_dim,
                 hid_layers_activation=None,
                 optim_param=None,
                 loss_param=None,
                 clamp_grad=False,
                 clamp_grad_val=1.0,
                 num_rnn_layers=1):
        '''
        in_dim: dimension of the states
        sequence_length: length of the history of being passed to the net
        state_processing_layers: dimensions of the layers for state processing. Each state in the sequence is passed through these layers before being passed to recurrent net as an input.
        hid_dim: dimension of the recurrent component hidden state
        out_dim: dimension of the ouputs
        optim_param: parameters for initializing the optimizer
        loss_param: measure of error between model
        predictions and correct outputs
        hid_layers_activation: activation function for the hidden layers
        out_activation_param: activation function for the last layer
        clamp_grad: whether to clamp the gradient
        clamp_grad_val: what value to clamp the gradient at
        num_rnn_layers: number of recurrent layers
        @example:
        net = RecurrentNet(
                4,
                8,
                [64],
                50,
                10,
                hid_layers_activation='relu',
                optim_param={'name': 'Adam'},
                loss_param={'name': 'mse_loss'},
                clamp_grad=False)
        '''
        super(RecurrentNet, self).__init__()
        # Create net and initialize params
        self.in_dim = in_dim
        self.sequence_length = sequence_length
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_rnn_layers = num_rnn_layers
        self.state_processing_layers = []
        self.state_proc_model = self.build_state_proc_layers(
            state_processing_layers, hid_layers_activation)
        self.rnn_input_dim = state_processing_layers[-1] if len(state_processing_layers) > 0 else self.in_dim
        self.rnn = nn.GRU(input_size=self.rnn_input_dim,
                          hidden_size=self.hid_dim,
                          num_layers=self.num_rnn_layers,
                          batch_first=True)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.num_hid_layers = None
        self.init_params()
        # Init other net variables
        self.params = list(self.state_proc_model.parameters()) + \
            list(self.rnn.parameters()) + list(self.fc_out.parameters())
        self.optim = net_util.get_optim_multinet(self.params, optim_param)
        self.loss_fn = net_util.get_loss_fn(self, loss_param)
        self.clamp_grad = clamp_grad
        self.clamp_grad_val = clamp_grad_val

    def build_state_proc_layers(self, state_processing_layers, hid_layers_activation):
        '''Builds all of the state processing layers in the network.
           These layers are turned into a Sequential model and stored
           in self.state_proc_model.
           The entire model consists of three parts:
                1. self.state_proc_model
                2. self.rnn
                3. self.fc_out'''
        for i, layer in enumerate(state_processing_layers):
            in_D = self.in_dim if i == 0 else state_processing_layers[i - 1]
            out_D = state_processing_layers[i]
            self.state_processing_layers += [nn.Linear(in_D, out_D)]
            self.state_processing_layers += [net_util.get_activation_fn(hid_layers_activation)]
        return nn.Sequential(*self.state_processing_layers)

    def init_hidden(self, batch_size, volatile=False):
        return Variable(torch.zeros(self.num_rnn_layers, batch_size, self.hid_dim), volatile=volatile)

    def forward(self, x):
        '''The feedforward step.
        Input is batch_size x sequence_length x state_dim'''
        '''Unstack input to (batch_size x sequence_length) x state_dim in order to transform all state inputs'''
        batch_size = x.size(0)
        x = x.view(-1, self.in_dim)
        x = self.state_proc_model(x)
        x = x.view(-1, self.sequence_length, self.rnn_input_dim)
        hid_0 = self.init_hidden(batch_size)
        _, final_hid = self.rnn(x, hid_0)
        final_hid.squeeze_(dim=0)
        x = self.fc_out(final_hid)
        return x

    def training_step(self, x, y):
        '''
        Takes a single training step: one forward and one backwards pass
        '''
        self.state_proc_model.train()
        self.rnn.train()
        self.fc_out.train()
        self.optim.zero_grad()
        out = self(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        if self.clamp_grad:
            torch.nn.utils.clip_grad_norm(
                self.state_proc_model.parameters(), self.clamp_grad_val)
            torch.nn.utils.clip_grad_norm(
                self.rnn.parameters(), self.clamp_grad_val)
            torch.nn.utils.clip_grad_norm(
                self.fc_out.parameters(), self.clamp_grad_val)
        self.optim.step()
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.state_proc_model.eval()
        self.rnn.eval()
        self.fc_out.eval()
        return self(x).data

    def init_params(self):
        '''
        Initializes all of the model's parameters using xavier uniform initialization.
        Biases are all set to 0.01
        '''
        layers = self.state_processing_layers + [self.fc_out]
        net_util.init_layers(layers, 'Linear')

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

    def __str__(self):
        '''Overriding so that print() will print the whole network'''
        return self.state_proc_model.__str__() + \
            '\n' + self.rnn.__str__() + '\n' + self.fc_out.__str__()
