from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import logger, util
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logger.get_logger(__name__)


class RecurrentNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized recurrent neural networks which take a sequence of
    states as input.

    Assumes that a single input example is organized into a 3D tensor
    batch_size x seq_len x state_dim
    The entire model consists of three parts:
         1. self.state_proc_model
         2. self.rnn
         3. self.out_layers
    '''

    def __init__(self, net_spec, algorithm, body):
        '''
        net_spec:
        in_dim: dimension of the states
        hid_layers: list containing dimensions of the hidden layers. The last element of the list is should be the dimension of the hidden state for the recurrent layer. The other elements in the list are the dimensions of the MLP (if desired) which is to transform the state space.
        out_dim: dimension of the output for one output, otherwise a list containing the dimensions of the ouputs for a multi-headed network
        seq_len: length of the history of being passed to the net
        hid_layers_activation: activation function for the hidden layers
        optim_spec: parameters for initializing the optimizer
        loss_spec: measure of error between model predictions and correct output
        clamp_grad: whether to clamp the gradient
        clamp_grad_val: what value to clamp the gradient at
        num_rnn_layers: number of recurrent layers
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        @example:
        dict(
            4,
            [32, 64],
            10,
            8,
            hid_layers_activation='relu',
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'mse_loss'},
            clamp_grad=False,
            gpu=True,
            decay_lr_factor=0.9)
        '''
        nn.Module.__init__(self)
        super(RecurrentNet, self).__init__(net_spec, algorithm, body)
        # set default
        util.set_attr(self, dict(
            num_rnn_layers=1,
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'mse_loss'},
            clamp_grad=False,
            clamp_grad_val=1.0,
            decay_lr_factor=0.9,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'num_rnn_layers',
            'seq_len',
            'optim_spec',
            'loss_spec',
            'clamp_grad',
            'clamp_grad_val',
            'decay_lr',
            'decay_lr_factor',
            'decay_lr_frequency',
            'decay_lr_min_timestep',
            'gpu',
        ])
        # Create net and initialize params
        self.in_dim = self.body.state_dim
        self.out_dim = np.reshape(self.body.action_dim, -1).tolist()
        # Create net and initialize params
        # TODO recursive naming. avoid
        self.hidden_size = self.hid_layers[-1]
        self.state_processing_layers = []
        self.state_proc_model = self.build_state_proc_layers(
            self.hid_layers[:-1], self.hid_layers_activation)
        self.rnn_input_dim = self.hid_layers[-2] if len(self.hid_layers) > 1 else self.in_dim
        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_rnn_layers,
            batch_first=True)
        # Init network output heads
        self.out_layers = []
        for dim in self.out_dim:
            self.out_layers.append(nn.Linear(self.hidden_size, dim))
        self.layers = [self.state_processing_layers] + [self.rnn] + [self.out_layers]
        self.num_hid_layers = None
        self.init_params()
        if torch.cuda.is_available() and self.gpu:
            self.state_proc_model.cuda()
            self.rnn.cuda()
            for l in self.out_layers:
                l.cuda()
        # Init other net variables
        self.params = list(self.state_proc_model.parameters()) + list(self.rnn.parameters())
        for layer in self.out_layers:
            self.params.extend(list(layer.parameters()))
        # Store named parameters for unit testing
        self.named_params = list(self.state_proc_model.named_parameters()) + list(self.rnn.named_parameters())
        for layer in self.out_layers:
            self.named_params.extend(list(layer.named_parameters()))
        self.optim = net_util.get_optim_multinet(self.params, self.optim_spec)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        logger.info(f'loss fn: {self.loss_fn}')
        logger.info(f'optimizer: {self.optim}')
        logger.info(f'decay lr: {self.decay_lr_factor}')

    def build_state_proc_layers(self, state_processing_layers, hid_layers_activation):
        '''
        Builds all of the state processing layers in the network.
        These layers are turned into a Sequential model and stored in self.state_proc_model.
        The entire model consists of three parts:
            1. self.state_proc_model
            2. self.rnn
            3. self.fc_out
        '''
        for i, layer in enumerate(state_processing_layers):
            in_D = self.in_dim if i == 0 else state_processing_layers[i - 1]
            out_D = state_processing_layers[i]
            self.state_processing_layers.append(nn.Linear(in_D, out_D))
            self.state_processing_layers.append(net_util.get_activation_fn(hid_layers_activation))
        return nn.Sequential(*self.state_processing_layers)

    def init_hidden(self, batch_size, volatile=False):
        hid = torch.zeros(self.num_rnn_layers, batch_size, self.hidden_size)
        if torch.cuda.is_available() and self.gpu:
            hid = hid.cuda()
        return Variable(hid, volatile=volatile)

    def forward(self, x):
        '''The feedforward step. Input is batch_size x seq_len x state_dim'''
        # Unstack input to (batch_size x seq_len) x state_dim in order to transform all state inputs
        batch_size = x.size(0)
        x = x.view(-1, self.in_dim)
        x = self.state_proc_model(x)
        '''Restack to batch_size x seq_len x rnn_input_dim'''
        x = x.view(-1, self.seq_len, self.rnn_input_dim)
        hid_0 = self.init_hidden(batch_size)
        _, final_hid = self.rnn(x, hid_0)
        final_hid.squeeze_(dim=0)
        # If only one head, return tensor, otherwise return list of outputs
        outs = []
        for layer in self.out_layers:
            out = layer(final_hid)
            outs.append(out)
        logger.debug3(f'Network input: {x.size()}')
        logger.debug3(f'Network input: {x.data}')
        logger.debug3(f'Network output: {outs}')
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def training_step(self, x, y):
        '''Takes a single training step: one forward and one backwards pass'''
        self.set_train_eval(train=True)
        self.optim.zero_grad()
        out = self(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        if self.clamp_grad:
            logger.debug(f'Clipping gradient...')
            torch.nn.utils.clip_grad_norm(
                self.params, self.clamp_grad_val)
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

    def set_train_eval(self, train=True):
        '''Helper function to set model in training or evaluation mode'''
        nets = [self.state_proc_model] + [self.rnn] + self.out_layers
        for net in nets:
            if train:
                net.train()
            else:
                net.eval()

    def init_params(self):
        '''
        Initializes all of the model's parameters using xavier uniform initialization.
        Biases are all set to 0.01, except for the GRU's biases which are set to 0.
        '''
        net_util.init_layers(self.layers, 'Linear')
        net_util.init_layers(self.layers, 'GRU')

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
        s = self.state_proc_model.__str__() + '\n' + self.rnn.__str__()
        for layer in self.out_layers:
            s += '\n' + layer.__str__()
        return s

    def update_lr(self):
        assert 'lr' in self.optim_spec
        old_lr = self.optim_spec['lr']
        self.optim_spec['lr'] = old_lr * self.decay_lr_factor
        logger.info(f'Learning rate decayed from {old_lr} to {self.optim_spec["lr"]}')
        self.optim = net_util.get_optim_multinet(self.params, self.optim_spec)
