from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import logger, util
import numpy as np
import pydash as ps
import torch
import torch.nn as nn

logger = logger.get_logger(__name__)


class RecurrentNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized recurrent neural networks which take a sequence of states as input.

    Assumes that a single input example is organized into a 3D tensor
    batch_size x seq_len x state_dim
    The entire model consists of three parts:
        1. self.fc_model (state processing)
        2. self.rnn_model
        3. self.model_tails

    e.g. net_spec
    "net": {
        "type": "RecurrentNet",
        "shared": true,
        "cell_type": "GRU",
        "fc_hid_layers": [],
        "hid_layers_activation": "relu",
        "out_layer_activation": null,
        "rnn_hidden_size": 32,
        "rnn_num_layers": 1,
        "bidirectional": False,
        "seq_len": 4,
        "init_fn": "xavier_uniform_",
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.01
        },
        "lr_scheduler_spec": {
            "name": "StepLR",
            "step_size": 30,
            "gamma": 0.1
        },
        "update_type": "replace",
        "update_frequency": 1,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        cell_type: any of RNN, LSTM, GRU
        fc_hid_layers: list of fc layers preceeding the RNN layers
        hid_layers_activation: activation function for the fc hidden layers
        out_layer_activation: activation function for the output layer, same shape as out_dim
        rnn_hidden_size: rnn hidden_size
        rnn_num_layers: number of recurrent layers
        bidirectional: if RNN should be bidirectional
        seq_len: length of the history of being passed to the net
        init_fn: weight initialization function
        clip_grad_val: clip gradient norm if value is not None
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_scheduler_spec: Pytorch optim.lr_scheduler
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        nn.Module.__init__(self)
        super().__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            out_layer_activation=None,
            cell_type='GRU',
            rnn_num_layers=1,
            bidirectional=False,
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'cell_type',
            'fc_hid_layers',
            'hid_layers_activation',
            'out_layer_activation',
            'rnn_hidden_size',
            'rnn_num_layers',
            'bidirectional',
            'seq_len',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])
        # restore proper in_dim from env stacked state_dim (stack_len, *raw_state_dim)
        self.in_dim = in_dim[1:] if len(in_dim) > 2 else in_dim[1]
        # fc body: state processing model
        if ps.is_empty(self.fc_hid_layers):
            self.rnn_input_dim = self.in_dim
        else:
            fc_dims = [self.in_dim] + self.fc_hid_layers
            self.fc_model = net_util.build_fc_model(fc_dims, self.hid_layers_activation)
            self.rnn_input_dim = fc_dims[-1]

        # RNN model
        self.rnn_model = getattr(nn, net_util.get_nn_name(self.cell_type))(
            input_size=self.rnn_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True, bidirectional=self.bidirectional)

        # tails. avoid list for single-tail for compute speed
        if ps.is_integer(self.out_dim):
            self.model_tail = net_util.build_fc_model([self.rnn_hidden_size, self.out_dim], self.out_layer_activation)
        else:
            if not ps.is_list(self.out_layer_activation):
                self.out_layer_activation = [self.out_layer_activation] * len(out_dim)
            assert len(self.out_layer_activation) == len(self.out_dim)
            tails = []
            for out_d, out_activ in zip(self.out_dim, self.out_layer_activation):
                tail = net_util.build_fc_model([self.rnn_hidden_size, out_d], out_activ)
                tails.append(tail)
            self.model_tails = nn.ModuleList(tails)

        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self, self.lr_scheduler_spec)
        self.to(self.device)
        self.train()

    def __str__(self):
        return super().__str__() + f'\noptim: {self.optim}'

    def forward(self, x):
        '''The feedforward step. Input is batch_size x seq_len x state_dim'''
        # Unstack input to (batch_size x seq_len) x state_dim in order to transform all state inputs
        batch_size = x.size(0)
        x = x.view(-1, self.in_dim)
        if hasattr(self, 'fc_model'):
            x = self.fc_model(x)
        # Restack to batch_size x seq_len x rnn_input_dim
        x = x.view(-1, self.seq_len, self.rnn_input_dim)
        if self.cell_type == 'LSTM':
            _output, (h_n, c_n) = self.rnn_model(x)
        else:
            _output, h_n = self.rnn_model(x)
        hid_x = h_n[-1]  # get final time-layer
        # return tensor if single tail, else list of tail tensors
        if hasattr(self, 'model_tails'):
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(hid_x))
            return outs
        else:
            return self.model_tail(hid_x)

    @net_util.dev_check_training_step
    def training_step(self, x=None, y=None, loss=None, retain_graph=False, lr_clock=None):
        '''Takes a single training step: one forward and one backwards pass'''
        if hasattr(self, 'model_tails') and x is not None:
            raise ValueError('Loss computation from x,y not supported for multitails')
        self.lr_scheduler.step(epoch=ps.get(lr_clock, 'total_t'))
        self.optim.zero_grad()
        if loss is None:
            out = self(x)
            loss = self.loss_fn(out, y)
        assert not torch.isnan(loss).any(), loss
        loss.backward(retain_graph=retain_graph)
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        self.optim.step()
        lr_clock.tick('grad_step')
        return loss
