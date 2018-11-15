from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import logger, util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logger.get_logger(__name__)


class RecurrentNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized recurrent neural networks which take a sequence of states as input.

    Assumes that a single input example is organized into a 3D tensor
    batch_size x seq_len x state_dim
    The entire model consists of three parts:
        1. self.state_proc_model
        2. self.rnn_model
        3. self.model_tails

    e.g. net_spec
    "net": {
        "type": "RecurrentNet",
        "hid_layers": [],
        "hid_layers_activation": "relu",
        "rnn_hidden_size": 32,
        "rnn_num_layers": 1,
        "seq_len": 4,
        "init_fn": "xavier_uniform_",
        "clip_grad": false,
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.01
        },
        "lr_decay": "rate_decay",
        "lr_decay_frequency": 500,
        "lr_decay_min_timestep": 1000,
        "lr_anneal_timestep": 1000000,
        "update_type": "replace",
        "update_frequency": 1,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        hid_layers: list containing dimensions of the hidden layers. The last element of the list is should be the dimension of the hidden state for the recurrent layer. The other elements in the list are the dimensions of the MLP (if desired) which is to transform the state space.
        hid_layers_activation: activation function for the state_proc hidden layers
        rnn_hidden_size: rnn hidden_size
        rnn_num_layers: number of recurrent layers
        seq_len: length of the history of being passed to the net
        init_fn: weight initialization function
        clip_grad: whether to clip the gradient
        clip_grad_val: the clip value
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_decay: function to decay learning rate
        lr_decay_frequency: how many total timesteps per decay
        lr_decay_min_timestep: minimum amount of total timesteps before starting decay
        lr_anneal_timestep: timestep to anneal lr decay
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        # use generic multi-output for RNN
        out_dim = np.reshape(out_dim, -1).tolist()
        nn.Module.__init__(self)
        super(RecurrentNet, self).__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            init_fn='xavier_uniform_',
            rnn_num_layers=1,
            clip_grad=False,
            clip_grad_val=1.0,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_decay='no_decay',
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'rnn_hidden_size',
            'rnn_num_layers',
            'seq_len',
            'init_fn',
            'clip_grad',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_decay',
            'lr_decay_frequency',
            'lr_decay_min_timestep',
            'lr_anneal_timestep',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])
        # state processing model
        state_proc_dims = [self.in_dim] + self.hid_layers
        self.state_proc_model = net_util.build_sequential(state_proc_dims, self.hid_layers_activation)

        # RNN model
        self.rnn_input_dim = state_proc_dims[-1]
        self.rnn_model = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True)

        # tails
        self.model_tails = nn.ModuleList([nn.Linear(self.rnn_hidden_size, out_d) for out_d in self.out_dim])

        net_util.init_layers(self, self.init_fn)
        for module in self.modules():
            module.to(self.device)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_decay = getattr(net_util, self.lr_decay)
        # store grad norms for debugging
        self.grad_norms = []

    def __str__(self):
        return super(RecurrentNet, self).__str__() + f'\noptim: {self.optim}'

    def init_hidden(self, batch_size):
        hid = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size)
        hid = hid.to(self.device)
        return hid

    def forward(self, x):
        '''The feedforward step. Input is batch_size x seq_len x state_dim'''
        # Unstack input to (batch_size x seq_len) x state_dim in order to transform all state inputs
        batch_size = x.size(0)
        x = x.view(-1, self.in_dim)
        x = self.state_proc_model(x)
        # Restack to batch_size x seq_len x rnn_input_dim
        x = x.view(-1, self.seq_len, self.rnn_input_dim)
        hid_0 = self.init_hidden(batch_size)
        _, final_hid = self.rnn_model(x, hid_0)
        final_hid.squeeze_(dim=0)
        # return tensor if single tail, else list of tail tensors
        if len(self.model_tails) == 1:
            return self.model_tails[0](final_hid)
        else:
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(final_hid))
            return outs

    def training_step(self, x=None, y=None, loss=None, retain_graph=False, global_net=None):
        '''Takes a single training step: one forward and one backwards pass'''
        self.train()
        self.zero_grad()
        self.optim.zero_grad()
        if loss is None:
            out = self(x)
            loss = self.loss_fn(out, y)
        assert not torch.isnan(loss).any(), loss
        if net_util.to_assert_trained():
            assert_trained = net_util.gen_assert_trained(self.rnn_model)
        loss.backward(retain_graph=retain_graph)
        if self.clip_grad:
            logger.debug(f'Clipping gradient: {self.clip_grad_val}')
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        if global_net is None:
            self.optim.step()
        else:  # distributed training with global net
            net_util.push_global_grad(self, global_net)
            self.optim.step()
            net_util.pull_global_param(self, global_net)
        if net_util.to_assert_trained():
            assert_trained(self.rnn_model, loss)
        logger.debug(f'Net training_step loss: {loss}')
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.eval()
        return self(x)

    def update_lr(self, clock):
        assert 'lr' in self.optim_spec
        old_lr = self.optim_spec['lr']
        new_lr = self.lr_decay(self, clock)
        if new_lr == old_lr:
            return
        self.optim_spec['lr'] = new_lr
        logger.debug(f'Learning rate decayed from {old_lr:.6f} to {self.optim_spec["lr"]:.6f}')
        self.optim = net_util.get_optim(self, self.optim_spec)

    def store_grad_norms(self):
        '''Stores the gradient norms for debugging.'''
        norms = []
        for p_name, param in self.named_parameters():
            norms.append(param.grad.norm().item())
        self.grad_norms = norms
