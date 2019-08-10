# special module for Q-networks, Q(s, a) -> q
from slm_lab.agent.net.base import Net
from slm_lab.agent.net.conv import ConvNet
from slm_lab.agent.net.mlp import MLPNet
from slm_lab.agent.net import net_util
from slm_lab.lib import util
import pydash as ps
import torch
import torch.nn as nn


class QMLPNet(MLPNet):
    def __init__(self, net_spec, in_dim, out_dim):
        state_dim, action_dim = in_dim
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            out_layer_activation=None,
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
            'shared',
            'hid_layers',
            'hid_layers_activation',
            'out_layer_activation',
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
        dims = [state_dim + action_dim] + self.hid_layers
        self.model = net_util.build_fc_model(dims, self.hid_layers_activation)
        # add last layer with no activation
        self.model_tail = net_util.build_fc_model([dims[-1], self.out_dim], self.out_layer_activation)

        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def forward(self, state, action):
        s_a = torch.cat((state, action), dim=-1)
        s_a = self.model(s_a)
        return self.model_tail(s_a)


class QConvNet(ConvNet):

    def __init__(self, net_spec, in_dim, out_dim):
        state_dim, action_dim = in_dim
        assert len(state_dim) == 3  # image shape (c,w,h)
        # conv body
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, state_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            out_layer_activation=None,
            init_fn=None,
            normalize=False,
            batch_norm=True,
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
            'conv_hid_layers',
            'fc_hid_layers',
            'hid_layers_activation',
            'out_layer_activation',
            'init_fn',
            'normalize',
            'batch_norm',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        self.conv_out_dim = self.get_conv_output_size()

        # fc bodies
        self.state_fc_model = net_util.build_fc_model([self.conv_out_dim] + self.fc_hid_layers, self.hid_layers_activation)
        self.action_fc_model = net_util.build_fc_model([action_dim] + self.fc_hid_layers, self.hid_layers_activation)

        # fc and action_fc to be concatenated
        tail_in_dim = self.fc_hid_layers[-1] * 2
        self.model_tail = net_util.build_fc_model([tail_in_dim, self.out_dim], self.out_layer_activation)

        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def forward(self, state, action):
        if self.normalize:
            state = state / 255.0
        state = self.conv_model(state)
        state = state.view(state.size(0), -1)  # to (batch_size, -1)
        state = self.state_fc_model(state)
        action = self.action_fc_model(action)
        s_a = torch.cat((state, action), dim=-1)
        return self.model_tail(s_a)
