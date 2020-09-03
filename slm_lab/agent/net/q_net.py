# special module for Q-networks, Q(s, a) -> q
from slm_lab.agent.net.base import Net
from slm_lab.agent.net.conv import ConvNet
from slm_lab.agent.net.mlp import MLPNet
from slm_lab.agent.net import net_util
from slm_lab.lib import util
import pydash as ps
import torch
import torch.nn as nn
from collections import Iterable
from slm_lab.lib import logger

logger = logger.get_logger(__name__)

class QMLPNet(MLPNet):
    def __init__(self, net_spec, in_dim, out_dim, clock, name):

        in_dim = self._adapt_input_dims_to_net(in_dim)

        state_dim, action_dim = in_dim
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim, clock, name)
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

    def _adapt_input_dims_to_net(self,in_dim):
        formated = []
        for el in in_dim:
            if isinstance(el, int):
                formated.append(el)
            elif isinstance(el, Iterable):
                flatten_in_dim = 1
                for dim in el:
                    flatten_in_dim *= dim
                flatten_in_dim = int(flatten_in_dim)
                logger.info("flatten_in_dim {}".format(flatten_in_dim))
                formated.append(flatten_in_dim)
        return tuple(formated)

    def forward(self, state, action):
        state = self._adapt_input_to_net(state)
        action = self._adapt_input_to_net(action)

        s_a = torch.cat((state, action), dim=-1)
        s_a = self.model(s_a)
        return self.model_tail(s_a)


class QConvNet(ConvNet):

    def __init__(self, net_spec, in_dim, out_dim, clock, name):
        state_dim, action_dim = in_dim
        assert len(state_dim) == 3  # image shape (c,w,h)
        # conv body
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, state_dim, out_dim, clock, name)
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
        # state conv model
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        self.conv_out_dim = self.get_conv_output_size()

        # state fc model
        self.fc_model = net_util.build_fc_model([self.conv_out_dim + action_dim] + self.fc_hid_layers, self.hid_layers_activation)

        # affine transformation applied to
        tail_in_dim = self.fc_hid_layers[-1]
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
        s_a = torch.cat((state, action), dim=-1)
        s_a = self.fc_model(s_a)
        return self.model_tail(s_a)


class FiLMQConvNet(ConvNet):

    def __init__(self, net_spec, in_dim, out_dim, clock, name):
        state_dim, action_dim = in_dim
        assert len(state_dim) == 3  # image shape (c,w,h)
        # conv body
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, state_dim, out_dim, clock, name)
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
        # state conv model
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        self.conv_out_dim = self.get_conv_output_size()

        # state fc model
        self.state_fc_model = net_util.build_fc_model([self.conv_out_dim] + self.fc_hid_layers, 'sigmoid')

        # use Feature-wise Linear Modulation applied to the outputs of the last state_fc_model hid_layers
        # https://arxiv.org/pdf/1709.07871.pdf
        state_fc_out_dim = self.fc_hid_layers[-1]
        # self.action_conv_scale = net_util.build_fc_model([action_dim, self.conv_out_dim], 'sigmoid')
        # self.action_conv_shift = net_util.build_fc_model([action_dim, self.conv_out_dim], 'sigmoid')
        self.action_fc_scale = net_util.build_fc_model([action_dim, state_fc_out_dim], 'sigmoid')
        self.action_fc_shift = net_util.build_fc_model([action_dim, state_fc_out_dim], 'sigmoid')

        # affine transformation applied to
        tail_in_dim = self.fc_hid_layers[-1]
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
        # action_conv_scale = self.action_conv_scale(action)
        # action_conv_shift = self.action_conv_shift(action)
        # state = state * action_conv_scale + action_conv_shift
        state = self.state_fc_model(state)
        action_fc_scale = self.action_fc_scale(action)
        action_fc_shift = self.action_fc_shift(action)
        s_a = state * action_fc_scale + action_fc_shift
        return self.model_tail(s_a)
