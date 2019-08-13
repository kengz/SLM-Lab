# Structured Control Nets http://proceedings.mlr.press/v80/srouji18a/srouji18a.pdf
# used for policy networks
from slm_lab.agent.net.base import Net
from slm_lab.agent.net.conv import ConvNet
from slm_lab.agent.net.mlp import MLPNet
from slm_lab.agent.net import net_util
from slm_lab.lib import util
import pydash as ps
import torch
import torch.nn as nn


class SCNMLPNet(MLPNet):
    def __init__(self, net_spec, in_dim, out_dim):
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
        dims = [in_dim] + self.hid_layers
        self.model = net_util.build_fc_model(dims, self.hid_layers_activation)
        # add last layer with no activation
        # self.model_tail = net_util.build_fc_model([dims[-1], self.out_dim], self.out_layer_activation)
        if ps.is_integer(self.out_dim):
            self.model_tail = net_util.build_fc_model([dims[-1], self.out_dim], self.out_layer_activation)
            # linear control layer
            self.linear_model_tail = net_util.build_fc_model([in_dim, self.out_dim], None)
        else:
            if not ps.is_list(self.out_layer_activation):
                self.out_layer_activation = [self.out_layer_activation] * len(out_dim)
            assert len(self.out_layer_activation) == len(self.out_dim)
            tails = []
            for out_d, out_activ in zip(self.out_dim, self.out_layer_activation):
                tail = net_util.build_fc_model([dims[-1], out_d], out_activ)
                tails.append(tail)
                if len(tails) == 1:
                    self.linear_model_tail = net_util.build_fc_model([in_dim, out_d], None)
            self.model_tails = nn.ModuleList(tails)


        net_util.init_layers(self, self.init_fn)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def forward(self, x):
        x_n = self.model(x)
        if hasattr(self, 'model_tails'):
            outs = []
            for idx, model_tail in enumerate(self.model_tails):
                sub_x_n = model_tail(x_n)
                if idx == 1:
                    x_l = self.linear_model_tail(x)
                    out = torch.add(sub_x_n, x_l)
                else:
                    out = sub_x_n
                outs.append(out)
            return outs
        else:
            x_n = self.model_tail(x_n)
            x_l = self.linear_model_tail(x)
            return torch.add(x_n, x_l)
