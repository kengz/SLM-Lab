from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import util
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import pydash as ps
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    '''The transformer for RL: only the encoder and multihead attention'''

    def __init__(self, in_dim, out_dim, num_heads, num_hids, num_layers, dropout=0.5):
        super(Transformer, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(in_dim, dropout)
        encoder_layers = TransformerEncoderLayer(in_dim, num_heads, num_hids, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.in_dim = in_dim

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        return output


class TransformerNet(Net, nn.Module):
    def __init__(self, net_spec, in_dim, out_dim):
        nn.Module.__init__(self)
        super().__init__(net_spec, in_dim, out_dim)
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
            'num_heads',
            'num_hids',
            'num_layers',
            'dropout',
            # 'hid_layers',
            # 'hid_layers_activation',
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
        assert len(self.in_dim) == 2, f'Transformer only works with stacked (sequence) states'
        in_dim = self.in_dim[-1]
        # the transformer encoder feeding to mlp tail
        self.model = Transformer(in_dim=in_dim, out_dim=self.out_dim, num_heads=self.num_heads, num_hids=self.num_hids, num_layers=self.num_layers, dropout=self.dropout)
        # usual tail architecture like MLP
        if ps.is_integer(self.out_dim):
            self.model_tail = net_util.build_fc_model([in_dim, self.out_dim], self.out_layer_activation)
        else:
            if not ps.is_list(self.out_layer_activation):
                self.out_layer_activation = [self.out_layer_activation] * len(out_dim)
            assert len(self.out_layer_activation) == len(self.out_dim)
            tails = []
            for out_d, out_activ in zip(self.out_dim, self.out_layer_activation):
                tail = net_util.build_fc_model([in_dim, out_d], out_activ)
                tails.append(tail)
            self.model_tails = nn.ModuleList(tails)

        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def forward(self, x):
        '''The feedforward step'''
        x = x.to(self.device)
        x = x.transpose(0, 1)  # batch-first into seq-first
        x = self.model(x)
        if hasattr(self, 'model_tails'):
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(x)[-1, :, :])  # seq first, get last
            return outs
        else:
            out = self.model_tail(x)[-1, :, :]  # seq first, get last
            return out
