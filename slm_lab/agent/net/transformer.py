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


class PosExpand(nn.Module):
    '''Construct the positional encoding and pass through an expansion layer.'''
    def __init__(self, in_dim, num_hids, dropout):
        super().__init__()
        self.pos_encoder = PositionalEncoding(in_dim, dropout)
        self.expander = nn.Linear(in_dim, num_hids)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.expander(x)
        return x


class PosEmbedding(nn.Module):
    '''Construct the embedding from input (state) and position embedding.'''

    def __init__(self, in_dim, num_hids, dropout):
        super().__init__()
        max_seq_len = 32
        self.in_embedding = nn.Linear(in_dim, num_hids)
        self.position_embedding = nn.Embedding(max_seq_len, num_hids)
        self.LayerNorm = nn.LayerNorm(num_hids, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.shape[:2])

        inputs_embeds = self.in_embedding(x)
        position_embedding = self.position_embedding(position_ids)

        embedding = inputs_embeds + position_embedding
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class Transformer(nn.Module):
    '''The transformer for RL: only the encoder and multihead attention'''

    def __init__(self, in_dim, out_dim, num_heads, num_hids, num_layers, dropout=0.5, pos_encoder=True):
        super(Transformer, self).__init__()
        self.src_mask = None
        if pos_encoder:
            self.embedding = PosExpand(in_dim, num_hids, dropout)
        else:
            self.embedding = PosEmbedding(in_dim, num_hids, dropout)
        encoder_layers = TransformerEncoderLayer(num_hids, num_heads, num_hids, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.in_dim = in_dim

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            mask = self._generate_square_subsequent_mask(len(x)).to(x.device)
            self.src_mask = mask
        x = self.embedding(x)
        output = self.transformer_encoder(x, self.src_mask)
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
            'pos_encoder',
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
        self.model = Transformer(in_dim=in_dim, out_dim=self.out_dim, num_heads=self.num_heads, num_hids=self.num_hids, num_layers=self.num_layers, dropout=self.dropout, pos_encoder=self.pos_encoder)
        # usual tail architecture like MLP
        if ps.is_integer(self.out_dim):
            self.model_tail = net_util.build_fc_model([self.num_hids, self.out_dim], self.out_layer_activation)
        else:
            if not ps.is_list(self.out_layer_activation):
                self.out_layer_activation = [self.out_layer_activation] * len(out_dim)
            assert len(self.out_layer_activation) == len(self.out_dim)
            tails = []
            for out_d, out_activ in zip(self.out_dim, self.out_layer_activation):
                tail = net_util.build_fc_model([self.num_hids, out_d], out_activ)
                tails.append(tail)
            self.model_tails = nn.ModuleList(tails)

        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def forward(self, x):
        '''The feedforward step'''
        x = self.model(x)[:, 0]  # batch-first, get first in seq like BERT
        if hasattr(self, 'model_tails'):
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(x))
            return outs
        else:
            out = self.model_tail(x)
            return out
