from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import logger, util
import numpy as np
import pydash as ps
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logger.get_logger(__name__)


class MLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        hid_layers: list containing dimensions of the hidden layers
        hid_layers_activation: activation function for the hidden layers
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

        e.g. net_spec
        "net": {
            "type": "MLPNet",
            "hid_layers": [32],
            "hid_layers_activation": "relu",
            "clip_grad": false,
            "clip_grad_val": 1.0,
            "loss_spec": {
              "name": "MSELoss"
            },
            "optim_spec": {
              "name": "Adam",
              "lr": 0.02
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
        nn.Module.__init__(self)
        super(MLPNet, self).__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
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

        dims = [self.in_dim] + self.hid_layers
        self.model = net_util.build_sequential(dims, self.hid_layers_activation)
        # add last layer with no activation
        self.model.add_module(str(len(self.model)), nn.Linear(dims[-1], self.out_dim))

        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_decay = getattr(net_util, self.lr_decay)

    def __str__(self):
        return super(MLPNet, self).__str__() + f'\noptim: {self.optim}'

    def forward(self, x):
        '''The feedforward step'''
        return self.model(x)

    def training_step(self, x=None, y=None, loss=None, retain_graph=False):
        '''
        Takes a single training step: one forward and one backwards pass
        More most RL usage, we have custom, often complicated, loss functions. Compute its value and put it in a pytorch tensor then pass it in as loss
        '''
        self.train()
        self.zero_grad()
        self.optim.zero_grad()
        if loss is None:
            out = self(x)
            loss = self.loss_fn(out, y)
        assert not torch.isnan(loss).any(), loss
        if net_util.to_assert_trained():
            # to accommodate split model in inherited classes
            model = getattr(self, 'model', None) or getattr(self, 'model_body')
            assert_trained = net_util.gen_assert_trained(model)
        loss.backward(retain_graph=retain_graph)
        if self.clip_grad:
            logger.debug(f'Clipping gradient: {self.clip_grad_val}')
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        self.optim.step()
        if net_util.to_assert_trained():
            model = getattr(self, 'model', None) or getattr(self, 'model_body')
            assert_trained(model)
        logger.debug(f'Net training_step loss: {loss}')
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model
        returns: network output given input x
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
        logger.info(f'Learning rate decayed from {old_lr:.6f} to {self.optim_spec["lr"]:.6f}')
        self.optim = net_util.get_optim(self, self.optim_spec)


class MLPHeterogenousTails(MLPNet):
    '''
    Class for generating arbitrary sized feedforward neural network, with a heterogenous set of output tails that may correspond to different values. For example, the mean or std deviation of a continous policy, the state-value estimate, or the logits of a categorical action distribution

    e.g. net_spec
    "net": {
        "type": "MLPHeterogenousTails",
        "hid_layers": [32],
        "hid_layers_activation": "relu",
        "clip_grad": false,
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.02
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
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
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

        dims = [self.in_dim] + self.hid_layers
        self.model_body = net_util.build_sequential(dims, self.hid_layers_activation)
        # multi-tail output layer with mean and std
        self.model_tails = nn.ModuleList([nn.Linear(dims[-1], out_d) for out_d in out_dim])

        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_decay = getattr(net_util, self.lr_decay)

    def forward(self, x):
        '''The feedforward step'''
        x = self.model_body(x)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(x))
        return outs


class HydraMLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network with multiple state and action heads, and a single shared body.

    e.g. net_spec
    "net": {
        "type": "HydraMLPNet",
        "hid_layers": [
            [[32],[32]], # 2 heads with hidden layers
            [64], # body
            [] # tail, no hidden layers
        ],
        "hid_layers_activation": "relu",
        "clip_grad": false,
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.02
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
        Multi state processing heads, single shared body, and multi action tails.
        There is one state and action head per body/environment
        Example:

          env 1 state       env 2 state
         _______|______    _______|______
        |    head 1    |  |    head 2    |
        |______________|  |______________|
                |                  |
                |__________________|
         ________________|_______________
        |          Shared body           |
        |________________________________|
                         |
                 ________|_______
                |                |
         _______|______    ______|_______
        |    tail 1    |  |    tail 2    |
        |______________|  |______________|
                |                |
           env 1 action      env 2 action
        '''
        nn.Module.__init__(self)
        super(HydraMLPNet, self).__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
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
        assert len(self.hid_layers) == 3, 'Your hidden layers must specify [*heads], [body], [*tails]. If not, use MLPHeterogenousTails'
        assert isinstance(self.in_dim, list), 'Hydra network needs in_dim as list'
        assert isinstance(self.out_dim, list), 'Hydra network needs out_dim as list'
        self.head_hid_layers = self.hid_layers[0]
        self.body_hid_layers = self.hid_layers[1]
        self.tail_hid_layers = self.hid_layers[2]
        if len(self.head_hid_layers) == 1:
            self.head_hid_layers = self.head_hid_layers * len(self.in_dim)
        if len(self.tail_hid_layers) == 1:
            self.tail_hid_layers = self.tail_hid_layers * len(self.out_dim)

        self.model_heads = self.build_model_heads(in_dim)
        heads_out_dim = np.sum([head_hid_layers[-1] for head_hid_layers in self.head_hid_layers])
        dims = [heads_out_dim] + self.body_hid_layers
        self.model_body = net_util.build_sequential(dims, self.hid_layers_activation)
        self.model_tails = self.build_model_tails(out_dim)

        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_decay = getattr(net_util, self.lr_decay)

    def __str__(self):
        return super(HydraMLPNet, self).__str__() + f'\noptim: {self.optim}'

    def build_model_heads(self, in_dim):
        '''Build each model_head. These are stored as Sequential models in model_heads'''
        assert len(self.head_hid_layers) == len(in_dim), 'Hydra head hid_params inconsistent with number in dims'
        model_heads = nn.ModuleList()
        for in_d, hid_layers in zip(in_dim, self.head_hid_layers):
            dims = [in_d] + hid_layers
            model_head = net_util.build_sequential(dims, self.hid_layers_activation)
            model_heads.append(model_head)
        return model_heads

    def build_model_tails(self, out_dim):
        '''Build each model_tail. These are stored as Sequential models in model_tails'''
        model_tails = nn.ModuleList()
        if ps.is_empty(self.tail_hid_layers):
            for out_d in out_dim:
                model_tails.append(nn.Linear(self.body_hid_layers[-1], out_d))
        else:
            assert len(self.tail_hid_layers) == len(out_dim), 'Hydra tail hid_params inconsistent with number out dims'
            for out_d, hid_layers in zip(out_dim, self.tail_hid_layers):
                dims = hid_layers
                model_tail = net_util.build_sequential(dims, self.hid_layers_activation)
                model_tail.add_module(str(len(model_tail)), nn.Linear(dims[-1], out_d))
                model_tails.append(model_tail)
        return model_tails

    def forward(self, xs):
        '''The feedforward step'''
        head_xs = []
        for model_head, x in zip(self.model_heads, xs):
            head_xs.append(model_head(x))
        head_xs = torch.cat(head_xs, dim=1)
        body_x = self.model_body(head_xs)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(body_x))
        return outs

    def training_step(self, xs=None, ys=None, loss=None, retain_graph=False):
        '''
        Takes a single training step: one forward and one backwards pass. Both x and y are lists of the same length, one x and y per environment
        '''
        self.train()
        self.zero_grad()
        self.optim.zero_grad()
        if loss is None:
            outs = self(xs)
            total_loss = torch.tensor(0.0)
            for out, y in zip(outs, ys):
                loss = self.loss_fn(out, y)
                total_loss += loss.cpu()
            loss = total_loss
        assert not torch.isnan(loss).any(), loss
        if net_util.to_assert_trained():
            assert_trained = net_util.gen_assert_trained(self.model_body)
        loss.backward(retain_graph=retain_graph)
        if self.clip_grad:
            logger.debug(f'Clipping gradient: {self.clip_grad_val}')
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        self.optim.step()
        if net_util.to_assert_trained():
            assert_trained(self.model_body)
        logger.debug(f'Net training_step loss: {loss}')
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model
        returns: network output given input x
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
        logger.info(f'Learning rate decayed from {old_lr:.6f} to {self.optim_spec["lr"]:.6f}')
        self.optim = net_util.get_optim(self, self.optim_spec)


class DuelingMLPNet(MLPNet):
    '''
    Class for generating arbitrary sized feedforward neural network, with dueling heads. Intended for Q-Learning algorithms only.
    Implementation based on "Dueling Network Architectures for Deep Reinforcement Learning" http://proceedings.mlr.press/v48/wangf16.pdf

    e.g. net_spec
    "net": {
        "type": "DuelingMLPNet",
        "hid_layers": [32],
        "hid_layers_activation": "relu",
        "clip_grad": false,
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.02
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
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
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

        # Guard against inappropriate algorithms and environments
        # Build model body
        dims = [self.in_dim] + self.hid_layers
        self.model_body = net_util.build_sequential(dims, self.hid_layers_activation)
        # output layers
        self.v = nn.Linear(dims[-1], 1)  # state value
        self.adv = nn.Linear(dims[-1], out_dim)  # action dependent raw advantage
        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_decay = getattr(net_util, self.lr_decay)

    def forward(self, x):
        '''The feedforward step'''
        x = self.model_body(x)
        state_value = self.v(x)
        raw_advantages = self.adv(x)
        out = net_util.calc_q_value_logits(state_value, raw_advantages)
        return out
