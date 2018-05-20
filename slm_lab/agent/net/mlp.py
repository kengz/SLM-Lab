from slm_lab.lib import logger, util
from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logger.get_logger(__name__)


class MLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network
    '''

    def __init__(self, net_spec, algorithm, body):
        '''
        net_spec:
        hid_layers: list containing dimensions of the hidden layers
        hid_layers_activation: activation function for the hidden layers
        optim_spec: parameters for initializing the optimizer
        loss_spec: measure of error between model predictions and correct outputs
        clip_grad: whether to clip the gradient
        clip_grad_val: the clip value
        decay_lr: whether to decay learning rate
        decay_lr_factor: the multiplicative decay factor
        decay_lr_frequency: how many total timesteps per decay
        decay_lr_min_timestep: minimum amount of total timesteps before starting decay
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_weight: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        nn.Module.__init__(self)
        super(MLPNet, self).__init__(net_spec, algorithm, body)
        # set default
        util.set_attr(self, dict(
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'MSELoss'},
            clip_grad=False,
            clip_grad_val=1.0,
            decay_lr_factor=0.9,
            update_type='replace',
            update_frequency=1,
            polyak_weight=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'optim_spec',
            'loss_spec',
            'clip_grad',
            'clip_grad_val',
            'decay_lr',
            'decay_lr_factor',
            'decay_lr_frequency',
            'decay_lr_min_timestep',
            'update_type',
            'update_frequency',
            'polyak_weight',
            'gpu',
        ])

        dims = [self.body.state_dim] + self.hid_layers
        self.model = net_util.build_sequential(dims, self.hid_layers_activation)
        # add last layer with no activation
        self.model.add_module(str(len(self.model)), nn.Linear(dims[-1], self.body.action_dim))
        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()

        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        logger.info(f'loss fn: {self.loss_fn}')
        logger.info(f'optimizer: {self.optim}')

    def forward(self, x):
        '''The feedforward step'''
        return self.model(x)

    def training_step(self, x=None, y=None, loss=None):
        '''
        Takes a single training step: one forward and one backwards pass
        More most RL usage, we have custom, often complication, loss functions. Compute its value and put it in a pytorch tensor then pass it in as loss
        '''
        self.train()
        self.zero_grad()
        self.optim.zero_grad()
        if loss is None:
            out = self(x)
            loss = self.loss_fn(out, y)
        loss.backward()
        if self.clip_grad:
            logger.debug(f'Clipping gradient...')
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_grad_val)
        self.optim.step()
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model
        returns: network output given input x
        '''
        self.eval()
        return self(x)

    def update_lr(self):
        assert 'lr' in self.optim_spec
        old_lr = self.optim_spec['lr']
        self.optim_spec['lr'] = old_lr * self.decay_lr_factor
        logger.info(f'Learning rate decayed from {old_lr} to {self.optim_spec["lr"]}')
        self.optim = net_util.get_optim(self, self.optim_spec)


class MLPHeterogenousTails(MLPNet):
    '''
    Class for generating arbitrary sized feedforward neural network, with a heterogenous set of output tails that may correspond to different values. For example, the mean or std deviation of a continous policy, the state-value estimate, or the logits of a categorical action distribution
    '''

    def __init__(self, net_spec, algorithm, body):
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, algorithm, body)
        # set default
        util.set_attr(self, dict(
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'MSELoss'},
            clip_grad=False,
            clip_grad_val=1.0,
            decay_lr_factor=0.9,
            update_type='replace',
            update_frequency=1,
            polyak_weight=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'optim_spec',
            'loss_spec',
            'clip_grad',
            'clip_grad_val',
            'decay_lr',
            'decay_lr_factor',
            'decay_lr_frequency',
            'decay_lr_min_timestep',
            'update_type',
            'update_frequency',
            'polyak_weight',
            'gpu',
        ])

        dims = [self.body.state_dim] + self.hid_layers
        self.model_body = net_util.build_sequential(dims, self.hid_layers_activation)
        # multi-tail output layer
        self.model_tails = nn.ModuleList([nn.Linear(dims[-1], out_d) for out_d in self.body.action_dim])
        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()

        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        logger.info(f'loss fn: {self.loss_fn}')
        logger.info(f'optimizer: {self.optim}')

    def forward(self, x):
        '''The feedforward step'''
        x = self.model_body(x)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(x))
        return outs

    def __str__(self):
        '''Overriding so that print() will print the whole network'''
        s = self.model_body.__str__()
        s += '\nTail:'
        for model_tail in self.model_tails:
            s += '\n' + model_tail.__str__()
        return s


class MultiMLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network with multiple state and action heads, and a single shared body.
    '''

    def __init__(self, net_spec, algorithm, body_list):
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
        super(MultiMLPNet, self).__init__(net_spec, algorithm, body_list)
        # set default
        util.set_attr(self, dict(
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'MSELoss'},
            clip_grad=False,
            clip_grad_val=1.0,
            decay_lr_factor=0.9,
            update_type='replace',
            update_frequency=1,
            polyak_weight=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'hid_layers',
            'hid_layers_activation',
            'optim_spec',
            'loss_spec',
            'clip_grad',
            'clip_grad_val',
            'decay_lr',
            'decay_lr_factor',
            'decay_lr_frequency',
            'decay_lr_min_timestep',
            'update_type',
            'update_frequency',
            'polyak_weight',
            'gpu',
        ])
        assert len(self.hid_layers) == 3, 'Your hidden layers must specify [*heads], [body], [*tails]. If not, use MLPHeterogenousTails'
        self.head_hid_layers = self.hid_layers[0]
        self.body_hid_layers = self.hid_layers[1]
        self.tail_hid_layers = self.hid_layers[2]
        if len(self.head_hid_layers) == 1:
            self.head_hid_layers = self.head_hid_layers * len(body_list)
        if len(self.tail_hid_layers) == 1:
            self.tail_hid_layers = self.tail_hid_layers * len(body_list)

        self.model_heads = self.build_model_heads(body_list)
        heads_out_dim = np.sum([head[-1].out_features for head in self.model_heads])
        dims = [heads_out_dim] + self.body_hid_layers
        self.model_body = net_util.build_sequential(dims, self.hid_layers_activation)
        self.model_tails = self.build_model_tails(body_list)
        net_util.init_layers(self.modules())
        if torch.cuda.is_available() and self.gpu:
            for module in self.modules():
                module.cuda()

        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        logger.info(f'loss fn: {self.loss_fn}')
        logger.info(f'optimizer: {self.optim}')

    def build_model_heads(self, body_list):
        '''Build each model_head. These are stored as Sequential models in model_heads'''
        assert len(self.head_hid_layers) == len(body_list), 'Hydra head hid_params inconsistent with number of bodies'
        model_heads = nn.ModuleList()
        for body, hid_layers in zip(body_list, self.head_hid_layers):
            dims = [body.state_dim] + hid_layers
            model_head = net_util.build_sequential(dims, self.hid_layers_activation)
            model_heads.append(model_head)
        return model_heads

    def build_model_tails(self, body_list):
        '''Build each model_tail. These are stored as Sequential models in model_tails'''
        assert len(self.tail_hid_layers) == len(body_list), 'Hydra tail hid_params inconsistent with number of bodies'
        model_tails = nn.ModuleList()
        for body, hid_layers in zip(body_list, self.tail_hid_layers):
            dims = hid_layers
            model_tail = net_util.build_sequential(dims, self.hid_layers_activation)
            model_tail.add_module(str(len(model_tail)), nn.Linear(dims[-1], body.action_dim))
            model_tails.append(model_tail)
        return model_tails

    def forward(self, xs):
        '''The feedforward step'''
        head_xs = []
        for model_head, x in zip(self.model_heads, xs):
            head_xs.append(model_head(x))
        head_xs = torch.cat(head_xs, dim=1)
        body_x = self.body_model(head_xs)
        outs = []
        for model_tail in self.model_tails:
            outs.append(model_tail(body_x))
        return outs

    def training_step(self, xs=None, ys=None, loss=None):
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
                total_loss += loss
        total_loss.backward()
        if self.clip_grad:
            logger.debug(f'Clipping gradient...')
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip_grad_val)
        self.optim.step()
        return total_loss

    def __str__(self):
        '''Overriding so that print() will print the whole network'''
        s = 'Head'
        for net in self.model_heads:
            s += net.__str__() + '\n'
        s += '\nBody:\n'
        s += self.body_model.__str__()
        s += '\nTail:'
        for net in self.model_tails:
            s += '\n' + net.__str__()
        return s
