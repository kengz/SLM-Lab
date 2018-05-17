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
        in_dim: dimension of the inputs
        hid_layers: list containing dimensions of the hidden layers
        out_dim: dimension of the ouputs
        hid_layers_activation: activation function for the hidden layers
        optim_spec: parameters for initializing the optimizer
        loss_spec: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        @example:
        dict(
            1000,
            [512, 256, 128],
            10,
            hid_layers_activation='relu',
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'mse_loss'},
            clamp_grad=True,
            clamp_grad_val=2.0,
            gpu=True,
            decay_lr_factor=0.9)
        '''
        nn.Module.__init__(self)
        super(MLPNet, self).__init__(net_spec, algorithm, body)
        # set default
        util.set_attr(self, dict(
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'mse_loss'},
            clamp_grad=False,
            clamp_grad_val=1.0,
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
            'clamp_grad',
            'clamp_grad_val',
            'decay_lr',
            'decay_lr_factor',
            'decay_lr_frequency',
            'decay_lr_min_timestep',
            'update_type',
            'update_frequency',
            'polyak_weight',
            'gpu',
        ])
        # Create net and initialize params
        self.in_dim = self.body.state_dim
        self.out_dim = self.body.action_dim
        self.layers = []
        for i, layer in enumerate(self.hid_layers):
            in_D = self.in_dim if i == 0 else self.hid_layers[i - 1]
            out_D = self.hid_layers[i]
            self.layers.append(nn.Linear(in_D, out_D))
            self.layers.append(net_util.get_activation_fn(self.hid_layers_activation))
        in_D = self.hid_layers[-1] if len(self.hid_layers) > 0 else self.in_dim
        self.layers.append(nn.Linear(in_D, self.out_dim))
        self.model = nn.Sequential(*self.layers)
        self.init_params()
        if torch.cuda.is_available() and self.gpu:
            self.model.cuda()
        # Init other net variables
        self.params = list(self.model.parameters())
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        logger.info(f'loss fn: {self.loss_fn}')
        logger.info(f'optimizer: {self.optim}')
        logger.info(f'decay lr: {self.decay_lr_factor}')

    def forward(self, x):
        '''The feedforward step'''
        return self.model(x)

    def training_step(self, x, y):
        '''Takes a single training step: one forward and one backwards pass'''
        self.model.train()
        self.model.zero_grad()
        self.optim.zero_grad()
        out = self(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        if self.clamp_grad:
            logger.debug(f'Clipping gradient...')
            torch.nn.utils.clip_grad_norm(
                self.model.parameters(), self.clamp_grad_val)
        self.optim.step()
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.eval()
        return self(x).data

    def init_params(self):
        '''
        Initializes all of the model's parameters using xavier uniform initialization.
        Note: There appears to be unreproduceable behaviours in pyTorch for xavier init
        Sometimes the trainable params tests pass (see nn_test.py), other times they dont.
        Biases are all set to 0.01
        '''
        net_util.init_layers(self.layers, 'Linear')

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
        '''
        in_dim: dimension of the inputs
        hid_layers: list containing dimensions of the hidden layers
        out_dim: list containing the dimensions of the ouputs
        hid_layers_activation: activation function for the hidden layers
        optim_spec: parameters for initializing the optimizer
        loss_spec: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        @example:
        dict(
            1000,
            [512, 256, 128],
            [1, 1],
            hid_layers_activation='relu',
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'mse_loss'},
            clamp_grad=True,
            clamp_grad_val=2.0,
            gpu=True,
            decay_lr_factor=0.9)
        '''
        Net.__init__(self, net_spec, algorithm, body)
        # set default
        util.set_attr(self, dict(
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'mse_loss'},
            clamp_grad=False,
            clamp_grad_val=1.0,
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
            'clamp_grad',
            'clamp_grad_val',
            'decay_lr',
            'decay_lr_factor',
            'decay_lr_frequency',
            'decay_lr_min_timestep',
            'update_type',
            'update_frequency',
            'polyak_weight',
            'gpu',
        ])
        # Create net and initialize params
        self.in_dim = self.body.state_dim
        # set this below to something like  self.body.action_dim = [self.body.action_dim] * 2
        self.out_dim = self.body.action_dim
        self.layers = []
        # Init network body
        for i, layer in enumerate(self.hid_layers):
            in_D = self.in_dim if i == 0 else self.hid_layers[i - 1]
            out_D = self.hid_layers[i]
            self.layers.append(nn.Linear(in_D, out_D))
            self.layers.append(net_util.get_activation_fn(self.hid_layers_activation))
        in_D = self.hid_layers[-1] if len(self.hid_layers) > 0 else self.in_dim
        self.body = nn.Sequential(*self.layers)
        # Init network output heads
        self.out_layers = []
        for i, dim in enumerate(self.out_dim):
            self.out_layers.append(nn.Linear(in_D, dim))
        self.layers.append(self.out_layers)
        self.init_params()
        if torch.cuda.is_available() and self.gpu:
            self.body.cuda()
            for l in self.out_layers:
                l.cuda()
        # Init other net variables
        self.params = list(self.body.parameters())
        for layer in self.out_layers:
            self.params.extend(list(layer.parameters()))
        self.optim = net_util.get_optim_multinet(self.params, self.optim_spec)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        logger.info(f'loss fn: {self.loss_fn}')
        logger.info(f'optimizer: {self.optim}')
        logger.info(f'decay lr: {self.decay_lr_factor}')

    def forward(self, x):
        '''The feedforward step'''
        x = self.body(x)
        outs = []
        for layer in self.out_layers:
            outs.append(layer(x))
        return outs

    def training_step(self, x, y):
        '''
        Takes a single training step: one forward and one backwards pass
        '''
        raise ValueError('Should not be called on a net with heterogenous heads')
        return np.nan

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.eval()
        return [o.data for o in self(x)]

    def __str__(self):
        '''Overriding so that print() will print the whole network'''
        s = self.body.__str__()
        for layer in self.out_layers:
            s += '\n' + layer.__str__()
        return s

    def update_lr(self):
        assert 'lr' in self.optim_spec
        old_lr = self.optim_spec['lr']
        self.optim_spec['lr'] = old_lr * self.decay_lr_factor
        logger.debug(f'Learning rate decayed from {old_lr} to {self.optim_spec["lr"]}')
        self.optim = net_util.get_optim_multinet(self.params, self.optim_spec)


class MultiMLPNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network with multiple state and action heads, and a single shared body.
    '''

    def __init__(self, net_spec, algorithm, body_list):
        '''
        Multi state processing heads, single shared body, and multi action heads.
        There is one state and action head per environment
        Example:

          Action env 1     Action env 2
         _______|______    _______|______
        |  Act head 1  |  |  Act head 2  |
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
        | State head 1 |  | State head 2 |
        |______________|  |______________|

        in_dim: list of lists containing dimensions of the state processing heads
        hid_layers: list containing dimensions of the hidden layers
        out_dim: list of lists containing dimensions of the ouputs
        hid_layers_activation: activation function for the hidden layers
        optim_spec: parameters for initializing the optimizer
        loss_spec: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        @example:
        net = dict(
            [[800, 200],[400, 200]],
             [100, 50, 25],
             [[10], [15]],
             hid_layers_activation='relu',
             optim_spec={'name': 'Adam'},
             loss_spec={'name': 'mse_loss'},
             clamp_grad=True,
             clamp_grad_val2.0,
             gpu=False,
             decay_lr_factor=0.9)
        '''
        nn.Module.__init__(self)
        super(MultiMLPNet, self).__init__(net_spec, algorithm, body_list)
        # set default
        util.set_attr(self, dict(
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'mse_loss'},
            clamp_grad=False,
            clamp_grad_val=1.0,
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
            'clamp_grad',
            'clamp_grad_val',
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

        # TODO move away from list-based construction API, too many things to keep track off and might drop the ball
        self.state_heads_layers = []
        self.state_heads_models = self.make_state_heads(body_list)

        self.body_layers = []
        self.body_model = self.make_shared_body(self.state_out_concat)

        self.action_tails_layers = []
        self.action_tails_models = self.make_action_tails(body_list)

        self.init_params()
        if torch.cuda.is_available() and self.gpu:
            for l in self.state_heads_models:
                l.cuda()
            self.body_model.cuda()
            for l in self.action_tails_models:
                l.cuda()
        # Init other net variables
        self.params = []
        for model in self.state_heads_models:
            self.params.extend(list(model.parameters()))
        self.params.extend(list(self.body_model.parameters()))
        for model in self.action_tails_models:
            self.params.extend(list(model.parameters()))
        self.optim = net_util.get_optim_multinet(self.params, self.optim_spec)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        logger.info(f'loss fn: {self.loss_fn}')
        logger.info(f'optimizer: {self.optim}')
        logger.info(f'decay lr: {self.decay_lr_factor}')

    def make_state_heads(self, body_list):
        '''Creates each state head. These are stored as Sequential models in self.state_heads_models'''
        assert len(self.head_hid_layers) == len(body_list), 'multihead head hid_params inconsistent with number of bodies'
        # TODO use actual last layers instead of numbers to track interface to body
        self.state_out_concat = 0
        state_heads_models = []
        for body, hid_layers in zip(body_list, self.head_hid_layers):
            layers = []
            for i, layer in enumerate(hid_layers):
                in_D = body.state_dim if i == 0 else hid_layers[i - 1]
                out_D = hid_layers[i]
                layers.append(nn.Linear(in_D, out_D))
                layers.append(net_util.get_activation_fn(self.hid_layers_activation))
            self.state_out_concat += hid_layers[-1]
            self.state_heads_layers.append(layers)
            state_heads_models.append(nn.Sequential(*layers))
        return state_heads_models

    def make_shared_body(self, head_out_concat):
        '''Creates the shared body of the network. Stored as a Sequential model in self.body_model'''
        for i, layer in enumerate(self.body_hid_layers):
            in_D = head_out_concat if i == 0 else self.body_hid_layers[i - 1]
            out_D = layer
            self.body_layers.append(nn.Linear(in_D, out_D))
            self.body_layers.append(net_util.get_activation_fn(self.hid_layers_activation))
        return nn.Sequential(*self.body_layers)

    def make_action_tails(self, body_list):
        '''Creates each action head. These are stored as Sequential models in self.action_tails_models'''
        action_tails_models = []
        for body, hid_layers in zip(body_list, self.tail_hid_layers):
            layers = []
            for i, layer in enumerate(hid_layers):
                in_D = self.body_hid_layers[-1] if i == 0 else hid_layers[i - 1]
                out_D = hid_layers[i]
                layers.append(nn.Linear(in_D, out_D))
                layers.append(net_util.get_activation_fn(self.hid_layers_activation))
            # final output layer, no activation
            in_D = self.body_hid_layers[-1] if len(hid_layers) == 0 else hid_layers[-1]
            layers.append(nn.Linear(in_D, body.action_dim))
            self.action_tails_layers.append(layers)
            action_tails_models.append(nn.Sequential(*layers))
        return action_tails_models

    def forward(self, states):
        '''The feedforward step'''
        state_outs = []
        final_outs = []
        for i, state in enumerate(states):
            state_outs.append(self.state_heads_models[i](state))
        state_outs = torch.cat(state_outs, dim=1)
        body_out = self.body_model(state_outs)
        for i, act_model in enumerate(self.action_tails_models):
            final_outs.append(act_model(body_out))
        return final_outs

    def set_train_eval(self, train=True):
        '''Helper function to set model in training or evaluation mode'''
        nets = self.state_heads_models + self.action_tails_models
        for net in nets:
            if train:
                net.train()
                self.body_model.train()
            else:
                net.eval()
                self.body_model.eval()

    def training_step(self, x, y):
        '''
        Takes a single training step: one forward and one backwards pass. Both x and y are lists of the same length, one x and y per environment
        '''
        self.set_train_eval(True)
        self.optim.zero_grad()
        outs = self(x)
        total_loss = 0
        losses = []
        for i, out in enumerate(outs):
            loss = self.loss_fn(out, y[i])
            total_loss += loss
            losses.append(loss)
        total_loss.backward()
        if self.clamp_grad:
            torch.nn.utils.clip_grad_norm(self.params, self.clamp_grad_val)
        self.optim.step()
        nanflat_loss_a = [loss.data[0] for loss in losses]
        return nanflat_loss_a

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.set_train_eval(False)
        return [y.data for y in self(x)]

    def init_params(self):
        '''
        Initializes all of the model's parameters using xavier uniform initialization.
        Biases are all set to 0.01
        '''
        layers = []
        for l in self.state_heads_layers:
            layers.extend(l)
        layers.extend(self.body_layers)
        for l in self.action_tails_layers:
            layers.extend(l)
        net_util.init_layers(layers, 'Linear')

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

    def __str__(self):
        '''Overriding so that print() will print the whole network'''
        s = ""
        for net in self.state_heads_models:
            s += net.__str__() + '\n'
        s += self.body_model.__str__()
        for net in self.action_tails_models:
            s += '\n' + net.__str__()
        return s

    def update_lr(self):
        assert 'lr' in self.optim_spec
        old_lr = self.optim_spec['lr']
        self.optim_spec['lr'] = old_lr * self.decay_lr_factor
        logger.info(f'Learning rate decayed from {old_lr} to {self.optim_spec["lr"]}')
        self.optim = net_util.get_optim_multinet(self.params, self.optim_spec)
