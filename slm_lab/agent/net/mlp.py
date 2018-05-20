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

        layers = []
        in_out_pairs = list(zip(
            [self.body.state_dim] + self.hid_layers,
            self.hid_layers + [self.body.action_dim]))
        for in_d, out_d in in_out_pairs[:-1]:  # all but last
            layers.append(nn.Linear(in_d, out_d))
            layers.append(net_util.get_activation_fn(self.hid_layers_activation)())
        # last layer no activation
        layers.append(nn.Linear(*in_out_pairs[-1]))
        self.model = nn.Sequential(*layers)
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
        self.model.train()
        self.model.zero_grad()
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

        layers = []
        in_out_pairs = list(zip(
            [self.body.state_dim] + self.hid_layers,
            self.hid_layers + [self.body.action_dim]))
        for in_d, out_d in in_out_pairs[:-1]:  # all but last
            layers.append(nn.Linear(in_d, out_d))
            layers.append(net_util.get_activation_fn(self.hid_layers_activation)())
        self.model_body = nn.Sequential(*layers)
        # multi-tail output layer
        in_d, out_ds = in_out_pairs[-1]
        self.model_tails = nn.ModuleList([nn.Linear(in_d, out_d) for out_d in out_ds])
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

    def training_step(self, x=None, y=None, loss=None):
        '''
        Takes a single training step: one forward and one backwards pass
        '''
        assert loss is not None, 'Heterogenous head should have custom loss defined elsewhere as sum of losses from the multi-tails'
        self.model_body.train()
        self.model_tails.train()
        self.model_body.zero_grad()
        self.model_tails.zero_grad()
        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad:
            logger.debug(f'Clipping gradient...')
            torch.nn.utils.clip_grad_norm(self.model_body.parameters(), self.clip_grad_val)
            torch.nn.utils.clip_grad_norm(self.model_tails.parameters(), self.clip_grad_val)
        self.optim.step()
        return loss

    def __str__(self):
        '''Overriding so that print() will print the whole network'''
        s = self.model_body.__str__()
        for model_tail in self.model_tails:
            s += '\n' + model_tail.__str__()
        return s


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

        TODO add special in dim stuff
        '''
        nn.Module.__init__(self)
        super(MultiMLPNet, self).__init__(net_spec, algorithm, body_list)
        # set default
        util.set_attr(self, dict(
            optim_spec={'name': 'Adam'},
            loss_spec={'name': 'mse_loss'},
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
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm(self.params, self.clip_grad_val)
        self.optim.step()
        nanflat_loss_a = [loss.data.item() for loss in losses]
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
