from slm_lab.lib import logger
from slm_lab.agent.net import net_util
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network
    '''

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 hid_layers_activation=None,
                 optim_param=None,
                 loss_param=None,
                 clamp_grad=False,
                 clamp_grad_val=1.0,
                 gpu=False):
        '''
        in_dim: dimension of the inputs
        hid_dim: list containing dimensions of the hidden layers
        out_dim: dimension of the ouputs
        hid_layers_activation: activation function for the hidden layers
        optim_param: parameters for initializing the optimizer
        loss_param: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        @example:
        net = MLPNet(
                1000,
                [512, 256, 128],
                10,
                hid_layers_activation='relu',
                optim_param={'name': 'Adam'},
                loss_param={'name': 'mse_loss'},
                clamp_grad=True,
                clamp_grad_val=2.0,
                gpu=True)
        '''
        super(MLPNet, self).__init__()
        # Create net and initialize params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        for i, layer in enumerate(hid_dim):
            in_D = in_dim if i == 0 else hid_dim[i - 1]
            out_D = hid_dim[i]
            self.layers += [nn.Linear(in_D, out_D)]
            self.layers += [net_util.get_activation_fn(hid_layers_activation)]
        in_D = hid_dim[-1] if len(hid_dim) > 0 else in_dim
        self.layers += [nn.Linear(in_D, out_dim)]
        self.model = nn.Sequential(*self.layers)
        self.init_params()
        if torch.cuda.is_available() and gpu:
            self.model.cuda()
        # Init other net variables
        self.params = list(self.model.parameters())
        self.optim_param = optim_param
        self.optim = net_util.get_optim(self, self.optim_param)
        self.loss_fn = net_util.get_loss_fn(self, loss_param)
        self.clamp_grad = clamp_grad
        self.clamp_grad_val = clamp_grad_val

    def forward(self, x):
        '''The feedforward step'''
        return self.model(x)

    def training_step(self, x, y):
        '''
        Takes a single training step: one forward and one backwards pass
        '''
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
        assert 'lr' in self.optim_param
        old_lr = self.optim_param['lr']
        self.optim_param['lr'] = old_lr * 0.9
        logger.debug(f'Learning rate decayed from {old_lr} to {self.optim_param["lr"]}')
        self.optim = net_util.get_optim(self, self.optim_param)


class MLPHeterogenousHeads(MLPNet):
    '''
    Class for generating arbitrary sized feedforward neural network, with a heterogenous set of output heads that may correspond to different values. For example, the mean or std deviation of a continous policy, the state-value estimate, or the logits of a categorical action distribution
    '''

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 hid_layers_activation=None,
                 optim_param=None,
                 loss_param=None,
                 clamp_grad=False,
                 clamp_grad_val=1.0,
                 gpu=False):
        '''
        in_dim: dimension of the inputs
        hid_dim: list containing dimensions of the hidden layers
        out_dim: list containing the dimensions of the ouputs
        hid_layers_activation: activation function for the hidden layers
        optim_param: parameters for initializing the optimizer
        loss_param: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        @example:
        net = MLPHeterogenousHeads(
                1000,
                [512, 256, 128],
                [1, 1],
                hid_layers_activation='relu',
                optim_param={'name': 'Adam'},
                loss_param={'name': 'mse_loss'},
                clamp_grad=True,
                clamp_grad_val=2.0,
                gpu=True)
        '''
        nn.Module.__init__(self)
        # Create net and initialize params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        # Init network body
        for i, layer in enumerate(hid_dim):
            in_D = in_dim if i == 0 else hid_dim[i - 1]
            out_D = hid_dim[i]
            self.layers += [nn.Linear(in_D, out_D)]
            self.layers += [net_util.get_activation_fn(hid_layers_activation)]
        in_D = hid_dim[-1] if len(hid_dim) > 0 else in_dim
        self.body = nn.Sequential(*self.layers)
        # Init network output heads
        self.out_layers = []
        for i, dim in enumerate(out_dim):
            self.out_layers += [nn.Linear(in_D, dim)]
        self.layers += [self.out_layers]
        self.init_params()
        if torch.cuda.is_available() and gpu:
            self.body.cuda()
            for l in self.out_layers:
                l.cuda()
        # Init other net variables
        self.params = list(self.body.parameters())
        for layer in self.out_layers:
            self.params.extend(list(layer.parameters()))
        self.optim_param = optim_param
        self.optim = net_util.get_optim_multinet(self.params, self.optim_param)
        self.loss_fn = net_util.get_loss_fn(self, loss_param)
        self.clamp_grad = clamp_grad
        self.clamp_grad_val = clamp_grad_val

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
        print("Error: Shouldn't be called on a net with heterogenous heads")
        sys.exit()
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
        assert 'lr' in self.optim_param
        old_lr = self.optim_param['lr']
        self.optim_param['lr'] = old_lr * 0.9
        logger.debug(f'Learning rate decayed from {old_lr} to {self.optim_param["lr"]}')
        self.optim = net_util.get_optim_multinet(self.params, self.optim_param)


class MultiMLPNet(nn.Module):
    '''
    Class for generating arbitrary sized feedforward neural network with multiple state and action heads, and a single shared body.
    '''

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 hid_layers_activation=None,
                 optim_param=None,
                 loss_param=None,
                 clamp_grad=False,
                 clamp_grad_val=1.0,
                 gpu=False):
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
        hid_dim: list containing dimensions of the hidden layers
        out_dim: list of lists containing dimensions of the ouputs
        hid_layers_activation: activation function for the hidden layers
        optim_param: parameters for initializing the optimizer
        loss_param: measure of error between model predictions and correct outputs
        clamp_grad: whether to clamp the gradient
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        @example:
        net = MLPNet(
            [[800, 200],[400, 200]],
             [100, 50, 25],
             [[10], [15]],
             hid_layers_activation='relu',
             optim_param={'name': 'Adam'},
             loss_param={'name': 'mse_loss'},
             clamp_grad=True,
             clamp_grad_val2.0,
             gpu=False)
        '''
        super(MultiMLPNet, self).__init__()
        # Create net and initialize params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_heads_layers = []
        self.state_heads_models = self.make_state_heads(
            self.in_dim, hid_layers_activation)
        self.shared_layers = []
        self.body = self.make_shared_body(
            self.state_out_concat, hid_dim, hid_layers_activation)
        self.action_heads_layers = []
        in_D = hid_dim[-1] if len(hid_dim) > 0 else self.state_out_concat
        self.action_heads_models = self.make_action_heads(
            in_D, self.out_dim, hid_layers_activation)
        self.init_params()
        if torch.cuda.is_available() and gpu:
            for l in self.state_heads_models:
                l.cuda()
            self.body.cuda()
            for l in self.action_heads_models:
                l.cuda()
        # Init other net variables
        self.params = []
        for model in self.state_heads_models:
            self.params.extend(list(model.parameters()))
        self.params.extend(list(self.body.parameters()))
        for model in self.action_heads_models:
            self.params.extend(list(model.parameters()))
        self.optim_param = optim_param
        self.optim = net_util.get_optim_multinet(self.params, self.optim_param)
        self.loss_fn = net_util.get_loss_fn(self, loss_param)
        self.clamp_grad = clamp_grad
        self.clamp_grad_val = clamp_grad_val

    def make_state_heads(self, state_heads, hid_layers_activation):
        '''Creates each state head. These are stored as Sequential
           models in self.state_heads_models'''
        self.state_out_concat = 0
        state_heads_models = []
        for head in state_heads:
            layers = []
            assert len(head) > 1
            for i, layer in enumerate(head):
                if i != 0:
                    in_D = head[i - 1]
                    out_D = head[i]
                    layers += [nn.Linear(in_D, out_D)]
                    layers += [net_util.get_activation_fn(
                        hid_layers_activation)]
            self.state_out_concat += head[-1]
            self.state_heads_layers.append(layers)
            state_heads_models.append(nn.Sequential(*layers))
        return state_heads_models

    def make_shared_body(self, in_dim, dims, hid_layers_activation):
        '''Creates the shared body of the network. Stored as a Sequential
           model in self.body'''
        for i, layer in enumerate(dims):
            in_D = in_dim if i == 0 else dims[i - 1]
            out_D = dims[i]
            self.shared_layers += [nn.Linear(in_D, out_D)]
            self.shared_layers += [
                net_util.get_activation_fn(hid_layers_activation)]
        return nn.Sequential(*self.shared_layers)

    def make_action_heads(self, in_dim, act_heads, hid_layers_activation):
        '''Creates each action head. These are stored as Sequential
           models in self.action_heads_models'''
        act_heads_models = []
        for head in act_heads:
            layers = []
            assert len(head) > 0
            for i, layer in enumerate(head):
                in_D = head[i - 1] if i > 0 else in_dim
                out_D = head[i]
                layers += [nn.Linear(in_D, out_D)]
                # No activation function in the last layer
                if i < len(head) - 1:
                    layers += [net_util.get_activation_fn(
                        hid_layers_activation)]
            self.action_heads_layers.append(layers)
            act_heads_models.append(nn.Sequential(*layers))
        return act_heads_models

    def forward(self, states):
        '''The feedforward step'''
        state_outs = []
        final_outs = []
        for i, state in enumerate(states):
            state_outs += [self.state_heads_models[i](state)]
        state_outs = torch.cat(state_outs, dim=1)
        body_out = self.body(state_outs)
        for i, act_model in enumerate(self.action_heads_models):
            final_outs += [act_model(body_out)]
        return final_outs

    def set_train_eval(self, train=True):
        '''Helper function to set model in training or evaluation mode'''
        nets = self.state_heads_models + self.action_heads_models
        for net in nets:
            if train:
                net.train()
                self.body.train()
            else:
                net.eval()
                self.body.eval()

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
        layers.extend(self.shared_layers)
        for l in self.action_heads_layers:
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
        s += self.body.__str__()
        for net in self.action_heads_models:
            s += '\n' + net.__str__()
        return s

    def update_lr(self):
        assert 'lr' in self.optim_param
        old_lr = self.optim_param['lr']
        self.optim_param['lr'] = old_lr * 0.9
        logger.debug(f'Learning rate decayed from {old_lr} to {self.optim_param["lr"]}')
        self.optim = net_util.get_optim_multinet(self.params, self.optim_param)
