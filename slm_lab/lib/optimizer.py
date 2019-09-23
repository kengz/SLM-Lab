# Custom PyTorch optimizer classes, to be registered in net_util.py
from torch.optim.optimizer import Optimizer
import itertools as it
import math
import torch


class GlobalAdam(torch.optim.Adam):
    '''
    Global Adam algorithm with shared states for Hogwild.
    Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py (MIT)
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)
        return loss


class GlobalRMSprop(torch.optim.RMSprop):
    '''
    Global RMSprop algorithm with shared states for Hogwild.
    Adapted from https://github.com/jingweiz/pytorch-rl/blob/master/optims/sharedRMSprop.py (MIT)
    '''

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=0, centered=False)

        # State initialisation (must be done before step, else will not be shared between threads)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = p.data.new().resize_(1).zero_()
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], grad, avg)
        return loss


class Lookahead(Optimizer):
    '''
    Lookahead Optimizer: k steps forward, 1 step back
    https://arxiv.org/abs/1907.08610
    Implementation modified from https://github.com/lonePatient/lookahead_pytorch; reference from https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d
    '''

    def __init__(self, params, alpha=0.5, k=5, optimizer='RAdam', **optimizer_kwargs):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        # construct base optimizer
        OptimClass = getattr(torch.optim, optimizer)
        self.optimizer = OptimClass(params, **optimizer_kwargs)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        # create and use defaults to track params to retain them in multiprocessing spawn
        self.defaults = self.optimizer.defaults
        self.defaults['alpha'] = alpha
        self.defaults['k'] = k
        for group in self.param_groups:
            group['step_counter'] = 0
        self.defaults['slow_weights'] = [[
            p.clone().detach() for p in group['params']]
            for group in self.param_groups]

        for w in it.chain(*self.defaults['slow_weights']):
            w.requires_grad = False

    def share_memory(self):
        self.optimizer.share_memory()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group, slow_weights in zip(self.param_groups, self.defaults['slow_weights']):
            group['step_counter'] += 1
            if group['step_counter'] % self.defaults['k'] != 0:
                continue
            for p, q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.defaults['alpha'], p.data - q.data)
                p.data.copy_(q.data)
        return loss


class RAdam(Optimizer):
    '''
    RAdam optimizer which stabilizes training vs. different learning rates.
    paper: On the Variance of the Adaptive Learning Rate and Beyond https://arxiv.org/abs/1908.03265
    Adapted from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py (Apache-2.0)
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=torch.zeros((10, 3)))
        super(RAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()
                state = self.state[p]
                state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.defaults['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                adap_lr = (-step_size * group['lr']).squeeze(dim=0).item()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(adap_lr, exp_avg, denom)
                else:
                    p_data_fp32.add_(adap_lr, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
