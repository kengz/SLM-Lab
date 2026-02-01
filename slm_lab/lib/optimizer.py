# Custom PyTorch optimizer classes, to be registered in net_util.py
# Global variants for A3C Hogwild training (CPU-only, see net_util.init_global_nets)
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
        '''Share optimizer state across processes for Hogwild (CPU-only)'''
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Move to CPU first (share_memory only works on CPU tensors)
                state['step'] = state['step'].cpu()
                state['exp_avg'] = state['exp_avg'].cpu()
                state['exp_avg_sq'] = state['exp_avg_sq'].cpu()
                # Now share memory for multiprocessing
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
                # Ensure grad is on same device as state (CPU for shared memory)
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if grad.device != exp_avg.device:
                    grad = grad.to(exp_avg.device)
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data.to(grad.device), alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
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
        '''Share optimizer state across processes for Hogwild (CPU-only)'''
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Move to CPU first (share_memory only works on CPU tensors)
                state['step'] = state['step'].cpu()
                state['square_avg'] = state['square_avg'].cpu()
                # Now share memory for multiprocessing
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
                # Ensure grad is on same device as state (CPU for shared memory)
                grad = p.grad.data
                state = self.state[p]
                square_avg = state['square_avg']
                if grad.device != square_avg.device:
                    grad = grad.to(square_avg.device)
                alpha = group['alpha']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data.to(grad.device), alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])
                p.data.addcdiv_(grad, avg, value=-group['lr'])
        return loss


