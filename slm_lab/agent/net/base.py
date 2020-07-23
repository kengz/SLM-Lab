from abc import ABC, abstractmethod
from slm_lab.agent.net import net_util
import pydash as ps
import torch
import torch.nn as nn
import numpy as np
from slm_lab.lib import logger
logger = logger.get_logger(__name__)

class Net(ABC):
    '''Abstract Net class to define the API methods'''

    def __init__(self, net_spec, in_dim, out_dim, clock):
        '''
        @param {dict} net_spec is the spec for the net
        @param {int|list} in_dim is the input dimension(s) for the network. Usually use in_dim=body.state_dim
        @param {int|list} out_dim is the output dimension(s) for the network. Usually use out_dim=body.action_dim
        '''
        self.net_spec = net_spec
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_norms = None  # for debugging
        if self.net_spec.get('gpu'):
            if torch.cuda.device_count():
                self.device = f'cuda:{net_spec.get("cuda_id", 0)}'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        self.opt_step = 0
        self.n_frames_at_init = clock.get('frame')

    @abstractmethod
    def forward(self):
        '''The forward step for a specific network architecture'''
        raise NotImplementedError

    @net_util.dev_check_train_step
    def train_step(self, loss, optim, lr_scheduler=None, clock=None, global_net=None):
        if lr_scheduler is not None:
            # Overwritte with scalar value
            if np.isscalar(lr_scheduler):
                print(lr_scheduler)
                for param_group in optim.param_groups:
                    param_group['lr'] = torch.tensor(lr_scheduler).float()
            else:

                n_frame_since_init = clock.get('frame') - self.n_frames_at_init
                if hasattr(lr_scheduler,"last_epoch"):
                    while lr_scheduler.last_epoch < n_frame_since_init:
                        lr_scheduler.step()
                else:
                    lr_scheduler.step(epoch=n_frame_since_init)

        # optim.zero_grad()
        loss.backward()

        #Debug
        # plot_grad_flow(self.named_parameters())

        if self.clip_grad_val is not None:
            tot_total_norm_before_clipping = nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
            if tot_total_norm_before_clipping > self.clip_grad_val:
                logger.info(f"clipping grad norm from {tot_total_norm_before_clipping} to {self.clip_grad_val}")
        if global_net is not None:
            net_util.push_global_grads(self, global_net)
        optim.step()
        if global_net is not None:
            net_util.copy(global_net, self)
        optim.zero_grad()

        # if clock is not None:
        # clock.tick('opt_step')
        # TODO check that this is supported
        self.opt_step += 1
        # optim.zero_grad()

        return loss

    def store_grad_norms(self):
        '''Stores the gradient norms for debugging.'''
        norms = [param.grad.norm().item() for param in self.parameters()]
        self.grad_norms = norms

    def _adapt_input_dims_to_net(self,in_dim):
        """Adapt the input dim for the net to the needed shape"""
        raise NotImplementedError

    def _adapt_input_to_net(self,observation):
        """Adapt the input for the net to the needed shape"""
        raise NotImplementedError
#
import matplotlib.pyplot as plt
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

#
# def plot_grad_flow_v2(named_parameters):
#     '''Plots the gradients flowing through different layers in the net during training.
#     Can be used for checking for possible gradient vanishing / exploding problems.
#
#     Usage: Plug this function in Trainer class after loss.backwards() as
#     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#     ave_grads = []
#     max_grads = []
#     layers = []
#     for n, p in named_parameters:
#         if (p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#             max_grads.append(p.grad.abs().max())
#     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
#     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
#     plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
#     plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(left=0, right=len(ave_grads))
#     plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.legend([Line2D([0], [0], color="c", lw=4),
#                 Line2D([0], [0], color="b", lw=4),
#                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])