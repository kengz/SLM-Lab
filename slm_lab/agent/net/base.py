from abc import ABC, abstractmethod
import torch


class Net(ABC):
    '''Abstract Net class to define the API methods'''

    def __init__(self, net_spec, in_dim, out_dim):
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

    def store_grad_norms(self):
        '''Stores the gradient norms for debugging.'''
        norms = [param.grad.norm().item() for param in self.parameters()]
        self.grad_norms = norms
