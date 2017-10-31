import pytest
import torch

@pytest.fixture
class TestNet:
    '''
    Base class for unit testing neural network training
    '''


    @staticmethod
    def gather_trainable_params(self, net):
        '''
        Gathers parameters that should be trained into a list
        net: instance of torch.nn.Module or a derived class
        returns: list of fixed params
        '''
        # TODO: Complete for vanilla nn
        pass


    @staticmethod
    def gather_fixed_params(self, net):
        '''
        Gathers parameters that should be fixed into a list
        net: instance of torch.nn.Module or a derived class
        returns: list of fixed params
        '''
        # TODO: Complete for vanilla nn
        pass


    @staticmethod
    def check_trainable(self, net, train_iter_func):
        '''
        Checks that trainable parameters actually change during training
        net: instance of torch.nn.Module or a derived class
        train_iter_func: function that takes a net as input and makes a single forward
                         pass and single backprop step
        returns: true if all trainable params change, false otherwise
        '''
        # TODO: Complete for vanilla nn
        pass


    @staticmethod
    def check_fixed(self, net, train_iter_func):
        '''
        Checks that fixed parameters don't change during training
        net: instance of torch.nn.Module or a derived class
        train_iter_func: function that takes a net as input and makes a single forward
                         pass and single backprop step
        returns: true if all fixed params don't change, false otherwise
        '''
        # TODO: Complete for vanilla nn
        pass


    @staticmethod
    def check_gradient_size(self, net, train_iter_func):
        '''
        Checks for exploding and vanishing gradients
        '''
        # TODO: Complete for vanilla nn
        pass


    @staticmethod
    def check_loss_input(self, net, loss):
        '''
        Checks that the inputs to the loss function are correct
        '''
        # TODO: e.g. loss is not CrossEntropy when output has one dimension
        #       e.g. softmax has not been applied with CrossEntropy loss (includes it)
        pass
