import pytest
import torch
from torch.autograd import Variable


class TestNet:
    '''
    Base class for unit testing neural network training
    '''


    @staticmethod
    def gather_trainable_params(net):
        '''
        Gathers parameters that should be trained into a list
        net: instance of torch.nn.Module or a derived class
        returns: list of fixed params
        '''
        return net.parameters()


    @staticmethod
    def gather_fixed_params(net):
        '''
        Gathers parameters that should be fixed into a list
        net: instance of torch.nn.Module or a derived class
        returns: list of fixed params
        '''
        return None


    @staticmethod
    def check_trainable(net):
        '''
        Checks that trainable parameters actually change during training
        net: instance of torch.nn.Module or a derived class
        returns: true if all trainable params change, false otherwise
        '''
        print("Running check_trainable test:")
        before_params = TestNet.gather_trainable_params(net)
        dummy_input = Variable(torch.ones((2, net.in_dim)))
        dummy_output = Variable(torch.zeros((2, net.out_dim)))
        loss = net.training_step(dummy_input, dummy_output)
        after_params = TestNet.gather_trainable_params(net)
        if before_params is not None:
            for b, a in zip(before_params, after_params):
                if torch.sum(b.data) == torch.sum(a.data):
                    print(a.data)
                    print(b.data)
                    print("FAIL")
                    return False
        print("PASS")
        return True


    @staticmethod
    def check_fixed(net):
        '''
        Checks that fixed parameters don't change during training
        net: instance of torch.nn.Module or a derived class
        returns: true if all fixed params don't change, false otherwise
        '''
        print("Running check_fixed test:")
        before_params = TestNet.gather_fixed_params(net)
        dummy_input = Variable(torch.ones((2, net.in_dim)))
        dummy_output = Variable(torch.zeros((2, net.out_dim)))
        loss = net.training_step(dummy_input, dummy_output)
        after_params = TestNet.gather_fixed_params(net)
        if before_params is not None:
            for b, a in zip(before_params, after_params):
                if torch.sum(b.data) != torch.sum(a.data):
                    print("FAIL")
                    return False
        print("PASS")
        return True


    @staticmethod
    def check_gradient_size(net):
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


    @staticmethod
    def check_output(self, net):
        '''
        Checks that the output of the net is not zero or nan
        '''
        # TODO Complete for vanilla nn
        pass


    @staticmethod
    def check_params_not_zero(self, net):
        '''
        Checks that the parameters of the net are not zero
        '''
        # TODO Complete for vanilla nn
        pass
