import pytest
import torch
from torch.autograd import Variable
import numpy as np
SMALL_NUM = 0.000000001
LARGE_NUM = 100000


class TestNet:
    '''
    Base class for unit testing neural network training
    '''

    def test_trainable(self, test_nets):
        '''
        Checks that trainable parameters actually change during training
        net: instance of torch.nn.Module or a derived class
        returns: true if all trainable params change, false otherwise
        '''
        net = test_nets[0]
        print("Running check_trainable test:")
        flag = True
        before_params = net.gather_trainable_params()
        dummy_input = Variable(torch.ones((2, net.in_dim)))
        dummy_output = Variable(torch.zeros((2, net.out_dim)))
        loss = net.training_step(dummy_input, dummy_output)
        after_params = net.gather_trainable_params()
        i = 0
        if before_params is not None:
            for b, a in zip(before_params, after_params):
                if torch.sum(b.data) == torch.sum(a.data):
                    print("Before gradient: {}".format(a.grad))
                    print("After gradient (should not be None): {}".format(
                        b.grad))
                    print("FAIL layer {}".format(i))
                    flag = False
                    i += 1
        if flag:
            print("PASS")
        assert flag is True

    def test_fixed(self, test_nets):
        '''
        Checks that fixed parameters don't change during training
        net: instance of torch.nn.Module or a derived class
        returns: true if all fixed params don't change, false otherwise
        '''
        net = test_nets[0]
        print("Running check_fixed test:")
        flag = True
        before_params = net.gather_fixed_params()
        dummy_input = Variable(torch.ones((2, net.in_dim)))
        dummy_output = Variable(torch.zeros((2, net.out_dim)))
        loss = net.training_step(dummy_input, dummy_output)
        after_params = net.gather_fixed_params()
        i = 0
        if before_params is not None:
            for b, a in zip(before_params, after_params):
                if torch.sum(b.data) != torch.sum(a.data):
                    print("FAIL")
                    flag = False
                    i += 1
        if flag:
            print("PASS")
        assert flag is True

    def test_gradient_size(self, test_nets):
        ''' Checks for exploding and vanishing gradients '''
        net = test_nets[0]
        x = test_nets[1], y = test_nets[2]
        loss = test_nets[3]
        steps = test_nets[4]
        print("Running check_gradient_size test:")
        for i in range(steps):
            _ = net.training_step(x, y)
        flag = True
        for p in net.parameters():
            if p.grad is None:
                print("FAIL: no gradient")
                flag = False
            else:
                if torch.sum(torch.abs(p.grad.data)) < SMALL_NUM:
                    print("FAIL: tiny gradients: {}".format(
                        torch.sum(torch.abs(p.grad))))
                    flag = False
                if torch.sum(torch.abs(p.grad.data)) > LARGE_NUM:
                    print("FAIL: large gradients: {}".format(
                        torch.sum(torch.abs(p.grad))))
                    flag = False
        if flag:
            print("PASS")
        assert flag is True

    def test_loss_input(self, test_nets):
        ''' Checks that the inputs to the loss function are correct '''
        net = test_nets[0]
        loss = test_nets[3]
        # TODO: e.g. loss is not CrossEntropy when output has one dimension
        #       e.g. softmax has not been applied with CrossEntropy loss
        #       (includes it)
        assert loss is None

    def test_output(self, test_nets):
        ''' Checks that the output of the net is not zero or nan '''
        net = test_nets[0]
        print("Running check_output test. Tests if output is not 0 or NaN")
        dummy_input = Variable(torch.ones((2, net.in_dim)))
        out = net(dummy_input)
        flag = True
        if torch.sum(torch.abs(out.data)) < SMALL_NUM:
            print("FAIL")
            print(out)
            flag = False
        if np.isnan(torch.sum(out.data)):
            print("FAIL")
            print(out)
            flag = False
        if flag:
            print("PASS")
        assert flag is True

    def test_params_not_zero(self, test_nets):
        ''' Checks that the parameters of the net are not zero '''
        net = test_nets[0]
        print("Running check_params_not_zero test")
        flag = True
        for i, param in enumerate(net.parameters()):
            if torch.sum(torch.abs(param.data)) < SMALL_NUM:
                print("FAIL: layer {}".format(i))
                flag = False
        if flag:
            print("PASS")
        assert flag is True
