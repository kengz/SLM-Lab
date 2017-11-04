import pytest
import torch
from torch.autograd import Variable
from unity_lab.agent.net.feedforward import MLPNet
import numpy as np
SMALL_NUM = 0.000000001
LARGE_NUM = 100000

nets = [MLPNet(10, [5, 3], 2),
        MLPNet(20, [10, 50, 5], 2),
        MLPNet(10, [], 5)]
xs = [Variable(torch.ones((2, 10))),
      Variable(torch.ones((2, 20))),
      Variable(torch.ones((5, 10)))]
ys = [Variable(torch.zeros((2, 2))),
      Variable(torch.zeros((2, 2))),
      Variable(torch.zeros((5, 5)))]
losses = [None, None, None]
steps_list = [3, 3, 3]


class TestNet:
    '''
    Base class for unit testing neural network training
    '''

    @pytest.mark.parametrize("net", [nets[0], nets[1], nets[2]])
    @staticmethod
    def gather_trainable_params(net):
        '''
        Gathers parameters that should be trained into a list
        net: instance of torch.nn.Module or a derived class
        returns: copy of a list of fixed params
        '''
        return [param.clone() for param in net.parameters()]

    @pytest.mark.parametrize("net", [nets[0], nets[1], nets[2]])
    @staticmethod
    def gather_fixed_params(net):
        '''
        Gathers parameters that should be fixed into a list
        net: instance of torch.nn.Module or a derived class
        returns: copy of a list of fixed params
        '''
        return None

    @pytest.mark.parametrize("net", [nets[0], nets[1], nets[2]])
    @staticmethod
    def test_trainable(net):
        '''
        Checks that trainable parameters actually change during training
        net: instance of torch.nn.Module or a derived class
        returns: true if all trainable params change, false otherwise
        '''
        print("Running check_trainable test:")
        flag = True
        before_params = TestNet.gather_trainable_params(net)
        dummy_input = Variable(torch.ones((2, net.in_dim)))
        dummy_output = Variable(torch.zeros((2, net.out_dim)))
        loss = net.training_step(dummy_input, dummy_output)
        after_params = TestNet.gather_trainable_params(net)
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
        assert flag == True

    @pytest.mark.parametrize("net", [nets[0], nets[1], nets[2]])
    @staticmethod
    def test_fixed(net):
        '''
        Checks that fixed parameters don't change during training
        net: instance of torch.nn.Module or a derived class
        returns: true if all fixed params don't change, false otherwise
        '''
        print("Running check_fixed test:")
        flag = True
        before_params = TestNet.gather_fixed_params(net)
        dummy_input = Variable(torch.ones((2, net.in_dim)))
        dummy_output = Variable(torch.zeros((2, net.out_dim)))
        loss = net.training_step(dummy_input, dummy_output)
        after_params = TestNet.gather_fixed_params(net)
        i = 0
        if before_params is not None:
            for b, a in zip(before_params, after_params):
                if torch.sum(b.data) != torch.sum(a.data):
                    print("FAIL")
                    flag = False
                    i += 1
        if flag:
            print("PASS")
        assert flag == True

    @pytest.mark.parametrize("net" "x", "y", "loss", "steps", [
        (nets[0], xs[0], ys[0], losses[0], steps_list[0]),
        (nets[1], xs[1], ys[1], losses[1], steps_list[1]),
        (nets[2], xs[2], ys[2], losses[2], steps_list[2])])
    @staticmethod
    def test_gradient_size(net, x, y, steps=3):
        ''' Checks for exploding and vanishing gradients '''
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
        assert flag == True

    @pytest.mark.parametrize("net", "loss", [
        (nets[0], losses[0]),
        (nets[1], losses[1]),
        (nets[2], losses[2])])
    @staticmethod
    def test_loss_input(net, loss):
        ''' Checks that the inputs to the loss function are correct '''
        # TODO: e.g. loss is not CrossEntropy when output has one dimension
        #       e.g. softmax has not been applied with CrossEntropy loss
        #       (includes it)
        assert loss == None

    @pytest.mark.parametrize("net", [nets[0], nets[1], nets[2]])
    @staticmethod
    def test_output(net):
        ''' Checks that the output of the net is not zero or nan '''
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
        assert flag == True

    @pytest.mark.parametrize("net", [nets[0], nets[1], nets[2]])
    @staticmethod
    def test_params_not_zero(net):
        ''' Checks that the parameters of the net are not zero '''
        print("Running check_params_not_zero test")
        flag = True
        for i, param in enumerate(net.parameters()):
            if torch.sum(torch.abs(param.data)) < SMALL_NUM:
                print("FAIL: layer {}".format(i))
                flag = False
        if flag:
            print("PASS")
        assert flag == True
