import pytest
from flaky import flaky
from torch.autograd import Variable
import numpy as np
import torch
SMALL_NUM = 0.000000001
LARGE_NUM = 100000


@flaky
class TestNet:
    '''
    Base class for unit testing neural network training
    '''

    def init_dummy_input(self, net):
        if 'RecurrentNet' in net.__class__.__name__:
            dummy_input = Variable(torch.ones(
                2, net.seq_len, net.in_dim))
        elif type(net.in_dim) is int:
            dummy_input = Variable(torch.ones(2, net.in_dim))
        elif 'MultiMLPNet' in net.__class__.__name__:
            dummy_input = []
            for indim in net.in_dim:
                dummy_input.append(Variable(torch.ones(2, indim[0])))
        else:
            dummy_input = Variable(torch.ones(2, *net.in_dim))
        return dummy_input

    def init_dummy_output(self, net):
        if type(net.out_dim) is int:
            dummy_output = Variable(torch.zeros((2, net.out_dim)))
        elif 'MultiMLPNet' in net.__class__.__name__:
            dummy_output = []
            for outdim in net.out_dim:
                dummy_output.append(Variable(torch.zeros((2, outdim[-1]))))
        elif 'MLPHeterogenousTails' in net.__class__.__name__ or len(net.out_dim) > 1:
            dummy_output = []
            for outdim in net.out_dim:
                print(type(outdim), outdim)
                dummy_output.append(Variable(torch.zeros((2, outdim))))
        else:
            dummy_output = Variable(torch.zeros(2, net.out_dim[0]))
        return dummy_output

    def check_net_type(self, net):
        # Skipping test for 'MLPHeterogenousTails' because there is no training step function
        if 'MLPHeterogenousTails' in net.__class__.__name__:
            return True
        # Skipping test for 'RecurrentNet' and 'ConvNet' with multiple output heads because training step not applicable
    elif ('RecurrentNet' in net.__class__.__name__) or ('ConvNet' in net.__class__.__name__) and len(net.out_dim) > 1:
            return True
        else:
            return False

    @pytest.mark.first
    def test_params_not_zero(self, test_nets):
        ''' Checks that the parameters of the net are not zero except for GRU biases which should be zero.'''
        net = test_nets[0]
        print(net)
        flag = True
        for i, param in enumerate(net.params):
            # If net is recurrent check that biases of the recurrent layer are zero
            if 'Recurrent' in net.__class__.__name__ and 'bias_' in net.named_params[i][0]:
                print(net.named_params[i][0])
                if torch.sum(torch.abs(param.data)) != 0:
                    print("FAIL: layer {}".format(i))
                    flag = False
            elif torch.sum(torch.abs(param.data)) < SMALL_NUM:
                print("FAIL: layer {}".format(i))
                flag = False
        if flag:
            print("PASS")
        assert flag is True

    @flaky(max_runs=10)
    def test_trainable(self, test_nets):
        '''Checks that trainable parameters actually change during training.
        returns: true if all trainable params change, false otherwise'''
        net = test_nets[0]
        if self.check_net_type(net):  # Checks if test needs to be skipped for a particular net
            assert True is True
            return
        flag = True
        before_params = net.gather_trainable_params()
        dummy_input = self.init_dummy_input(net)
        dummy_output = self.init_dummy_output(net)
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
        if self.check_net_type(net):  # Checks if test needs to be skipped for a particular net
            assert True is True
            return
        flag = True
        before_params = net.gather_fixed_params()
        dummy_input = self.init_dummy_input(net)
        dummy_output = self.init_dummy_output(net)
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

    @flaky(max_runs=10)
    def test_gradient_size(self, test_nets):
        ''' Checks for exploding and vanishing gradients '''
        net = test_nets[0]
        if self.check_net_type(net):  # Checks if test needs to be skipped for a particular net
            assert True is True
            return
        x = self.init_dummy_input(net)
        y = self.init_dummy_output(net)
        loss = test_nets[1]
        steps = test_nets[2]
        for i in range(steps):
            net.training_step(x, y)
        flag = True
        for p in net.params:
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
        loss = test_nets[1]
        # TODO e.g. loss is not CrossEntropy when output has one dimension
        #       e.g. softmax has not been applied with CrossEntropy loss
        #       (includes it)
        assert loss is None

    def check_multi_output(self, net):
        if 'MultiMLPNet' in net.__class__.__name__ or \
           'MLPHeterogenousTails' in net.__class__.__name__ or \
           ((('RecurrentNet' in net.__class__.__name__) or ('ConvNet' in net.__class__.__name__)) and len(net.out_dim) > 1):
            return True
        else:
            return False

    def test_output(self, test_nets):
        ''' Checks that the output of the net is not zero or nan '''
        net = test_nets[0]
        dummy_input = self.init_dummy_input(net)
        dummy_output = self.init_dummy_output(net)
        out = net(dummy_input)
        flag = True
        if self.check_multi_output(net):
            zero_test = sum([torch.sum(torch.abs(x.data)) for x in out])
            nan_test = np.isnan(sum([torch.sum(x.data) for x in out]))
        else:
            zero_test = torch.sum(torch.abs(out.data))
            nan_test = np.isnan(torch.sum(out.data))
        if zero_test < SMALL_NUM:
            print("FAIL")
            print(out)
            flag = False
        if nan_test:
            print("FAIL")
            print(out)
            flag = False
        if flag:
            print("PASS")
        assert flag is True
