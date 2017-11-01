import torch
from torch.autograd import Variable
from test.nn_test import TestNet
from unity_lab.agent.net.feedforward import MLPNet

in_dim = 10
out_dim = 2
hid = [5, 3]
net = MLPNet(in_dim, hid, out_dim)

test = TestNet()
print(net)
_ = test.check_params_not_zero(net)
_ = test.check_output(net)
_ = test.check_trainable(net)
_ = test.check_fixed(net)

dummy_input = Variable(torch.ones((2, net.in_dim)))
dummy_output = Variable(torch.zeros((2, net.out_dim)))
_ = test.check_gradient_size(net, dummy_input, dummy_output)
