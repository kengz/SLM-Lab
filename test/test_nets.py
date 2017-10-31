from test.nntest import TestNet
from unity_lab.agent.nets.feedforward import MLPNet
from torch.autograd import Variable

in_dim = 100
out_dim = 10
hid = [50, 25]
net = MLPNet(in_dim, hid, out_dim)
test = TestNet()
net
