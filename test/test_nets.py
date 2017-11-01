from test.nntest import TestNet
from unity_lab.agent.nets.feedforward import MLPNet
from torch.autograd import Variable

in_dim = 10
out_dim = 2
hid = [5, 3]
net = MLPNet(in_dim, hid, out_dim)
test = TestNet()
print(net)
# TODO: Fix, not passing yet
test.check_trainable(net)
test.check_fixed(net)
