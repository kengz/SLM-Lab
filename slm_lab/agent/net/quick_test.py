from convnet import ConvNet
from feedforward import MultiMLPNet
from torch.autograd import Variable
import torch
import torch.nn as nn

dummy_input = Variable(torch.ones((2, 3, 32, 32)))
net = ConvNet((3, 32, 32),
              ([[3, 16, (5, 5), 2, 0, 1],
               [16, 32, (5, 5), 2, 0, 1]],
              [100, 50]),
              10,
              optim_param={'name': 'Adam'},
              loss_param={'name': 'smooth_l1_loss'},
              clamp_grad=False,
              batch_norm=False)

print(net)
out = net(dummy_input)

net = ConvNet((3, 32, 32),
              ([[3, 16, (5, 5), 2, 0, 1],
               [16, 32, (5, 5), 2, 0, 1]],
              [100, 50]),
              10,
              optim_param={'name': 'Adam'},
              loss_param={'name': 'smooth_l1_loss'},
              clamp_grad=False,
              batch_norm=True)

print(net)
out = net(dummy_input)

net = ConvNet((3, 32, 32),
              ([[3, 16, (7, 7), 1, 0, 1],
               [16, 32, (5, 5), 1, 0, 1],
               [32, 64, (3, 3), 1, 0, 1]],
              [100, 50]),
              10,
              optim_param={'name': 'Adam'},
              loss_param={'name': 'smooth_l1_loss'},
              clamp_grad=False,
              batch_norm=False)

print(net)
out = net(dummy_input)

net = ConvNet((3, 32, 32),
              ([[3, 16, (7, 7), 1, 0, 1],
               [16, 32, (5, 5), 1, 0, 1],
               [32, 64, (3, 3), 1, 0, 1]],
              [100, 50]),
              10,
              optim_param={'name': 'Adam'},
              loss_param={'name': 'smooth_l1_loss'},
              clamp_grad=False,
              batch_norm=True)

print(net)
out = net(dummy_input)

dummy_input = [Variable(torch.ones((2, 10))), Variable(torch.ones((2, 5)))]

net = MultiMLPNet([[10, 8], [5, 4]],
                  [10, 8],
                  [[2], [3]],
                  'sigmoid',
                  optim_param={'name': 'Adam'},
                  loss_param={'name': 'mse_loss'})

net.print_nets()
out = net(dummy_input)
print(out)
