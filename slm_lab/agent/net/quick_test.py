import torch.nn as nn
from torch import optim
from convnet import ConvNet

net = ConvNet(
        (3, 32, 32),
        [[3, 36, (5, 5), 1, 0, (2, 2)],[36, 128, (5, 5), 1, 0, (3, 3)]],
        [100],
        10,
        optim.Adam,
        nn.SmoothL1Loss,
        False,
        False)

print(net)

net = ConvNet(
        (3, 32, 32),
        [[3, 36, (5, 5), 1, 0, (2, 2)],[36, 128, (5, 5), 1, 0, (2, 2)]],
        [100],
        10,
        optim.Adam,
        nn.SmoothL1Loss,
        False,
        True)

print(net)
