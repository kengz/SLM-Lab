'''
The nets module
Contains classes of neural network architectures
'''

from slm.agent.net.convnet import ConvNet
from slm.agent.net.feedforward import MLPNet

nets = {'conv' : ConvNet,
        'mlp'  : MLPNet}
