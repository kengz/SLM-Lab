'''
The nets module
Contains classes of neural network architectures
'''

from slm_lab.agent.net.convnet import ConvNet
from slm_lab.agent.net.feedforward import MLPNet

nets = {'conv' : ConvNet,
        'mlp'  : MLPNet}
