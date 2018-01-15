'''
The algorithm module
Contains implementations of reinforcement learning algorithms.
Uses the nets module to build neural networks as the relevant function approximators
'''

# expose all the classes
from .dqn import *
from .reinforce import *
from .actor_critic import *
from .random import *
