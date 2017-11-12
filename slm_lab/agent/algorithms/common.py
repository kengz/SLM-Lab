import torch
import random

'''Functions used by more than one algorithm'''

def select_action_epsilon_greedy(net, state, epsilon):
    '''
    With probability episilon select a random action,
    otherwise select the action associated with the
    largest q value
    '''
    i = random.random()
    a_dim = net.out_dim
    action = None
    if i < epsilon:
        action = random.randint(0, a_dim)
    else:
        out = net.eval(state)
        _, action = torch.max(out)
    one_hot_a = torch.zeros(1, a_dim)
    one_hot_a[0][action] = 1
    return one_hot_a

def select_action_boltzmann(net, state, tau):
    # TODO: implement select_action_boltzmann
    pass
