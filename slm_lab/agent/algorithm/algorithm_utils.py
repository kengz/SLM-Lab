import torch
import random

'''Functions used by more than one algorithm'''

act_fns = {'epsilon_greedy' : act_with_epsilon_greedy,
           'boltzmann'      : act_with_boltzmann,
           'gaussian'       : act_with_gaussian}

update_fns = {'epsilon_greedy' : update_epsilon_greedy,
              'boltzmann'      : update_boltzmann,
              'gaussian'       : update_gaussian}

def act_with_epsilon_greedy(net, state, epsilon):
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

def act_with_boltzmann(net, state, tau):
    # TODO implement act_with_boltzmann
    pass

def act_with_gaussian(net, state, stddev):
    # TODO implement act_with_gaussian
    pass
