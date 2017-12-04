import numpy as np
import torch
from torch.autograd import Variable

'''Functions used by more than one algorithm'''


def act_with_epsilon_greedy(net, state, epsilon):
    '''
    With probability episilon select a random action,
    otherwise select the action associated with the
    largest q value
    '''
    # TODO discrete int
    a_dim = net.out_dim
    print(f'epsilon {epsilon}')
    if epsilon > np.random.rand():
        print('random action')
        action = np.random.randint(a_dim)
    else:
        print('net action')
        torch_state = Variable(torch.from_numpy(state).float())
        out = net.wrap_eval(torch_state)
        action = int(torch.max(out, dim=0)[1][0])
    return action


def act_with_boltzmann(net, state, tau):
    # TODO implement act_with_boltzmann
    pass


def act_with_gaussian(net, state, stddev):
    # TODO implement act_with_gaussian
    pass


def update_epsilon_greedy(net, state, stddev):
    # TODO implement act_with_gaussian
    pass


def update_boltzmann(net, state, stddev):
    # TODO implement act_with_gaussian
    pass


def update_gaussian(net, state, stddev):
    # TODO implement act_with_gaussian
    pass


act_fns = {'epsilon_greedy': act_with_epsilon_greedy,
           'boltzmann': act_with_boltzmann,
           'gaussian': act_with_gaussian}

update_fns = {'epsilon_greedy': update_epsilon_greedy,
              'boltzmann': update_boltzmann,
              'gaussian': update_gaussian}
