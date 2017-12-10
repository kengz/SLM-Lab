'''
Functions used by more than one algorithm
TODO refactor properly later
'''
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F


def act_with_epsilon_greedy(body, state, net, epsilon):
    '''
    Single body action with probability epsilon to select a random action,
    otherwise select the action associated with the largest q value
    '''
    # print(f'epsilon {epsilon}')
    if epsilon > np.random.rand():
        # print('random action')
        action = np.random.randint(body.action_dim)
    else:
        # print('net action')
        torch_state = Variable(torch.from_numpy(state).float())
        out = net.wrap_eval(torch_state)
        action = int(torch.max(out, dim=0)[1][0])
    return action


def multi_act_with_epsilon_greedy(net, state, epsilon):
    '''
    Multi-body action at a batch
    With probability epsilon select a random action,
    otherwise select the action associated with the
    largest q value
    '''
    return action


def act_with_boltzmann(body, state, net, tau):
    torch_state = Variable(torch.from_numpy(state).float())
    out = net.wrap_eval(torch_state)
    out_with_temp = torch.div(out, tau)
    probs = F.softmax(out_with_temp).data.numpy()
    action = np.random.choice(list(range(body.action_dim)), p=probs)
    # print('Probs: {}, action: {}'.format(probs, action))
    return action


def act_with_gaussian(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass


def update_epsilon_greedy(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass


def update_boltzmann(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass


def update_gaussian(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass


act_fns = {
    'epsilon_greedy': act_with_epsilon_greedy,
    'multi_epsilon_greedy': multi_act_with_epsilon_greedy,
    'boltzmann': act_with_boltzmann,
    'gaussian': act_with_gaussian
}

act_update_fns = {
    'epsilon_greedy': update_epsilon_greedy,
    'boltzmann': update_boltzmann,
    'gaussian': update_gaussian
}
