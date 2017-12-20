'''
Functions used by more than one algorithm
TODO refactor properly later
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


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


def multi_act_with_epsilon_greedy(bodies, state, net, epsilon):
    '''Multi-body action on a single-pass from net. Uses epsilon-greedy but in a batch manner.'''
    flat_body_states = np.concatenate(state)
    # print(f'epsilon {epsilon}')
    if epsilon > np.random.rand():
        # print('random action')
        action = np.random.randint(a_dim, size=len(bodies))
    else:
        # print('net action')
        torch_state = Variable(torch.from_numpy(flat_body_states).float())
        out = net.wrap_eval(torch_state)
        action = []
        start_idx = 0
        for eb_idx, body in enumerate(bodies):
            end_idx = start_idx + body.action_dim
            body_action = int(torch.max(out[start_idx: end_idx], dim=0)[1][0])
            action.append(body_action)
            start_idx = end_idx
    return action


def act_with_boltzmann(body, state, net, tau):
    torch_state = Variable(torch.from_numpy(state).float())
    out = net.wrap_eval(torch_state)
    out_with_temp = torch.div(out, tau)
    probs = F.softmax(out_with_temp).data.numpy()
    action = np.random.choice(list(range(body.action_dim)), p=probs)
    # print('Probs: {}, action: {}'.format(probs, action))
    return action


def multi_act_with_boltzmann(bodies, state, net, tau):
    flat_body_states = np.concatenate(state)
    torch_state = Variable(torch.from_numpy(flat_body_states).float())
    out = net.wrap_eval(torch_state)
    out_with_temp = torch.div(out, tau)
    action = []
    start_idx = 0
    # print("Acting...")
    for eb_idx, body in enumerate(bodies):
        end_idx = start_idx + body.action_dim
        probs = F.softmax(out_with_temp[start_idx: end_idx]).data.numpy()
        body_action = np.random.choice(list(range(body.action_dim)), p=probs)
        # print("Start idx: {}, end: idx: {}, action: {}, dims: {}".format(
        #     start_idx,
        #     end_idx,
        #     body_action,
        #     out_with_temp[start_idx: end_idx].size()
        # ))
        action.append(body_action)
        start_idx = end_idx
    return action

# Adapted from  https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
def act_with_softmax(body, state, net):
    torch_state = Variable(torch.from_numpy(state).float())
    out = net.wrap_eval(torch_state)
    probs = F.softmax(out)
    m = Categorical(probs)
    action = m.sample()
    net.saved_log_probs.append(m.log_prob(action))
    return action.data[0]

def act_with_gaussian(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass

act_fns = {
    'epsilon_greedy': act_with_epsilon_greedy,
    'multi_epsilon_greedy': multi_act_with_epsilon_greedy,
    'boltzmann': act_with_boltzmann,
    'multi_boltzmann': multi_act_with_boltzmann,
    'gaussian': act_with_gaussian
}
