'''
Functions used by more than one algorithm
'''

from copy import deepcopy
from slm_lab.lib import logger, util
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.distributions import Categorical


def act_with_epsilon_greedy(body, state, net, epsilon):
    '''
    Single body action with probability epsilon to select a random action,
    otherwise select the action associated with the largest q value
    '''
    if epsilon > np.random.rand():
        action = np.random.randint(body.action_dim)
    else:
        torch_state = Variable(torch.from_numpy(state).float())
        out = net.wrap_eval(torch_state)
        action = int(torch.max(out, dim=0)[1][0])
    return action


def multi_act_with_epsilon_greedy(flat_nonan_body_a, state_a, net, epsilon):
    '''Multi-body flat_nonan_action_a on a single-pass from net. Uses epsilon-greedy but in a batch manner.'''
    flat_nonan_state_a = util.flatten_nonan(state_a)
    cat_state_a = np.concatenate(flat_nonan_state_a)
    if epsilon > np.random.rand():
        flat_nonan_action_a = np.random.randint(
            a_dim, size=len(flat_nonan_body_a))
    else:
        torch_state = Variable(torch.from_numpy(cat_state_a).float())
        out = net.wrap_eval(torch_state)
        flat_nonan_action_a = []
        start_idx = 0
        for body in flat_nonan_body_a:
            end_idx = start_idx + body.action_dim
            action = int(torch.max(out[start_idx: end_idx], dim=0)[1][0])
            flat_nonan_action_a.append(action)
            start_idx = end_idx
            logger.debug(f'''
            body: {body.aeb}, net idx: {start_idx}-{end_idx}
            action: {action}''')
    return flat_nonan_action_a


def act_with_boltzmann(body, state, net, tau):
    torch_state = Variable(torch.from_numpy(state).float())
    out = net.wrap_eval(torch_state)
    out_with_temp = torch.div(out, tau)
    probs = F.softmax(out_with_temp).data.numpy()
    action = np.random.choice(list(range(body.action_dim)), p=probs)
    logger.debug('prob: {}, action: {}'.format(probs, action))
    return action


def multi_act_with_boltzmann(flat_nonan_body_a, state_a, net, tau):
    flat_nonan_state_a = util.flatten_nonan(state_a)
    cat_state_a = np.concatenate(flat_nonan_state_a)
    torch_state = Variable(torch.from_numpy(cat_state_a).float())
    out = net.wrap_eval(torch_state)
    out_with_temp = torch.div(out, tau)
    flat_nonan_action_a = []
    start_idx = 0
    for body in flat_nonan_body_a:
        end_idx = start_idx + body.action_dim
        probs = F.softmax(out_with_temp[start_idx: end_idx]).data.numpy()
        action = np.random.choice(list(range(body.action_dim)), p=probs)
        logger.debug(f'''
        body: {body.aeb}, net idx: {start_idx}-{end_idx}
        probs: {probs}, action: {action}''')
        flat_nonan_action_a.append(action)
        start_idx = end_idx
    return flat_nonan_action_a


# Adapted from  https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
def act_with_softmax(agent, body, state, net):
    torch_state = Variable(torch.from_numpy(state).float())
    out = net(torch_state)
    probs = F.softmax(out)
    action = np.random.choice(list(range(body.action_dim)), p=probs.data.numpy())
    agent.algorithm.saved_log_probs.append(torch.log(probs))
    # print(type(probs))
    # print("Action: {}".format(action))
    return action
    # m = Categorical(probs)
    # action = m.sample()
    # net.saved_log_probs.append(m.log_prob(action))
    # return action.data[0]

  
def act_with_gaussian(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass


def update_linear_decay(cls, clock):
    t = clock.get('total_t')
    epi = clock.get('e')
    rise = cls.explore_var_end - cls.explore_var_start
    slope = rise / float(cls.explore_anneal_epi)
    cls.explore_var = max(
        slope * (epi - 1) + cls.explore_var_start, cls.explore_var_end)
    logger.debug(f'explore_var: {cls.explore_var}')
    return cls.explore_var


def update_gaussian(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass


act_fns = {
    'epsilon_greedy': act_with_epsilon_greedy,
    'multi_epsilon_greedy': multi_act_with_epsilon_greedy,
    'boltzmann': act_with_boltzmann,
    'multi_boltzmann': multi_act_with_boltzmann,
    'gaussian': act_with_gaussian,
    'softmax': act_with_softmax
}


act_update_fns = {
    'linear_decay': update_linear_decay,
    'gaussian': update_gaussian
}
