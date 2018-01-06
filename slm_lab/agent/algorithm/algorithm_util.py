'''
Functions used by more than one algorithm
'''
from slm_lab.lib import logger, util
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F


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
    flat_nonan_action_a = []
    start_idx = 0
    for body, e in zip(flat_nonan_body_a, epsilon):
        logger.debug(f'body: {body.aeb}, epsilon: {e}')
        end_idx = start_idx + body.action_dim
        if e > np.random.rand():
            logger.debug(f'Random action')
            action = np.random.randint(body.action_dim)
        else:
            logger.debug(f'Greedy action')
            cat_state_a = cat_state_a.astype('float')
            torch_state = Variable(torch.from_numpy(cat_state_a).float())
            out = net.wrap_eval(torch_state)
            action = int(torch.max(out[start_idx: end_idx], dim=0)[1][0])
        flat_nonan_action_a.append(action)
        start_idx = end_idx
        logger.debug(f'''
        body: {body.aeb}, net idx: {start_idx}-{end_idx}
        action: {action}''')
    return flat_nonan_action_a


def multi_head_act_with_epsilon_greedy(flat_nonan_body_a, state_a, net, epsilon):
    '''Multi-headed body flat_nonan_action_a on a single-pass from net. Uses epsilon-greedy but in a batch manner.'''
    flat_nonan_state_a = util.flatten_nonan(state_a)
    flat_nonan_action_a = []
    torch_states = []
    for state in flat_nonan_state_a:
        state = state.astype('float')
        torch_states.append(
            Variable(torch.from_numpy(state).float().unsqueeze_(dim=0)))
    outs = net.wrap_eval(torch_states)
    for body, e, output in zip(flat_nonan_body_a, epsilon, outs):
        logger.debug(f'body: {body.aeb}, epsilon: {e}')
        if e > np.random.rand():
            logger.debug(f'Random action')
            action = np.random.randint(body.action_dim)
        else:
            logger.debug(f'Greedy action')
            action = torch.max(output, dim=1)[1][0]
        flat_nonan_action_a.append(action)
        logger.debug(f'epsilon: {e}, outputs: {output}, action: {action}')
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
    cat_state_a = np.concatenate(flat_nonan_state_a).astype(float)
    torch_state = Variable(torch.from_numpy(cat_state_a).float())
    out = net.wrap_eval(torch_state)
    flat_nonan_action_a = []
    start_idx = 0
    logger.debug(f'taus: {tau}')
    for body, t in zip(flat_nonan_body_a, tau):
        end_idx = start_idx + body.action_dim
        out_with_temp = torch.div(out[start_idx: end_idx], t)
        logger.debug(f'''
        tau: {t}, out: {out},
        out select: {out[start_idx: end_idx]},
        out with temp: {out_with_temp}''')
        probs = F.softmax(out_with_temp).data.numpy()
        action = np.random.choice(list(range(body.action_dim)), p=probs)
        logger.debug(f'''
        body: {body.aeb}, net idx: {start_idx}-{end_idx}
        probs: {probs}, action: {action}''')
        flat_nonan_action_a.append(action)
        start_idx = end_idx
    return flat_nonan_action_a


def multi_head_act_with_boltzmann(flat_nonan_body_a, state_a, net, tau):
    flat_nonan_state_a = util.flatten_nonan(state_a)
    torch_states = []
    for state in flat_nonan_state_a:
        state = state.astype('float')
        torch_states.append(
            Variable(torch.from_numpy(state).float().unsqueeze_(dim=0)))
    outs = net.wrap_eval(torch_states)
    out_with_temp = [torch.div(x, t) for x, t in zip(outs, tau)]
    logger.debug(f'taus: {tau}, outs: {outs}, out_with_temp: {out_with_temp}')
    flat_nonan_action_a = []
    for body, output in zip(flat_nonan_body_a, out_with_temp):
        probs = F.softmax(output).data.numpy()[0]
        action = np.random.choice(list(range(body.action_dim)), p=probs)
        logger.debug(f'''
        body: {body.aeb}, output: {output},
        probs: {probs}, action: {action}''')
        flat_nonan_action_a.append(action)
    return flat_nonan_action_a


def act_with_gaussian(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass


def update_linear_decay(cls, clock):
    epi = clock.get('epi')
    rise = cls.explore_var_end - cls.explore_var_start
    slope = rise / float(cls.explore_anneal_epi)
    cls.explore_var = max(
        slope * (epi - 1) + cls.explore_var_start, cls.explore_var_end)
    logger.debug(f'explore_var: {cls.explore_var}')
    return cls.explore_var


def update_multi_linear_decay(cls, flat_nonan_body_a):
    explore_var = []
    for body, e in zip(flat_nonan_body_a, cls.explore_var):
        epi = body.env.clock.get('epi')
        rise = cls.explore_var_end - cls.explore_var_start
        slope = rise / float(cls.explore_anneal_epi)
        e = max(slope * (epi - 1) + cls.explore_var_start, cls.explore_var_end)
        explore_var.append(e)
    cls.explore_var = explore_var
    logger.debug(f'explore_var: {cls.explore_var}')
    # TODO Handle returning all explore vars
    return cls.explore_var[0]


def update_gaussian(body, state, net, stddev):
    # TODO implement act_with_gaussian
    pass


act_fns = {
    'epsilon_greedy': act_with_epsilon_greedy,
    'multi_epsilon_greedy': multi_act_with_epsilon_greedy,
    'multi_head_epsilon_greedy': multi_head_act_with_epsilon_greedy,
    'boltzmann': act_with_boltzmann,
    'multi_boltzmann': multi_act_with_boltzmann,
    'multi_head_boltzmann': multi_head_act_with_boltzmann,
    'gaussian': act_with_gaussian
}

act_update_fns = {
    'linear_decay': update_linear_decay,
    'multi_linear_decay': update_multi_linear_decay,
    'gaussian': update_gaussian
}
