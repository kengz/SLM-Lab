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
from torch.distributions import Categorical, Normal


def create_torch_state(state, state_buffer, recurrent=False, length=0):
    if recurrent:
        '''Create sequence of inputs for recurrent net'''
        logger.debug3(f'length of state buffer: {length}')
        if len(state_buffer) < length:
            PAD = np.zeros_like(state)
            while len(state_buffer) < length:
                state_buffer.insert(0, PAD)
        state_buffer = np.asarray(state_buffer)
        '''Hack to fix buffer not storing the very first state in an epi'''
        if np.sum(state_buffer) == 0:
            state_buffer[-1] = state
        torch_state = Variable(torch.from_numpy(state_buffer).float())
        torch_state.unsqueeze_(dim=0)
    else:
        torch_state = Variable(torch.from_numpy(state).float())
    logger.debug2(f'State size: {torch_state.size()}')
    logger.debug3(f'Original state: {state}')
    logger.debug3(f'State: {torch_state}')
    return torch_state


def act_with_epsilon_greedy(body, state, net, epsilon):
    '''
    Single body action with probability epsilon to select a random action,
    otherwise select the action associated with the largest q value
    '''
    if epsilon > np.random.rand():
        action = np.random.randint(body.action_dim)
    else:
        recurrent = body.agent.len_state_buffer > 0
        logger.debug2(f'Length state buffer: {body.agent.len_state_buffer}')
        torch_state = create_torch_state(state, body.state_buffer, recurrent, body.agent.len_state_buffer)
        out = net.wrap_eval(torch_state).squeeze_(dim=0)
        action = int(torch.max(out, dim=0)[1][0])
        logger.debug2(f'Outs {out} Action {action}')
    return action


def multi_act_with_epsilon_greedy(nanflat_body_a, state_a, net, nanflat_epsilon_a):
    '''Multi-body nanflat_action_a on a single-pass from net. Uses epsilon-greedy but in a batch manner.'''
    nanflat_state_a = util.nanflatten(state_a)
    cat_state_a = np.concatenate(nanflat_state_a)
    nanflat_action_a = []
    start_idx = 0
    for body, e in zip(nanflat_body_a, nanflat_epsilon_a):
        logger.debug2(f'body: {body.aeb}, epsilon: {e}')
        end_idx = start_idx + body.action_dim
        if e > np.random.rand():
            logger.debug2(f'Random action')
            action = np.random.randint(body.action_dim)
        else:
            logger.debug2(f'Greedy action')
            cat_state_a = cat_state_a.astype('float')
            torch_state = Variable(torch.from_numpy(cat_state_a).float())
            out = net.wrap_eval(torch_state)
            action = int(torch.max(out[start_idx: end_idx], dim=0)[1][0])
        nanflat_action_a.append(action)
        start_idx = end_idx
        logger.debug2(f'''
        body: {body.aeb}, net idx: {start_idx}-{end_idx}
        action: {action}''')
    return nanflat_action_a


def multi_head_act_with_epsilon_greedy(nanflat_body_a, state_a, net, nanflat_epsilon_a):
    '''Multi-headed body nanflat_action_a on a single-pass from net. Uses epsilon-greedy but in a batch manner.'''
    nanflat_state_a = util.nanflatten(state_a)
    nanflat_action_a = []
    torch_states = []
    for state in nanflat_state_a:
        state = state.astype('float')
        torch_states.append(
            Variable(torch.from_numpy(state).float().unsqueeze_(dim=0)))
    outs = net.wrap_eval(torch_states)
    for body, e, output in zip(nanflat_body_a, nanflat_epsilon_a, outs):
        logger.debug2(f'body: {body.aeb}, epsilon: {e}')
        if e > np.random.rand():
            logger.debug2(f'Random action')
            action = np.random.randint(body.action_dim)
        else:
            logger.debug2(f'Greedy action')
            action = torch.max(output, dim=1)[1][0]
        nanflat_action_a.append(action)
        logger.debug2(f'epsilon: {e}, outputs: {output}, action: {action}')
    return nanflat_action_a


def act_with_boltzmann(body, state, net, tau):
    recurrent = body.agent.len_state_buffer > 0
    logger.debug2(f'Length state buffer: {body.agent.len_state_buffer}')
    torch_state = create_torch_state(state, body.state_buffer, recurrent, body.agent.len_state_buffer)
    out = net.wrap_eval(torch_state)
    out_with_temp = torch.div(out, tau).squeeze_(dim=0)
    probs = F.softmax(Variable(out_with_temp), dim=0).data.numpy()
    action = np.random.choice(list(range(body.action_dim)), p=probs)
    logger.debug2('out with temp: {}, prob: {}, action: {}'.format(out_with_temp, probs, action))
    return action


def multi_act_with_boltzmann(nanflat_body_a, state_a, net, nanflat_tau_a):
    nanflat_state_a = util.nanflatten(state_a)
    cat_state_a = np.concatenate(nanflat_state_a).astype(float)
    torch_state = Variable(torch.from_numpy(cat_state_a).float())
    out = net.wrap_eval(torch_state)
    nanflat_action_a = []
    start_idx = 0
    logger.debug2(f'taus: {nanflat_tau_a}')
    for body, tau in zip(nanflat_body_a, nanflat_tau_a):
        end_idx = start_idx + body.action_dim
        out_with_temp = torch.div(out[start_idx: end_idx], tau)
        logger.debug3(f'''
        tau: {tau}, out: {out},
        out select: {out[start_idx: end_idx]},
        out with temp: {out_with_temp}''')
        probs = F.softmax(Variable(out_with_temp), dim=0).data.numpy()
        action = np.random.choice(list(range(body.action_dim)), p=probs)
        logger.debug3(f'''
        body: {body.aeb}, net idx: {start_idx}-{end_idx}
        probs: {probs}, action: {action}''')
        nanflat_action_a.append(action)
        start_idx = end_idx
    return nanflat_action_a


def multi_head_act_with_boltzmann(nanflat_body_a, state_a, net, nanflat_tau_a):
    nanflat_state_a = util.nanflatten(state_a)
    torch_states = []
    for state in nanflat_state_a:
        state = state.astype('float')
        torch_states.append(
            Variable(torch.from_numpy(state).float().unsqueeze_(dim=0)))
    outs = net.wrap_eval(torch_states)
    out_with_temp = [torch.div(x, t) for x, t in zip(outs, nanflat_tau_a)]
    logger.debug2(
        f'taus: {nanflat_tau_a}, outs: {outs}, out_with_temp: {out_with_temp}')
    nanflat_action_a = []
    for body, output in zip(nanflat_body_a, out_with_temp):
        probs = F.softmax(Variable(output), dim=1).data.numpy()[0]
        action = np.random.choice(list(range(body.action_dim)), p=probs)
        logger.debug3(f'''
        body: {body.aeb}, output: {output},
        probs: {probs}, action: {action}''')
        nanflat_action_a.append(action)
    return nanflat_action_a


# Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
def act_with_softmax(algo, state, body):
    '''Assumes actor network outputs one variable; the logits of a categorical probability distribution over the actions'''
    recurrent = algo.agent.len_state_buffer > 0
    torch_state = create_torch_state(state, body.state_buffer, recurrent, algo.agent.len_state_buffer)
    out = algo.get_actor_output(torch_state, evaluate=False)
    if type(out) is list:
        out = out[0]
    out.squeeze_(dim=0)
    probs = F.softmax(out, dim=0)
    m = Categorical(probs)
    action = m.sample()
    logger.debug2(f'Network output: {out.data}')
    logger.debug2(f'Probability of actions: {probs.data}')
    logger.debug(
        f'Action: {action.data[0]}, log prob: {m.log_prob(action).data[0]}')
    algo.saved_log_probs.append(m.log_prob(action))
    # Calculate entropy of the distribution
    H = - torch.sum(torch.mul(probs, torch.log(probs)))
    if np.isnan(H.data.numpy()):
        logger.debug(f'NaN entropy, setting to 0')
        H = Variable(torch.zeros(1))
    algo.entropy.append(H)
    return action.data[0]


# Denny Britz has a very helpful implementation of an Actor Critic algorithm. This function is adapted from his approach. I highly recommend looking at his full implementation available here https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb
def act_with_gaussian(algo, state, body):
    '''Assumes net outputs two variables; the mean and std dev of a normal distribution'''
    recurrent = algo.agent.len_state_buffer > 0
    torch_state = create_torch_state(state, body.state_buffer, recurrent, algo.agent.len_state_buffer)
    [mu, sigma] = algo.get_actor_output(torch_state, evaluate=False)
    sigma = F.softplus(sigma) + 1e-5  # Ensures sigma > 0
    m = Normal(mu, sigma)
    action = m.sample()
    action = torch.clamp(action, -algo.continuous_action_clip, algo.continuous_action_clip)
    logger.debug2(
        f'Action: {action.data[0]}, log prob: {m.log_prob(action).data[0]}')
    algo.saved_log_probs.append(m.log_prob(action))
    # Calculate entropy of the distribution
    H = 0.5 * torch.log(2.0 * np.pi * np.e * sigma * sigma)
    if np.isnan(H.data.numpy()):
        logger.debug(f'NaN entropy, setting to 0')
        H = Variable(torch.zeros(1))
    algo.entropy.append(H)
    return action.data


def act_with_multivariate_gaussian(algo, state, body):
    '''Assumes net outputs two tensors which contain the mean and std dev of a multivariate normal distribution'''
    raise NotImplementedError
    return np.nan


def update_linear_decay(cls, space_clock):
    epi = space_clock.get('epi')
    rise = cls.explore_var_end - cls.explore_var_start
    slope = rise / float(cls.explore_anneal_epi)
    explore_var = max(
        slope * (epi - 1) + cls.explore_var_start, cls.explore_var_end)
    cls.nanflat_explore_var_a = [explore_var] * cls.agent.body_num
    logger.debug3(
        f'nanflat_explore_var_a: {cls.nanflat_explore_var_a[0]}')
    return cls.nanflat_explore_var_a


def update_multi_linear_decay(cls, _space_clock):
    nanflat_explore_var_a = []
    for body in cls.agent.nanflat_body_a:
        # use body-clock instead of space clock
        epi = body.env.clock.get('epi')
        rise = cls.explore_var_end - cls.explore_var_start
        slope = rise / float(cls.explore_anneal_epi)
        explore_var = max(
            slope * (epi - 1) + cls.explore_var_start, cls.explore_var_end)
        nanflat_explore_var_a.append(explore_var)
    cls.nanflat_explore_var_a = nanflat_explore_var_a
    logger.debug3(f'nanflat_explore_var_a: {cls.nanflat_explore_var_a}')
    return cls.nanflat_explore_var_a


def decay_learning_rate(algo, nets):
    '''
    Decay learning rate for each net by the decay method update_lr() defined in them.
    In the future, might add more flexible lr adjustment, like boosting and decaying on need.
    '''
    space_clock = util.s_get(algo, 'aeb_space.clock')
    t = space_clock.get('total_t')
    if algo.decay_lr and t > algo.decay_lr_min_timestep:
        if t % algo.decay_lr_frequency == 0:
            for net in nets:
                net.update_lr()


act_fns = {
    'epsilon_greedy': act_with_epsilon_greedy,
    'multi_epsilon_greedy': multi_act_with_epsilon_greedy,
    'multi_head_epsilon_greedy': multi_head_act_with_epsilon_greedy,
    'boltzmann': act_with_boltzmann,
    'multi_boltzmann': multi_act_with_boltzmann,
    'multi_head_boltzmann': multi_head_act_with_boltzmann,
    'gaussian': act_with_gaussian,
    'softmax': act_with_softmax
}


act_update_fns = {
    'linear_decay': update_linear_decay,
    'multi_linear_decay': update_multi_linear_decay,
}
