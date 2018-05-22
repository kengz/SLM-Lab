'''
Functions used by more than one algorithm
'''
from copy import deepcopy
from slm_lab.lib import logger, util
from torch.distributions import Categorical, Normal
import numpy as np
import torch
import torch.nn.functional as F

logger = logger.get_logger(__name__)


def set_flags(body):
    atari = False
    flatten = False
    if hasattr(body.memory, 'atari'):
        atari = body.memory.atari
    if hasattr(body.memory, 'stacked'):
        flatten = body.memory.stacked
    return atari, flatten


def first_state_check(state_buffer, state, atari):
    '''Hack to fix buffer not storing the very first state in an epi'''
    if np.sum(state_buffer) == 0:
        if atari:
            state_buffer[-1] = util.transform_image(state)
        else:
            state_buffer[-1] = state
    return state_buffer


def create_torch_state(state, state_buf, gpu, state_seq=False, length=0, atari=False, flatten=False):
    if state_seq:
        '''Create sequence of inputs for nets that take sequences of states as input'''
        state_buffer = deepcopy(state_buf)  # Copy so as not to mutate running state buffer
        logger.debug3(f'length of state buffer: {length}')
        if len(state_buffer) < length:
            pad = np.zeros_like(state)
            while len(state_buffer) < length:
                state_buffer.insert(0, pad)
        # Preprocess the state if necessary
        if atari:
            logger.debug3(f'Preprocesssing the atari states')
            for _, s in enumerate(state_buffer):
                state_buffer[_] = util.transform_image(s)
        state_buffer = np.asarray(state_buffer)
        logger.debug3(f'state buffer: {state_buffer.size}')
        '''Hack to fix buffer not storing the very first state in an epi'''
        state_buffer = first_state_check(state_buffer, state, atari)
        if atari:
            state_buffer = np.transpose(state_buffer, (1, 2, 0))
        torch_state = torch.from_numpy(state_buffer).float()
        torch_state.unsqueeze_(dim=0)
        if flatten:
            torch_state = torch_state.view(-1)
    else:
        torch_state = torch.from_numpy(state).float()

    # Optionally convert to cuda
    if torch.cuda.is_available() and gpu:
        torch_state = torch_state.cuda()

    logger.debug3(f'State size: {torch_state.size()}')
    logger.debug3(f'Original state: {state}')
    logger.debug3(f'State: {torch_state}')

    return torch_state


def act_with_epsilon_greedy(body, state, net, epsilon, gpu):
    '''
    Single body action with probability epsilon to select a random action,
    otherwise select the action associated with the largest q value
    '''
    if epsilon > np.random.rand():
        action = np.random.randint(body.action_dim)
    else:
        state_seq = body.memory.state_buffer.maxlen > 0
        logger.debug(f'Length state buffer: {body.memory.state_buffer.maxlen}')
        atari, flatten = set_flags(body)
        torch_state = create_torch_state(state, body.memory.state_buffer, gpu, state_seq, body.memory.state_buffer.maxlen, atari, flatten)
        out = net.wrap_eval(torch_state).squeeze_(dim=0)
        action = torch.max(out, dim=0)[1].item()
        logger.debug2(f'Outs {out} Action {action}')
    return action


def multi_act_with_epsilon_greedy(nanflat_body_a, state_a, net, nanflat_epsilon_a, gpu):
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
            torch_state = torch.from_numpy(cat_state_a).float()
            if torch.cuda.is_available() and gpu:
                torch_state = torch_state.cuda()
            out = net.wrap_eval(torch_state)
            action = torch.max(out[start_idx: end_idx], dim=0)[1].item()
        nanflat_action_a.append(action)
        start_idx = end_idx
        logger.debug2(f'''
        body: {body.aeb}, net idx: {start_idx}-{end_idx}
        action: {action}''')
    return nanflat_action_a


def multi_head_act_with_epsilon_greedy(nanflat_body_a, state_a, net, nanflat_epsilon_a, gpu):
    '''Multi-headed body nanflat_action_a on a single-pass from net. Uses epsilon-greedy but in a batch manner.'''
    nanflat_state_a = util.nanflatten(state_a)
    nanflat_action_a = []
    torch_states = []
    for state in nanflat_state_a:
        state = state.astype('float')
        torch_states.append(
            torch.from_numpy(state).float().unsqueeze_(dim=0))
    if torch.cuda.is_available() and gpu:
        for idx, torch_state in enumerate(torch_states):
            torch_states[idx] = torch_state.cuda()
    for idx, torch_state in enumerate(torch_states):
        torch_states[idx] = torch_state
    outs = net.wrap_eval(torch_states)
    for body, e, output in zip(nanflat_body_a, nanflat_epsilon_a, outs):
        logger.debug2(f'body: {body.aeb}, epsilon: {e}')
        if e > np.random.rand():
            logger.debug2(f'Random action')
            action = np.random.randint(body.action_dim)
        else:
            logger.debug2(f'Greedy action')
            action = torch.max(output, dim=1)[1].item()
        nanflat_action_a.append(action)
        logger.debug2(f'epsilon: {e}, outputs: {output}, action: {action}')
    return nanflat_action_a


def act_with_boltzmann(body, state, net, tau, gpu):
    state_seq = body.memory.state_buffer.maxlen > 0
    logger.debug2(f'Length state buffer: {body.memory.state_buffer.maxlen}')
    atari, flatten = set_flags(body)
    torch_state = create_torch_state(state, body.memory.state_buffer, gpu, state_seq, body.memory.state_buffer.maxlen, atari, flatten)
    out = net.wrap_eval(torch_state)
    out_with_temp = torch.div(out, tau).squeeze_(dim=0)
    probs = F.softmax(out_with_temp.cpu(), dim=0).data.numpy()
    action = np.random.choice(list(range(body.action_dim)), p=probs)
    logger.debug2('out with temp: {}, prob: {}, action: {}'.format(out_with_temp, probs, action))
    return action


def multi_act_with_boltzmann(nanflat_body_a, state_a, net, nanflat_tau_a, gpu):
    nanflat_state_a = util.nanflatten(state_a)
    cat_state_a = np.concatenate(nanflat_state_a).astype(float)
    torch_state = torch.from_numpy(cat_state_a).float()
    if torch.cuda.is_available() and gpu:
        torch_state = torch_state.cuda()
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
        probs = F.softmax(out_with_temp.cpu(), dim=0).data.numpy()
        action = np.random.choice(list(range(body.action_dim)), p=probs)
        logger.debug3(f'''
        body: {body.aeb}, net idx: {start_idx}-{end_idx}
        probs: {probs}, action: {action}''')
        nanflat_action_a.append(action)
        start_idx = end_idx
    return nanflat_action_a


def multi_head_act_with_boltzmann(nanflat_body_a, state_a, net, nanflat_tau_a, gpu):
    nanflat_state_a = util.nanflatten(state_a)
    torch_states = []
    for state in nanflat_state_a:
        state = state.astype('float')
        torch_states.append(torch.from_numpy(state).float().unsqueeze_(dim=0))
    if torch.cuda.is_available() and gpu:
        for idx, torch_state in enumerate(torch_states):
            torch_states[idx] = torch_state.cuda()
    for idx, torch_state in enumerate(torch_states):
        torch_states[idx] = torch_state
    outs = net.wrap_eval(torch_states)
    out_with_temp = [torch.div(x, t) for x, t in zip(outs, nanflat_tau_a)]
    logger.debug2(f'taus: {nanflat_tau_a}, outs: {outs}, out_with_temp: {out_with_temp}')
    nanflat_action_a = []
    for body, output in zip(nanflat_body_a, out_with_temp):
        probs = F.softmax(output.cpu(), dim=1).data.numpy()[0]
        action = np.random.choice(list(range(body.action_dim)), p=probs)
        logger.debug3(f'''
        body: {body.aeb}, output: {output},
        probs: {probs}, action: {action}''')
        nanflat_action_a.append(action)
    return nanflat_action_a


# Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
def act_with_softmax(algorithm, state, body, gpu):
    '''Assumes actor network outputs one variable; the logits of a categorical probability distribution over the actions'''
    state_seq = body.memory.state_buffer.maxlen > 0
    torch_state = create_torch_state(state, body.memory.state_buffer, gpu, state_seq, body.memory.state_buffer.maxlen)
    out = algorithm.get_actor_output(torch_state, evaluate=False)
    if isinstance(out, list):  # guard multi-tails for cont action
        out = out[0]
    out.squeeze_(dim=0)
    probs = F.softmax(out, dim=0)
    m = Categorical(probs)
    action = m.sample()
    logger.debug2(f'Network output: {out.data}')
    logger.debug2(f'Probability of actions: {probs.data}')
    logger.debug(f'Action: {action.item()}, log prob: {m.log_prob(action).item()}')
    algorithm.saved_log_probs.append(m.log_prob(action))
    # Calculate entropy of the distribution
    H = - torch.sum(torch.mul(probs, torch.log(probs)))
    if np.isnan(H.data.cpu().numpy()):
        logger.debug(f'NaN entropy, setting to 0')
        H = torch.zeros(1)
        if torch.cuda.is_available() and gpu:
            H = H.cuda()
    algorithm.entropy.append(H)
    return action.item()


# Denny Britz has a very helpful implementation of an Actor Critic algorithm. This function is adapted from his approach. I highly recommend looking at his full implementation available here https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb
def act_with_gaussian(algorithm, state, body, gpu):
    '''Assumes net outputs two variables; the mean and std dev of a normal distribution'''
    state_seq = body.memory.state_buffer.maxlen > 0
    torch_state = create_torch_state(state, body.memory.state_buffer, gpu, state_seq, body.memory.state_buffer.maxlen)
    mu, sigma = algorithm.get_actor_output(torch_state, evaluate=False)
    mu.squeeze_(dim=0)
    sigma.squeeze_(dim=0)
    sigma = F.softplus(sigma) + 1e-5  # Ensures sigma > 0
    m = Normal(mu, sigma)
    action = m.sample()
    action = torch.clamp(action, -algorithm.continuous_action_clip, algorithm.continuous_action_clip)
    logger.debug2(f'Action: {action.item()}, log prob: {m.log_prob(action).item()}')
    algorithm.saved_log_probs.append(m.log_prob(action))
    # Calculate entropy of the distribution
    H = 0.5 * torch.log(2.0 * np.pi * np.e * sigma * sigma)
    if np.isnan(H.data.cpu().numpy()):
        logger.debug(f'NaN entropy, setting to 0')
        H = torch.zeros(1)
        if torch.cuda.is_available() and gpu:
            H = H.cuda()
    algorithm.entropy.append(H)
    return action.data


def act_with_multivariate_gaussian(algorithm, state, body, gpu):
    '''Assumes net outputs two tensors which contain the mean and std dev of a multivariate normal distribution'''
    raise NotImplementedError
    return np.nan


def update_linear_decay(cls, space_clock):
    epi = space_clock.get('epi')
    rise = cls.explore_var_end - cls.explore_var_start
    slope = rise / float(cls.explore_anneal_epi)
    explore_var = max(slope * (epi - 1) + cls.explore_var_start, cls.explore_var_end)
    cls.nanflat_explore_var_a = [explore_var] * cls.agent.body_num
    logger.debug3(f'nanflat_explore_var_a: {cls.nanflat_explore_var_a[0]}')
    return cls.nanflat_explore_var_a


def update_multi_linear_decay(cls, _space_clock):
    nanflat_explore_var_a = []
    for body in cls.agent.nanflat_body_a:
        # use body-clock instead of space clock
        epi = body.env.clock.get('epi')
        rise = cls.explore_var_end - cls.explore_var_start
        slope = rise / float(cls.explore_anneal_epi)
        explore_var = max(slope * (epi - 1) + cls.explore_var_start, cls.explore_var_end)
        nanflat_explore_var_a.append(explore_var)
    cls.nanflat_explore_var_a = nanflat_explore_var_a
    logger.debug3(f'nanflat_explore_var_a: {cls.nanflat_explore_var_a}')
    return cls.nanflat_explore_var_a


def decay_learning_rate(algorithm, nets):
    '''
    Decay learning rate for each net by the decay method update_lr() defined in them.
    In the future, might add more flexible lr adjustment, like boosting and decaying on need.
    '''
    space_clock = util.s_get(algorithm, 'aeb_space.clock')
    t = space_clock.get('total_t')
    epi = space_clock.get('epi')
    if algorithm.net.decay_lr and t >= algorithm.net.decay_lr_min_timestep:
        if t % algorithm.net.decay_lr_frequency == 0:
            logger.info(f'Epi {epi}: Decaying learning rate...')
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
