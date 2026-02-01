from functools import partial, wraps
from slm_lab.lib import logger, optimizer, util
from slm_lab.lib.env_var import lab_mode
import numpy as np
import os
import pydash as ps
import torch
import torch.nn as nn

logger = logger.get_logger(__name__)

# register custom torch.optim (Global variants for A3C Hogwild)
setattr(torch.optim, 'GlobalAdam', optimizer.GlobalAdam)
setattr(torch.optim, 'GlobalRMSprop', optimizer.GlobalRMSprop)


class NoOpLRScheduler:
    '''Symbolic LRScheduler class for API consistency'''

    def __init__(self, optim):
        self.optim = optim

    def step(self, epoch=None):
        pass

    def get_last_lr(self):
        if hasattr(self.optim, 'defaults'):
            return self.optim.defaults['lr']
        else:  # TODO retrieve lr more generally
            return self.optim.param_groups[0]['lr']


def build_fc_model(dims, activation=None):
    '''Build a full-connected model by interleaving nn.Linear and activation_fn'''
    assert len(dims) >= 2, 'dims need to at least contain input, output'
    # shift dims and make pairs of (in, out) dims per layer
    dim_pairs = list(zip(dims[:-1], dims[1:]))
    layers = []
    for in_d, out_d in dim_pairs:
        layers.append(nn.Linear(in_d, out_d))
        if activation is not None:
            layers.append(get_activation_fn(activation))
    model = nn.Sequential(*layers)
    return model


def get_nn_name(uncased_name):
    '''Helper to get the proper name in PyTorch nn given a case-insensitive name'''
    for nn_name in nn.__dict__:
        if uncased_name.lower() == nn_name.lower():
            return nn_name
    raise ValueError(f'Name {uncased_name} not found in {nn.__dict__}')


def get_activation_fn(activation):
    '''Helper to generate activation function layers for net'''
    ActivationClass = getattr(nn, get_nn_name(activation))
    return ActivationClass()


def get_loss_fn(cls, loss_spec):
    '''Helper to parse loss param and construct loss_fn for net'''
    LossClass = getattr(nn, get_nn_name(loss_spec['name']))
    loss_spec = ps.omit(loss_spec, 'name')
    loss_fn = LossClass(**loss_spec)
    return loss_fn


def get_lr_scheduler(optim, lr_scheduler_spec, steps_per_schedule=1):
    '''Helper to parse lr_scheduler param and construct Pytorch optim.lr_scheduler.

    Args:
        optim: The optimizer to schedule
        lr_scheduler_spec: Scheduler configuration dict
        steps_per_schedule: Number of env frames processed per scheduler.step() call.
            For PPO: training_frequency * num_envs (e.g., 128 * 8 = 1024)
            This converts frame-based specs to update-based scheduling.
    '''
    if ps.is_empty(lr_scheduler_spec):
        lr_scheduler = NoOpLRScheduler(optim)
    elif lr_scheduler_spec['name'] == 'LinearToZero':
        LRSchedulerClass = getattr(torch.optim.lr_scheduler, 'LambdaLR')
        frame = float(lr_scheduler_spec['frame'])
        # Convert from total frames to number of scheduler updates
        num_updates = max(1, frame / steps_per_schedule)
        lr_scheduler = LRSchedulerClass(optim, lr_lambda=lambda x, n=num_updates: max(0, 1 - x / n))
    else:
        LRSchedulerClass = getattr(torch.optim.lr_scheduler, lr_scheduler_spec['name'])
        lr_scheduler_spec = ps.omit(lr_scheduler_spec, 'name')
        lr_scheduler = LRSchedulerClass(optim, **lr_scheduler_spec)
    return lr_scheduler


def get_optim(net, optim_spec):
    '''Helper to parse optim param and construct optim for net'''
    OptimClass = getattr(torch.optim, optim_spec['name'])
    optim_spec = ps.omit(optim_spec, 'name')
    if torch.is_tensor(net):  # for non-net tensor variable
        optim = OptimClass([net], **optim_spec)
    else:
        optim = OptimClass(net.parameters(), **optim_spec)
    return optim


def get_policy_out_dim(agent):
    '''Helper method to construct the policy network out_dim for an agent according to is_discrete, action_type'''
    action_dim = agent.action_dim
    if agent.is_discrete:
        if agent.action_type == 'multi_discrete':
            assert ps.is_list(action_dim), action_dim
            policy_out_dim = action_dim
        else:
            assert isinstance(action_dim, (int, np.integer)), action_dim
            policy_out_dim = action_dim
    else:
        assert isinstance(action_dim, (int, np.integer)), action_dim
        if action_dim == 1:  # single action, use [loc, scale]
            policy_out_dim = 2
        else:  # multi-action, use [locs], [scales]
            policy_out_dim = [action_dim, action_dim]
    return policy_out_dim


def get_out_dim(agent, add_critic=False):
    '''Construct the NetClass out_dim for an agent according to is_discrete, action_type, and whether to add a critic unit'''
    policy_out_dim = get_policy_out_dim(agent)
    if add_critic:
        if ps.is_list(policy_out_dim):
            out_dim = policy_out_dim + [1]
        else:
            out_dim = [policy_out_dim, 1]
    else:
        out_dim = policy_out_dim
    return out_dim


def init_layers(net, init_fn_name):
    '''Primary method to initialize the weights of the layers of a network'''
    if init_fn_name is None:
        return

    # get nonlinearity
    nonlinearity = get_nn_name(net.hid_layers_activation).lower()
    if nonlinearity == 'leakyrelu':
        nonlinearity = 'leaky_relu'  # guard name

    # get init_fn and add arguments depending on nonlinearity
    init_fn = getattr(nn.init, init_fn_name)
    if 'kaiming' in init_fn_name:  # has 'nonlinearity' as arg
        assert nonlinearity in ['relu', 'leaky_relu'], f'Kaiming initialization not supported for {nonlinearity}'
        init_fn = partial(init_fn, nonlinearity=nonlinearity)
    elif 'orthogonal' in init_fn_name or 'xavier' in init_fn_name:  # has 'gain' as arg
        gain = nn.init.calculate_gain(nonlinearity)
        init_fn = partial(init_fn, gain=gain)
    else:
        pass

    # finally, apply init_params to each layer in its modules
    net.apply(partial(init_params, init_fn=init_fn))


def init_params(module, init_fn):
    '''Initialize module's weights using init_fn, and biases to 0.0'''
    bias_init = 0.0
    classname = util.get_class_name(module)
    if 'Net' in classname:  # skip if it's a net, not pytorch layer
        pass
    elif classname == 'BatchNorm2d':
        pass  # can't init BatchNorm2d
    elif any(k in classname for k in ('Conv', 'Linear')):
        init_fn(module.weight)
        nn.init.constant_(module.bias, bias_init)
    elif 'GRU' in classname:
        for name, param in module.named_parameters():
            if 'weight' in name:
                init_fn(param)
            elif 'bias' in name:
                nn.init.constant_(param, bias_init)
    else:
        pass


def init_tails(net, actor_init_std=None, critic_init_std=None):
    '''Reinitialize output head layers with specific stds (CleanRL-style).

    For PPO/ActorCritic with shared network, proper head initialization is critical:
    - Actor head: small std (0.01) for near-uniform initial policy
    - Critic head: std=1 for unbiased initial value estimates

    This follows CleanRL's layer_init pattern where output heads get different
    initialization than hidden layers.

    Args:
        net: Network with self.tails attribute (ModuleList of [actor_tail, critic_tail])
        actor_init_std: std for actor output head orthogonal init (default: None = no reinit)
        critic_init_std: std for critic output head orthogonal init (default: None = no reinit)
    '''
    if not hasattr(net, 'tails') or not isinstance(net.tails, nn.ModuleList):
        return  # Only applies to multi-tail networks (shared actor-critic)

    tails = list(net.tails)
    if len(tails) < 2:
        return  # Need at least actor and critic tails

    # Actor tail is first, critic tail is last
    actor_tail = tails[0]
    critic_tail = tails[-1]

    # Reinitialize actor head with small std for near-uniform initial policy
    if actor_init_std is not None:
        for module in actor_tail.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, actor_init_std)
                nn.init.constant_(module.bias, 0.0)
        logger.debug(f'Reinitialized actor tail with std={actor_init_std}')

    # Reinitialize critic head
    if critic_init_std is not None:
        for module in critic_tail.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, critic_init_std)
                nn.init.constant_(module.bias, 0.0)
        logger.debug(f'Reinitialized critic tail with std={critic_init_std}')


# params methods


def save(net, model_path):
    '''Save model weights to path'''
    torch.save(net.state_dict(), util.smart_path(model_path))


def save_algorithm(algorithm, ckpt=None):
    '''Save all the nets for an algorithm'''
    agent = algorithm.agent
    net_names = algorithm.net_names
    model_prepath = agent.spec['meta']['model_prepath']
    if ckpt is not None:
        model_prepath += f'_ckpt-{ckpt}'
    for net_name in net_names:
        net = getattr(algorithm, net_name)
        model_path = f'{model_prepath}_{net_name}_model.pt'
        save(net, model_path)
        optim_name = net_name.replace('net', 'optim')
        optim = getattr(algorithm, optim_name, None)
        if optim is not None:  # only trainable net has optim
            optim_path = f'{model_prepath}_{net_name}_optim.pt'
            save(optim, optim_path)
    logger.debug(f'Saved algorithm {util.get_class_name(algorithm)} nets {net_names} to {model_prepath}_*.pt')


def load(net, model_path):
    '''Load model weights from a path into a net module'''
    device = None if torch.cuda.is_available() else 'cpu'
    net.load_state_dict(torch.load(util.smart_path(model_path), map_location=device))


def load_algorithm(algorithm):
    '''Load all the nets for an algorithm'''
    agent = algorithm.agent
    net_names = algorithm.net_names
    model_prepath = agent.spec['meta']['model_prepath']
    is_enjoy = lab_mode() == 'enjoy'
    if is_enjoy:
        model_prepath += '_ckpt-best'
    logger.info(f'Loading algorithm {util.get_class_name(algorithm)} nets {net_names} from {model_prepath}_*.pt')
    for net_name in net_names:
        net = getattr(algorithm, net_name)
        model_path = f'{model_prepath}_{net_name}_model.pt'
        load(net, model_path)
        if is_enjoy:  # skip loading optim in enjoy mode - not needed for inference
            continue
        optim_name = net_name.replace('net', 'optim')
        optim = getattr(algorithm, optim_name, None)
        if optim is not None:  # only trainable net has optim
            optim_path = f'{model_prepath}_{net_name}_optim.pt'
            load(optim, optim_path)


def copy(src_net, tar_net):
    '''Copy model weights from src to target'''
    state_dict = src_net.state_dict()
    # Transfer state dict to target device if different
    tar_device = next(tar_net.parameters()).device
    src_device = next(src_net.parameters()).device
    if tar_device != src_device:
        state_dict = {k: v.to(tar_device) for k, v in state_dict.items()}
    tar_net.load_state_dict(state_dict)


def polyak_update(src_net, tar_net, old_ratio=0.5):
    '''
    Polyak weight update to update a target tar_net, retain old weights by its ratio, i.e.
    target <- old_ratio * source + (1 - old_ratio) * target
    '''
    for src_param, tar_param in zip(src_net.parameters(), tar_net.parameters()):
        tar_param.data.copy_(old_ratio * src_param.data + (1.0 - old_ratio) * tar_param.data)


def update_target_net(src_net, tar_net, frame, num_envs):
    '''
    Update target network using replace or polyak strategy.
    For replace: only updates every update_frequency frames.
    For polyak: updates every call with exponential moving average.
    
    @param src_net: Source network to copy/blend from
    @param tar_net: Target network to update
    @param frame: Current training frame for frequency gating
    @param num_envs: Number of parallel environments (for frame_mod calculation)
    '''
    from slm_lab.lib import util
    
    if src_net.update_type == 'replace':
        if util.frame_mod(frame, src_net.update_frequency, num_envs):
            copy(src_net, tar_net)
    elif src_net.update_type == 'polyak':
        polyak_update(src_net, tar_net, src_net.polyak_coef)
    else:
        raise ValueError(f'Unknown update_type "{src_net.update_type}". Should be "replace" or "polyak".')


def to_check_train_step():
    '''Condition for running assert_trained'''
    return os.environ.get('PY_ENV') == 'test' or lab_mode() == 'dev'


def dev_check_train_step(fn):
    '''
    Decorator to check if net.train_step actually updates the network weights properly
    Triggers only if to_check_train_step is True (dev/test mode)
    @example

    @net_util.dev_check_train_step
    def train_step(self, ...):
        ...
    '''
    @wraps(fn)
    def check_fn(*args, **kwargs):
        if not to_check_train_step():
            return fn(*args, **kwargs)

        net = args[0]  # first arg self
        # get pre-update parameters to compare
        pre_params = [param.clone() for param in net.parameters()]

        # run train_step, get loss
        loss = fn(*args, **kwargs)
        # Skip checks if loss indicates skipped update (NaN protection returns 1e-10)
        loss_val = loss.item()
        if loss_val < 1e-9:  # Sentinel value or near-zero loss, skip checks
            return loss
        assert not torch.isnan(loss).any(), loss

        # get post-update parameters to compare
        post_params = [param.clone() for param in net.parameters()]
        if loss_val == 0.0:
            # if loss is 0, there should be no updates
            # TODO if without momentum, parameters should not change too
            for p_name, param in net.named_parameters():
                assert param.grad.norm() == 0
        else:
            # check parameter updates
            try:
                assert not all(torch.equal(w1, w2) for w1, w2 in zip(pre_params, post_params)), f'Model parameter is not updated in train_step(), check if your tensor is detached from graph. Loss: {loss:g}'
            except Exception as e:
                logger.error(e)
                if os.environ.get('PY_ENV') == 'test':
                    # raise error if in unit test
                    raise(e)

            # check grad norms
            min_norm, max_norm = 0.0, 1e5
            for p_name, param in net.named_parameters():
                try:
                    grad_norm = param.grad.norm()
                    assert min_norm < grad_norm < max_norm, f'Gradient norm for {p_name} is {grad_norm:g}, fails the extreme value check {min_norm} < grad_norm < {max_norm}. Loss: {loss:g}. Check your network and loss computation.'
                except Exception as e:
                    logger.warning(e)
        logger.debug('Passed network parameter update check.')
        # store grad norms for debugging
        net.store_grad_norms()
        return loss
    return check_fn


def get_grad_norms(algorithm):
    '''Gather all the net's grad norms of an algorithm for debugging'''
    grad_norms = []
    for net_name in algorithm.net_names:
        net = getattr(algorithm, net_name)
        if net.grad_norms is not None:
            grad_norms.extend(net.grad_norms)
    return grad_norms


def init_global_nets(algorithm):
    '''
    Initialize global_nets for Hogwild using an identical instance of an algorithm from an isolated Session
    in spec.meta.distributed, specify either:
    - 'shared': global network parameter is shared all the time. In this mode, algorithm local network will be replaced directly by global_net via overriding by identify attribute name
    - 'synced': global network parameter is periodically synced to local network after each gradient push. In this mode, algorithm will keep a separate reference to `global_{net}` for each of its network

    NOTE: A3C Hogwild is CPU-only because PyTorch share_memory_() requires CPU tensors.
    For GPU-accelerated training, use A2C or PPO instead.
    '''
    dist_mode = algorithm.agent.spec['meta']['distributed']
    assert dist_mode in ('shared', 'synced'), 'Unrecognized distributed mode'
    global_nets = {}
    for net_name in algorithm.net_names:
        optim_name = net_name.replace('net', 'optim')
        if not hasattr(algorithm, optim_name):  # only for trainable network, i.e. has an optim
            continue
        g_net = getattr(algorithm, net_name)
        # Move to CPU for share_memory_() (required by PyTorch multiprocessing)
        g_net.to('cpu')
        g_net.share_memory()  # make net global
        if dist_mode == 'shared':  # use the same name to override the local net
            global_nets[net_name] = g_net
        else:  # keep a separate reference for syncing
            global_nets[f'global_{net_name}'] = g_net
        # if optim is Global, set to override the local optim and its scheduler
        optim = getattr(algorithm, optim_name)
        if hasattr(optim, 'share_memory'):
            optim.share_memory()  # make optim global
            global_nets[optim_name] = optim
            lr_scheduler_name = net_name.replace('net', 'lr_scheduler')
            lr_scheduler = getattr(algorithm, lr_scheduler_name)
            global_nets[lr_scheduler_name] = lr_scheduler
    logger.info(f'Initialized global_nets attr {list(global_nets.keys())} for Hogwild')
    return global_nets


def set_global_nets(algorithm, global_nets):
    '''For Hogwild, set attr built in init_global_nets above. Use in algorithm init.'''
    # set attr first so algorithm always has self.global_{net} to pass into train_step
    for net_name in algorithm.net_names:
        setattr(algorithm, f'global_{net_name}', None)
    # set attr created in init_global_nets
    if global_nets is not None:
        # set global nets and optims
        util.set_attr(algorithm, global_nets)
        logger.info(f'Set global_nets attr {list(global_nets.keys())} for Hogwild')


def push_global_grads(net, global_net):
    '''Push gradients to global_net, call inside train_step between loss.backward() and optim.step()'''
    for param, global_param in zip(net.parameters(), global_net.parameters()):
        if global_param.grad is not None:
            return  # quick skip
        if param.grad is not None:
            # Transfer grad to global_net's device (CPU for shared memory)
            global_param._grad = param.grad.to(global_param.device)


def build_tails(tail_in_dim, out_dim, out_layer_activation, log_std_init=None):
    '''Build output tails with optional state-independent log_std (CleanRL-style for continuous control).'''
    import numpy as np
    import pydash as ps

    if isinstance(out_dim, (int, np.integer)):
        return build_fc_model([tail_in_dim, out_dim], out_layer_activation), None

    # State-independent log_std: out_dim = [action_dim, action_dim] for continuous actions
    if log_std_init is not None and len(out_dim) == 2 and out_dim[0] == out_dim[1]:
        action_dim = out_dim[0]
        out_activ = out_layer_activation[0] if ps.is_list(out_layer_activation) else out_layer_activation
        return build_fc_model([tail_in_dim, action_dim], out_activ), nn.Parameter(torch.ones(action_dim) * log_std_init)

    # Multi-tail output
    if not ps.is_list(out_layer_activation):
        out_layer_activation = [out_layer_activation] * len(out_dim)
    return nn.ModuleList([build_fc_model([tail_in_dim, d], a) for d, a in zip(out_dim, out_layer_activation)]), None


def forward_tails(x, tails, log_std=None):
    '''Forward pass through tails, handling log_std expansion if present.'''
    if log_std is not None:
        return [tails(x), log_std.expand_as(tails(x))]
    return [t(x) for t in tails] if isinstance(tails, nn.ModuleList) else tails(x)
