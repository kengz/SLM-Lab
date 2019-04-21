'''
The monitor module with data_space
Monitors agents, environments, sessions, trials, experiments, evolutions, and handles all the data produced by the Lab components.
InfoSpace handles the unified hyperdimensional data for SLM Lab, used for analysis and experiment planning. Sources data from monitor.
Each dataframe resolves from the coarsest dimension to the finest, with data coordinates coor in the form: (experiment,trial,session,agent,env,body)
The resolution after session is the AEB space, hence it is a subspace.
AEB space is not necessarily tabular, and hence the data is NoSQL.

The data_space is congruent to the coor, with proper resolution.
E.g. (experiment,trial,session) specifies the session_data of a session, ran over multiple episodes on the AEB space.

Space ordering:
InfoSpace: the general space for complete information
AEBSpace: subspace of InfoSpace for a specific session
AgentSpace: space agent instances, subspace of AEBSpace
EnvSpace: space of env instances, subspace of AEBSpace
DataSpace: a data space storing an AEB data projected to a-axis, and its dual projected to e-axis. This is so that a-proj data like action_space from agent_space can be used by env_space, which requires e-proj data, and vice versa.

Object reference (for agent to access env properties, vice versa):
Agents - AgentSpace - AEBSpace - EnvSpace - Envs
'''
from gym import spaces
from slm_lab.agent import AGENT_DATA_NAMES
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.net import net_util
from slm_lab.env import ENV_DATA_NAMES
from slm_lab.experiment import analysis
from slm_lab.lib import logger, math_util, util
from slm_lab.spec import spec_util
import numpy as np
import pandas as pd
import pydash as ps
import time
import torch

# These correspond to the control unit classes, lower cased
COOR_AXES = [
    'experiment',
    'trial',
    'session',
]
COOR_AXES_ORDER = {
    axis: idx for idx, axis in enumerate(COOR_AXES)
}
COOR_DIM = len(COOR_AXES)
logger = logger.get_logger(__name__)


def enable_aeb_space(session):
    '''Enable aeb_space to session use Lab's data-monitor and analysis modules'''
    session.aeb_space = AEBSpace(session.spec, session.info_space)
    # make compatible with the generic multiagent setup
    session.aeb_space.body_space = DataSpace('body', session.aeb_space)
    body_v = np.full(session.aeb_space.aeb_shape, np.nan, dtype=object)
    body_v[0, 0, 0] = session.agent.body
    session.aeb_space.body_space.add(body_v)
    session.agent.aeb_space = session.aeb_space
    session.env.aeb_space = session.aeb_space


def get_action_type(action_space):
    '''Method to get the action type to choose prob. dist. to sample actions from NN logits output'''
    if isinstance(action_space, spaces.Box):
        shape = action_space.shape
        assert len(shape) == 1
        if shape[0] == 1:
            return 'continuous'
        else:
            return 'multi_continuous'
    elif isinstance(action_space, spaces.Discrete):
        return 'discrete'
    elif isinstance(action_space, spaces.MultiDiscrete):
        return 'multi_discrete'
    elif isinstance(action_space, spaces.MultiBinary):
        return 'multi_binary'
    else:
        raise NotImplementedError


class Body:
    '''
    Body of an agent inside an environment. This acts as the main variable storage and bridge between agent and environment to pair them up properly in the generalized multi-agent-env setting.
    '''

    def __init__(self, env, agent_spec, aeb=(0, 0, 0), aeb_space=None):
        # essential reference variables
        self.agent = None  # set later
        self.env = env
        self.aeb = aeb
        self.a, self.e, self.b = aeb
        self.nanflat_a_idx, self.nanflat_e_idx = self.a, self.e

        # for action policy exploration, so be set in algo during init_algorithm_params()
        self.explore_var = np.nan

        # body stats variables
        self.loss = np.nan  # training losses
        # diagnostics variables/stats from action_policy prob. dist.
        self.action_tensor = None
        self.action_pd = None  # for the latest action, to compute entropy and log prob
        self.entropies = []  # action entropies for exploration
        self.log_probs = []  # action log probs
        # mean values for debugging
        self.mean_entropy = np.nan
        self.mean_log_prob = np.nan
        self.mean_grad_norm = np.nan

        # stores running mean and std dev of states
        self.state_mean = np.nan
        self.state_std_dev_int = np.nan
        self.state_std_dev = np.nan
        self.state_n = 0

        self.total_reward = np.nan
        # store current and best reward_ma for model checkpointing and early termination if all the environments are solved
        self.best_reward_ma = -np.inf
        self.eval_reward_ma = np.nan

        # dataframes to track data for analysis.analyze_session
        # track training data within run_episode
        self.train_df = pd.DataFrame(columns=[
            'epi', 'total_t', 't', 'wall_t', 'fps', 'reward', 'loss', 'lr',
            'explore_var', 'entropy_coef', 'entropy', 'log_prob', 'grad_norm'])
        # track eval data within run_eval_episode. the same as train_df except for reward
        self.eval_df = self.train_df.copy()

        if aeb_space is None:  # singleton mode
            # the specific agent-env interface variables for a body
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.observable_dim = self.env.observable_dim
            self.state_dim = self.observable_dim['state']
            self.action_dim = self.env.action_dim
            self.is_discrete = self.env.is_discrete
        else:
            self.space_init(aeb_space)

        self.action_type = get_action_type(self.action_space)
        self.action_pdtype = agent_spec[self.a]['algorithm'].get('action_pdtype')
        if self.action_pdtype in (None, 'default'):
            self.action_pdtype = policy_util.ACTION_PDS[self.action_type][0]

    def action_pd_update(self):
        '''Calculate and update action entropy and log_prob using self.action_pd. Call this in agent.update()'''
        if self.action_pd is None:  # skip if None
            return
        # mean for single and multi-action
        entropy = self.action_pd.entropy().mean(dim=0)
        self.entropies.append(entropy)
        log_prob = self.action_pd.log_prob(self.action_tensor).mean(dim=0)
        self.log_probs.append(log_prob)
        assert not torch.isnan(log_prob)

    def update(self, state, action, reward, next_state, done):
        '''Interface update method for body at agent.update()'''
        self.total_reward = math_util.nan_add(self.total_reward, reward)

    def calc_df_row(self, env):
        '''Calculate a row for updating train_df or eval_df.'''
        total_t = self.env.clock.get('total_t')
        wall_t = env.clock.get_elapsed_wall_t()
        fps = 0 if wall_t == 0 else total_t / wall_t
        row = pd.Series({
            # epi and total_t are always measured from training env
            'epi': self.env.clock.get('epi'),
            'total_t': total_t,
            # t and reward are measured from a given env or eval_env
            't': env.clock.get('t'),
            'wall_t': wall_t,
            'fps': fps,
            'reward': self.total_reward,
            'loss': self.loss,
            'lr': self.get_mean_lr(),
            'explore_var': self.explore_var,
            'entropy_coef': self.entropy_coef if hasattr(self, 'entropy_coef') else np.nan,
            'entropy': self.mean_entropy,
            'log_prob': self.mean_log_prob,
            'grad_norm': self.mean_grad_norm,
        }, dtype=np.float32)
        assert all(col in self.train_df.columns for col in row.index), f'Mismatched row keys: {row.index} vs df columns {self.train_df.columns}'
        return row

    def epi_reset(self):
        '''
        Handles any body attribute reset at the start of an episode.
        This method is called automatically at base memory.epi_reset().
        '''
        t = self.env.clock.t
        assert t == 0, f'aeb: {self.aeb}, t: {t}'
        if hasattr(self, 'aeb_space'):
            self.space_fix_stats()

    def epi_update(self):
        '''Update to append data at the end of an episode (when env.done is true)'''
        assert self.env.done
        row = self.calc_df_row(self.env)
        # append efficiently to df
        self.train_df.loc[len(self.train_df)] = row

    def eval_update(self, eval_env, total_reward):
        '''Update to append data at eval checkpoint'''
        row = self.calc_df_row(eval_env)
        row['total_reward'] = total_reward
        # append efficiently to df
        self.eval_df.loc[len(self.eval_df)] = row
        # update current reward_ma
        self.eval_reward_ma = self.eval_df[-analysis.MA_WINDOW:]['reward'].mean()

    def flush(self):
        '''Update and flush gradient-related variables after training step similar.'''
        # update
        self.mean_entropy = torch.tensor(self.entropies).mean().item()
        self.mean_log_prob = torch.tensor(self.log_probs).mean().item()
        # net.grad_norms is only available in dev mode for efficiency
        grad_norms = net_util.get_grad_norms(self.agent.algorithm)
        self.mean_grad_norm = np.nan if ps.is_empty(grad_norms) else np.mean(grad_norms)

        # flush
        self.action_tensor = None
        self.action_pd = None
        self.entropies = []
        self.log_probs = []

    def __str__(self):
        return 'body: ' + util.to_json(util.get_class_attr(self))

    def get_mean_lr(self):
        '''Gets the average current learning rate of the algorithm's nets.'''
        if not hasattr(self.agent.algorithm, 'net_names'):
            return np.nan
        lrs = []
        for net_name in self.agent.algorithm.net_names:
            # we are only interested in directly trainable network, so exclude target net
            if net_name is 'target_net':
                continue
            net = getattr(self.agent.algorithm, net_name)
            lrs.append(net.lr_scheduler.get_lr())
        return np.mean(lrs)

    def get_log_prefix(self):
        '''Get the prefix for logging'''
        spec = self.agent.spec
        info_space = self.agent.info_space
        clock = self.env.clock
        prefix = f'{spec["name"]}_t{info_space.get("trial")}_s{info_space.get("session")}, aeb{self.aeb}'
        return prefix

    def log_summary(self, body_df_kind='eval'):
        '''Log the summary for this body when its environment is done'''
        prefix = self.get_log_prefix()
        df = self.eval_df if body_df_kind == 'eval' else self.train_df
        last_row = df.iloc[-1]
        row_str = ', '.join([f'{k}: {v:g}' for k, v in last_row.items()])
        reward_ma = df[-analysis.MA_WINDOW:]['reward'].mean()
        reward_ma_str = f'last-{analysis.MA_WINDOW}-epi avg: {reward_ma:g}'
        msg = f'{prefix} [{body_df_kind}_df] {row_str}, {reward_ma_str}'
        logger.info(msg)

    def space_init(self, aeb_space):
        '''Post init override for space body. Note that aeb is already correct from __init__'''
        self.aeb_space = aeb_space
        # to be reset properly later
        self.nanflat_a_idx, self.nanflat_e_idx = None, None

        self.observation_space = self.env.observation_spaces[self.a]
        self.action_space = self.env.action_spaces[self.a]
        self.observable_dim = self.env._get_observable_dim(self.observation_space)
        self.state_dim = self.observable_dim['state']
        self.action_dim = self.env._get_action_dim(self.action_space)
        self.is_discrete = self.env._is_discrete(self.action_space)

    def space_fix_stats(self):
        '''the space control loop will make agent append stat at done, so to offset for that, pop it at reset'''
        for action_stat in [self.entropies, self.log_probs]:
            if len(action_stat) > 0:
                action_stat.pop()


class DataSpace:
    '''
    AEB data space. Store all data from RL system in standard aeb-shaped tensors.
    '''

    def __init__(self, data_name, aeb_space):
        self.data_name = data_name
        self.aeb_space = aeb_space
        self.aeb_shape = aeb_space.aeb_shape

        # data from env have shape (eab), need to swap
        self.to_swap = self.data_name in ENV_DATA_NAMES
        self.swap_aeb_shape = self.aeb_shape[1], self.aeb_shape[0], self.aeb_shape[2]

        self.data_shape = self.swap_aeb_shape if self.to_swap else self.aeb_shape
        self.data_type = object if self.data_name in ['state', 'action'] else np.float32
        self.data = None  # standard data in aeb_shape
        self.swap_data = None

    def __str__(self):
        if self.data is None:
            return '<None>'
        s = '['
        for a, a_arr in enumerate(self.data):
            s += f'\n  a:{a} ['
            for e, e_arr in enumerate(a_arr):
                s += f'\n    e:{e} ['
                for b, val in enumerate(e_arr):
                    s += f'\n      b:{b} {val}'
                s += ']'
            s += ']'
        s += '\n]'
        return s

    def __bool__(self):
        return util.nonan_all(self.data)

    def init_data_v(self):
        '''Method to init a data volume filled with np.nan'''
        data_v = np.full(self.data_shape, np.nan, dtype=self.data_type)
        return data_v

    def init_data_s(self, a=None, e=None):
        '''Method to init a data surface (subset of data volume) filled with np.nan.'''
        body_s = self.aeb_space.body_space.get(a=a, e=e)
        data_s = np.full(body_s.shape, np.nan, dtype=self.data_type)
        return data_s

    def add(self, data_v):
        '''
        Take raw data from RL system and construct numpy object self.data.
        If data is from env, auto-swap the data to aeb standard shape.
        @param {[x: [y: [body_v]]} data_v As collected in RL sytem.
        @returns {array} data Tensor in standard aeb shape.
        '''
        new_data = np.array(data_v)  # no type restriction, auto-infer
        if self.to_swap:  # data from env has shape eab
            self.swap_data = new_data
            self.data = new_data.swapaxes(0, 1)
        else:
            self.data = new_data
            self.swap_data = new_data.swapaxes(0, 1)
        return self.data

    def get(self, a=None, e=None):
        '''
        Get the data projected on a or e axes for use by agent_space, env_space.
        @param {int} a The index a of an agent in agent_space
        @param {int} e The index e of an env in env_space
        @returns {array} data_x Where x is a or e.
        '''
        if e is None:
            return self.data[a]
        elif a is None:
            return self.swap_data[e]
        else:
            return self.data[a][e]


class AEBSpace:

    def __init__(self, spec, info_space):
        self.info_space = info_space
        self.spec = spec
        self.clock = None  # the finest common refinement as space clock
        self.agent_space = None
        self.env_space = None
        self.body_space = None
        (self.aeb_list, self.aeb_shape, self.aeb_sig) = self.get_aeb_info(self.spec)
        self.data_spaces = self.init_data_spaces()

    def get_aeb_info(cls, spec):
        '''
        Get from spec the aeb_list, aeb_shape and aeb_sig, which are used to resolve agent_space and env_space.
        @returns {list, (a,e,b), array([a, e, b])} aeb_list, aeb_shape, aeb_sig
        '''
        aeb_list = spec_util.resolve_aeb(spec)
        aeb_shape = util.get_aeb_shape(aeb_list)
        aeb_sig = np.full(aeb_shape, np.nan)
        for aeb in aeb_list:
            aeb_sig.itemset(aeb, 1)
        return aeb_list, aeb_shape, aeb_sig

    def init_data_spaces(self):
        self.data_spaces = {
            data_name: DataSpace(data_name, self)
            for data_name in AGENT_DATA_NAMES + ENV_DATA_NAMES
        }
        return self.data_spaces

    def init_data_s(self, data_names, a=None, e=None):
        '''Shortcut to init data_s_1, data_s_2, ...'''
        return tuple(self.data_spaces[data_name].init_data_s(a=a, e=e) for data_name in data_names)

    def init_data_v(self, data_names):
        '''Shortcut to init data_v_1, data_v_2, ...'''
        return tuple(self.data_spaces[data_name].init_data_v() for data_name in data_names)

    def init_body_space(self):
        '''Initialize the body_space (same class as data_space) used for AEB body resolution, and set reference in agents and envs'''
        self.body_space = DataSpace('body', self)
        body_v = np.full(self.aeb_shape, np.nan, dtype=object)
        for (a, e, b), sig in np.ndenumerate(self.aeb_sig):
            if sig == 1:
                env = self.env_space.get(e)
                body = Body(env, self.spec['agent'], aeb=(a, e, b), aeb_space=self)
                body_v[(a, e, b)] = body
        self.body_space.add(body_v)
        # complete the backward reference to env_space
        for env in self.env_space.envs:
            body_e = self.body_space.get(e=env.e)
            env.set_body_e(body_e)
        self.clock = self.env_space.get_base_clock()
        logger.info(util.self_desc(self))
        return self.body_space

    def add(self, data_name, data_v):
        '''
        Add a data to a data space, e.g. data actions collected per body, per agent, from agent_space, with AEB shape projected on a-axis, added to action_space.
        Could also be a shortcut to do batch add data_v_1, data_v_2, ...
        @param {str|[str]} data_name
        @param {[x: [yb_idx:[body_v]]} data_v, where x, y could be a, e interchangeably.
        @returns {DataSpace} data_space (aeb is implied)
        '''
        if ps.is_string(data_name):
            data_space = self.data_spaces[data_name]
            data_space.add(data_v)
            return data_space
        else:
            return tuple(self.add(d_name, d_v) for d_name, d_v in zip(data_name, data_v))

    def tick(self, unit=None):
        '''Tick all the clocks in env_space, and tell if all envs are done'''
        end_sessions = []
        for env in self.env_space.envs:
            if env.done:
                for body in env.nanflat_body_e:
                    body.log_summary(body_df_kind='train')
            env.clock.tick(unit or ('epi' if env.done else 't'))
            end_session = not (env.clock.get(env.max_tick_unit) < env.max_tick)
            end_sessions.append(end_session)
        return all(end_sessions)


class InfoSpace:
    def __init__(self, last_coor=None):
        '''
        Initialize the coor, the global point in info space that will advance according to experiment progress.
        The coor starts with null first since the coor may not start at the origin.
        '''
        self.coor = last_coor or {k: None for k in COOR_AXES}
        self.covered_space = []
        # used to id experiment sharing the same spec name
        self.experiment_ts = util.get_ts()
        # ckpt gets appened to extend prepath using util.get_prepath for saving models, e.g. ckpt_str = ckpt-epi10-totalt1000
        # ckpt = 'eval' is special for eval mode, so data files will save with `ckpt-eval`; no models will be saved, but to load models with normal ckpt it will find them using eval_model_prepath
        # e.g. 'epi24-totalt1000', 'eval', 'best'
        self.ckpt = None
        # e.g. 'data/dqn_cartpole_2018_12_19_085843/dqn_cartpole_t0_s0_ckpt-epi24-totalt1000'
        self.eval_model_prepath = None

    def reset_lower_axes(cls, coor, axis):
        '''Reset the axes lower than the given axis in coor'''
        axis_idx = COOR_AXES_ORDER[axis]
        for post_idx in range(axis_idx + 1, COOR_DIM):
            post_axis = COOR_AXES[post_idx]
            coor[post_axis] = None
        return coor

    def tick(self, axis):
        '''
        Advance the coor to the next point in axis (control unit class).
        If the axis value has been reset, update to 0, else increment. For all axes lower than the specified axis, reset to None.
        Note this will not skip coor in space, even though the covered space may not be rectangular.
        @example

        info_space.tick('session')
        session = Session(spec, info_space)
        '''
        assert axis in self.coor
        if axis == 'experiment':
            self.experiment_ts = util.get_ts()
        new_coor = self.coor.copy()
        if new_coor[axis] is None:
            new_coor[axis] = 0
        else:
            new_coor[axis] += 1
        new_coor = self.reset_lower_axes(new_coor, axis)
        self.covered_space.append(self.coor)
        self.coor = new_coor
        return self.coor

    def get(self, axis):
        return self.coor[axis]

    def set(self, axis, val):
        self.coor[axis] = val
        return self.coor[axis]
