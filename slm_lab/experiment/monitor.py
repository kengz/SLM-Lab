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


logger = logger.get_logger(__name__)


def enable_aeb_space(session):
    '''Enable aeb_space to session use Lab's data-monitor and analysis modules'''
    session.aeb_space = AEBSpace(session.spec)
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
    Body of an agent inside an environment, it:
    - enables the automatic dimension inference for constructing network input/output
    - acts as reference bridge between agent and environment (useful for multi-agent, multi-env)
    - acts as non-gradient variable storage for monitoring and analysis
    '''

    def __init__(self, env, agent_spec, aeb=(0, 0, 0), aeb_space=None):
        # essential reference variables
        self.agent = None  # set later
        self.env = env
        self.aeb = aeb
        self.a, self.e, self.b = aeb
        self.nanflat_a_idx, self.nanflat_e_idx = self.a, self.e

        # variables set during init_algorithm_params
        self.explore_var = np.nan  # action exploration: epsilon or tau
        self.entropy_coef = np.nan  # entropy for exploration

        # debugging/logging variables, set in train or loss function
        self.loss = np.nan
        self.mean_entropy = np.nan
        self.mean_grad_norm = np.nan

        self.ckpt_total_reward = np.nan
        self.total_reward = 0  # init to 0, but dont ckpt before end of an epi
        self.total_reward_ma = np.nan
        # store current and best reward_ma for model checkpointing and early termination if all the environments are solved
        self.best_reward_ma = -np.inf
        self.eval_reward_ma = np.nan

        # dataframes to track data for analysis.analyze_session
        # track training data per episode
        self.train_df = pd.DataFrame(columns=[
            'epi', 't', 'wall_t', 'opt_step', 'frame', 'fps', 'reward', 'reward_ma', 'loss', 'lr',
            'explore_var', 'entropy_coef', 'entropy', 'grad_norm'])
        # track eval data within run_eval. the same as train_df except for reward
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

        # set the ActionPD class for sampling action
        self.action_type = get_action_type(self.action_space)
        self.action_pdtype = agent_spec[self.a]['algorithm'].get('action_pdtype')
        if self.action_pdtype in (None, 'default'):
            self.action_pdtype = policy_util.ACTION_PDS[self.action_type][0]
        self.ActionPD = policy_util.get_action_pd_cls(self.action_pdtype, self.action_type)

    def update(self, state, action, reward, next_state, done):
        '''Interface update method for body at agent.update()'''
        if hasattr(self.env.u_env, 'raw_reward'):  # use raw_reward if reward is preprocessed
            reward = self.env.u_env.raw_reward
        if self.ckpt_total_reward is np.nan:  # init
            self.ckpt_total_reward = reward
        else:  # reset on epi_start, else keep adding. generalized for vec env
            self.ckpt_total_reward = self.ckpt_total_reward * (1 - self.epi_start) + reward
        self.total_reward = done * self.ckpt_total_reward + (1 - done) * self.total_reward
        self.epi_start = done

    def __str__(self):
        return f'body: {util.to_json(util.get_class_attr(self))}'

    def calc_df_row(self, env):
        '''Calculate a row for updating train_df or eval_df.'''
        frame = self.env.clock.get('frame')
        wall_t = env.clock.get_elapsed_wall_t()
        fps = 0 if wall_t == 0 else frame / wall_t

        # update debugging variables
        if net_util.to_check_train_step():
            grad_norms = net_util.get_grad_norms(self.agent.algorithm)
            self.mean_grad_norm = np.nan if ps.is_empty(grad_norms) else np.mean(grad_norms)

        row = pd.Series({
            # epi and frame are always measured from training env
            'epi': self.env.clock.get('epi'),
            # t and reward are measured from a given env or eval_env
            't': env.clock.get('t'),
            'wall_t': wall_t,
            'opt_step': self.env.clock.get('opt_step'),
            'frame': frame,
            'fps': fps,
            'reward': np.nanmean(self.total_reward),  # guard for vec env
            'reward_ma': np.nan,  # update outside
            'loss': self.loss,
            'lr': self.get_mean_lr(),
            'explore_var': self.explore_var,
            'entropy_coef': self.entropy_coef if hasattr(self, 'entropy_coef') else np.nan,
            'entropy': self.mean_entropy,
            'grad_norm': self.mean_grad_norm,
        }, dtype=np.float32)
        assert all(col in self.train_df.columns for col in row.index), f'Mismatched row keys: {row.index} vs df columns {self.train_df.columns}'
        return row

    def train_ckpt(self):
        '''Checkpoint to update body.train_df data'''
        row = self.calc_df_row(self.env)
        # append efficiently to df
        self.train_df.loc[len(self.train_df)] = row
        # update current reward_ma
        self.total_reward_ma = self.train_df[-analysis.MA_WINDOW:]['reward'].mean()
        self.train_df.iloc[-1]['reward_ma'] = self.total_reward_ma

    def eval_ckpt(self, eval_env, total_reward):
        '''Checkpoint to update body.eval_df data'''
        row = self.calc_df_row(eval_env)
        row['reward'] = total_reward
        # append efficiently to df
        self.eval_df.loc[len(self.eval_df)] = row
        # update current reward_ma
        self.eval_reward_ma = self.eval_df[-analysis.MA_WINDOW:]['reward'].mean()
        self.eval_df.iloc[-1]['reward_ma'] = self.eval_reward_ma

    def get_mean_lr(self):
        '''Gets the average current learning rate of the algorithm's nets.'''
        if not hasattr(self.agent.algorithm, 'net_names'):
            return np.nan
        lrs = []
        for attr, obj in self.agent.algorithm.__dict__.items():
            if attr.endswith('lr_scheduler'):
                lrs.append(obj.get_lr())
        return np.mean(lrs)

    def get_log_prefix(self):
        '''Get the prefix for logging'''
        spec = self.agent.spec
        spec_name = spec['name']
        trial_index = spec['meta']['trial']
        session_index = spec['meta']['session']
        prefix = f'Trial {trial_index} session {session_index} {spec_name}_t{trial_index}_s{session_index}'
        return prefix

    def log_metrics(self, metrics, df_mode):
        '''Log session metrics'''
        prefix = self.get_log_prefix()
        row_str = '  '.join([f'{k}: {v:g}' for k, v in metrics.items()])
        msg = f'{prefix} [{df_mode}_df metrics] {row_str}'
        logger.info(msg)

    def log_summary(self, df_mode):
        '''
        Log the summary for this body when its environment is done
        @param str:df_mode 'train' or 'eval'
        '''
        prefix = self.get_log_prefix()
        df = getattr(self, f'{df_mode}_df')
        last_row = df.iloc[-1]
        row_str = '  '.join([f'{k}: {v:g}' for k, v in last_row.items()])
        msg = f'{prefix} [{df_mode}_df] {row_str}'
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

    def __init__(self, spec):
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
                    body.log_summary('train')
            env.clock.tick(unit or ('epi' if env.done else 't'))
            end_session = not (env.clock.get() < env.clock.max_frame)
            end_sessions.append(end_session)
        return all(end_sessions)
