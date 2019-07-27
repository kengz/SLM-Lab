# The agent module
from slm_lab.agent import algorithm, memory
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util, viz
from slm_lab.lib.decorator import lab_api
import numpy as np
import pandas as pd
import pydash as ps
import torch
import warnings


logger = logger.get_logger(__name__)


class Agent:
    '''
    Agent abstraction; implements the API to interface with Env in SLM Lab
    Contains algorithm, memory, body
    '''

    def __init__(self, spec, body, global_nets=None):
        self.spec = spec
        self.agent_spec = spec['agent'][0]  # idx 0 for single-agent
        self.name = self.agent_spec['name']
        assert not ps.is_list(global_nets), f'single agent global_nets must be a dict, got {global_nets}'
        # set components
        self.body = body
        body.agent = self
        MemoryClass = getattr(memory, ps.get(self.agent_spec, 'memory.name'))
        self.body.memory = MemoryClass(self.agent_spec['memory'], self.body)
        AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
        self.algorithm = AlgorithmClass(self, global_nets)

        logger.info(util.self_desc(self))

    @lab_api
    def act(self, state):
        '''Standard act method from algorithm.'''
        with torch.no_grad():  # for efficiency, only calc grad in algorithm.train
            action = self.algorithm.act(state)
        return action

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''
        self.body.update(state, action, reward, next_state, done)
        if util.in_eval_lab_modes():  # eval does not update agent for training
            return
        self.body.memory.update(state, action, reward, next_state, done)
        loss = self.algorithm.train()
        if not np.isnan(loss):  # set for log_summary()
            self.body.loss = loss
        explore_var = self.algorithm.update()
        return loss, explore_var

    @lab_api
    def save(self, ckpt=None):
        '''Save agent'''
        if util.in_eval_lab_modes():  # eval does not save new models
            return
        self.algorithm.save(ckpt=ckpt)

    @lab_api
    def close(self):
        '''Close and cleanup agent at the end of a session, e.g. save model'''
        self.save()


class Body:
    '''
    Body of an agent inside an environment, it:
    - enables the automatic dimension inference for constructing network input/output
    - acts as reference bridge between agent and environment (useful for multi-agent, multi-env)
    - acts as non-gradient variable storage for monitoring and analysis
    '''

    def __init__(self, env, agent_spec, aeb=(0, 0, 0)):
        # essential reference variables
        self.agent = None  # set later
        self.env = env
        # agent, env, body index for multi-agent-env
        self.a, self.e, self.b = self.aeb = aeb

        # variables set during init_algorithm_params
        self.explore_var = np.nan  # action exploration: epsilon or tau
        self.entropy_coef = np.nan  # entropy for exploration

        # debugging/logging variables, set in train or loss function
        self.loss = np.nan
        self.mean_entropy = np.nan
        self.mean_grad_norm = np.nan

        # total_reward_ma from eval for model checkpoint saves
        self.best_total_reward_ma = -np.inf
        self.total_reward_ma = np.nan

        # dataframes to track data for analysis.analyze_session
        # track training data per episode
        self.train_df = pd.DataFrame(columns=[
            'epi', 't', 'wall_t', 'opt_step', 'frame', 'fps', 'total_reward', 'total_reward_ma', 'loss', 'lr',
            'explore_var', 'entropy_coef', 'entropy', 'grad_norm'])
        # track eval data within run_eval. the same as train_df except for reward
        self.eval_df = self.train_df.copy()

        # the specific agent-env interface variables for a body
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.observable_dim = self.env.observable_dim
        self.state_dim = self.observable_dim['state']
        self.action_dim = self.env.action_dim
        self.is_discrete = self.env.is_discrete
        # set the ActionPD class for sampling action
        self.action_type = policy_util.get_action_type(self.action_space)
        self.action_pdtype = agent_spec[self.a]['algorithm'].get('action_pdtype')
        if self.action_pdtype in (None, 'default'):
            self.action_pdtype = policy_util.ACTION_PDS[self.action_type][0]
        self.ActionPD = policy_util.get_action_pd_cls(self.action_pdtype, self.action_type)

    def update(self, state, action, reward, next_state, done):
        '''Interface update method for body at agent.update()'''
        pass

    def __str__(self):
        return f'body: {util.to_json(util.get_class_attr(self))}'

    def calc_df_row(self, env):
        '''Calculate a row for updating train_df or eval_df.'''
        frame = self.env.clock.get('frame')
        wall_t = env.clock.get_elapsed_wall_t()
        fps = 0 if wall_t == 0 else frame / wall_t
        with warnings.catch_warnings():  # mute np.nanmean warning
            warnings.filterwarnings('ignore')
            total_reward = np.nanmean(env.total_reward)  # guard for vec env

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
            'total_reward': total_reward,
            'total_reward_ma': np.nan,  # update outside
            'loss': self.loss,
            'lr': self.get_mean_lr(),
            'explore_var': self.explore_var,
            'entropy_coef': self.entropy_coef if hasattr(self, 'entropy_coef') else np.nan,
            'entropy': self.mean_entropy,
            'grad_norm': self.mean_grad_norm,
        }, dtype=np.float32)
        assert all(col in self.train_df.columns for col in row.index), f'Mismatched row keys: {row.index} vs df columns {self.train_df.columns}'
        return row

    def ckpt(self, env, df_mode):
        '''
        Checkpoint to update body.train_df or eval_df data
        @param OpenAIEnv|UnityEnv:env self.env or self.eval_env
        @param str:df_mode 'train' or 'eval'
        '''
        row = self.calc_df_row(env)
        df = getattr(self, f'{df_mode}_df')
        df.loc[len(df)] = row  # append efficiently to df
        df.iloc[-1]['total_reward_ma'] = total_reward_ma = df[-viz.PLOT_MA_WINDOW:]['total_reward'].mean()
        if df_mode == 'eval':
            self.total_reward_ma = total_reward_ma

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
