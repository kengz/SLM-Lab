import numpy as np
import os
import pandas as pd
import pydash as ps
import torch
import warnings
import copy

from collections import Iterable, OrderedDict
from torch.utils.tensorboard import SummaryWriter

from slm_lab.agent.agent import agent_util, observability
from slm_lab.agent import algorithm, memory, agent
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util, viz
from slm_lab.lib.decorator import lab_api




logger = logger.get_logger(__name__)

# ma for mean average
BASIC_COLS = ['epi', 't', 'wall_t', 'frame', 'fps', 'tot_r', 'tot_r_ma', 'lr']

# class Agent(observability.ObservableAgentInterface):
class Agent(object):

    '''
    Agent abstraction; implements the API to interface with Env in SLM Lab
    Contains algorithm, memory, body

    world < env
            + welfare_function
            + agents < algorithm + memory + body

    '''

    def __init__(self, spec, env, world, global_nets=None, agent_idx=0):
        logger.debug("Start Agent __init__")
        self.world = world
        self.spec = spec
        self.agent_spec = spec['agent']
        self.name = self.agent_spec['name']
        self.playing_agent = self.agent_spec['playing_agent'] if 'playing_agent' in self.agent_spec.keys() else True
        assert not ps.is_list(global_nets), f'single agent global_nets must be a dict, got {global_nets}'
        self.agent_idx = agent_idx

        ###### Set components ######

        # Welfare function (set if define in agent (can be overwritten by algorithms)
        self.welfare_function = getattr(agent_util, ps.get(self.agent_spec, 'welfare_function',
                                                           default="default_welfare"))
        # Body
        body = Body(env, spec, aeb=(agent_idx, 0, 0))
        body.agent = self
        self.body = body



        # World co-living agents awareness
        # self.other_ag_observations = OrderedDict()
        self.observed_agent_mode = ps.get(self.agent_spec, 'observing_other_agents.name', default=None)
        if self. observed_agent_mode is None or self.observed_agent_mode == "NotObservable":
            # self.observed_agents = None
            self.observable = False
        elif self.observed_agent_mode == "FullyObservable":
            # self.observed_agents = self._observed_agents
            self.observable = True
        else:
            raise NotImplementedError()

        # print("self.observed_agents", len(self.observed_agents()))
        # assert 0

        # Algorithm
        AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
        self.algorithm = AlgorithmClass(self, global_nets,
                                         algorithm_spec=ps.get(self.agent_spec, 'algorithm'),
                                         memory_spec=ps.get(self.agent_spec, 'memory'),
                                         net_spec=ps.get(self.agent_spec, 'net'))

        self.body.init_part2()



        logger.info("self.name {}".format(self.name))
        logger.info(util.self_desc(self))
        logger.debug("End Agent __init__")


    @property
    def observed_agents(self):
        # print("self.world.agents", len(self.world.agents))
        return [ agent for ag_idx, agent in enumerate(self.world.agents)
                 if ag_idx != self.agent_idx]

    @lab_api
    def act(self, state):
        '''Standard act method from algorithm.'''
        with torch.no_grad():  # for efficiency, only calc grad in algorithm.train
            action, action_pd = self.algorithm.act(state)
            self.action = action
            self.action_pd = action_pd
            self.state = state
        return self.action, self.action_pd

    @lab_api
    def act_after_env_step(self):
        '''Standard act method from algorithm.'''
        if hasattr(self.algorithm, "act_after_env_step"):
            with torch.no_grad():  # for efficiency, only calc grad in algorithm.train
                retro_action = self.algorithm.act_after_env_step()
            return retro_action
        else:
            return {}

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''

        self.body.reward = reward

        # Still usefull to allow for player to modify the reward (give/take reward to other)
        welfare = self.welfare_function(self, reward)
        self.reward = reward
        self.welfare = welfare
        self.next_state = next_state
        self.done = done

        self.body.update(state, action, welfare, next_state, done)
        if util.in_eval_lab_modes():  # eval does not update agent for training
            return
        self.algorithm.memory_update(state, action, self.welfare, next_state, done)

    def set_step_values(self, reward, next_state, done):
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def compute_welfare(self):
        welfare = self.welfare_function(self, self.reward)
        self.welfare = welfare

    def train(self):
        loss = self.algorithm.train()
        # TODO confusing that agent.train calls algo.update (since agent also has the update method)
        explore_var = self.algorithm.update()
        return loss, explore_var

    def observe_other_agents(self):
        pass
        #TODO this is now useless ?
        # if self.observed_agent_mode is not None:
        #     self.other_ag_observations = OrderedDict()
        #     for observed_agent in self.observed_agents:
        #         if observed_agent.agent_idx != self.agent_idx and observed_agent.observable:
        #             # TODO rm str()
        #             print("oberseved", observed_agent.algorithm)
        #             self.other_ag_observations[str(observed_agent.agent_idx)] = {
        #                                                             "state": observed_agent.state,
        #                                                             "action": observed_agent.action,
        #                                                             "action_pd": observed_agent.action_pd,
        #                                                             "reward": observed_agent.reward,
        #                                                             "welfare": observed_agent.welfare,
        #                                                             "next_state": observed_agent.next_state,
        #                                                             "done": observed_agent.done,
        #                                                             "algorithm": observed_agent.algorithm,
        #                                                           }

            # print("self.other_ag_observations",self.other_ag_observations)

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

    @lab_api
    def apply_feedback(self, state, action, reward, next_state, done, info, additional_list_of_dict):
        """No modifications to apply by default"""
        (state, action, reward,
         next_state, done, info,
         additional_list_of_dict) = self.algorithm.apply_feedback(state, action, reward,
                                                                  next_state, done, info,
                                                                  additional_list_of_dict)

        return state, action, reward, next_state, done, info, additional_list_of_dict

    # @property
    # def reward(self):
    #     return agent_util.get_from_current_agents(self, key="reward")
    #
    # @property
    # def action(self):
    #     return agent_util.get_from_current_agents(self, key="action")

    # TODO add this in observability API
    # @property
    # def action_pd(self):
    #     return self.action_pd

    # @property
    # def state(self):
    #     return agent_util.get_from_current_agents(self, key="state")
    #
    # # @property
    # # def welfare(self):
    # #     return agent_util.get_from_current_agents(self, key="welfare")
    #
    # @property
    # def next_state(self):
    #     return agent_util.get_from_current_agents(self, key="next_state")
    #
    # @property
    # def done(self):
    #     return agent_util.get_from_current_agents(self, key="done")
    #
    # @property
    # def algorithm(self):
    #     return self._algorithm

class Body:
    # TODO change the body doc to reflect the new use in a world (as well as in an agent)
    '''
    Body of an agent inside an environment, it:
    - enables the automatic dimension inference for constructing network input/output
    - acts as reference bridge between agent and environment (useful for multi-agent, multi-env)
    - acts as non-gradient variable storage for monitoring and analysis
    '''

    def __init__(self, env, spec, aeb=(0, 0, 0)):
        # essential reference variables
        self.agent = None  # set later
        self.env = env
        self.spec = spec
        # agent, env, body index for multi-agent-env
        self.a, self.e, self.b = self.aeb = aeb

        # variables set during init_algorithm_params
        # self.explore_var = np.nan  # action exploration: epsilon or tau
        # self.entropy_coef = np.nan  # entropy for exploration

        # debugging/logging variables, set in train or loss function
        # self.loss = np.nan
        # self.mean_entropy = np.nan
        # self.mean_grad_norm = np.nan

        # total_reward_ma from eval for model checkpoint saves
        self.best_total_reward_ma = -np.inf
        self.total_reward_ma = np.nan

        self.reward = np.nan
        self.welfare = np.nan

        # the specific agent-env interface variables for a body
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.observation_dim = self.env.observable_dim
        logger.info("self.observable_dim {}".format(self.observation_dim))
        self.action_dim = self.env.action_dim
        self.action_space_is_discrete = self.env.action_space_is_discrete
        # set the ActionPD class for sampling action
        self.action_type = policy_util.get_space_type(self.action_space)
        self.action_pdtype = ps.get(spec, f'agent.{self.a}.algorithm.action_pdtype')
        if self.action_pdtype in (None, 'default'):
            self.action_pdtype = policy_util.ACTION_PDS[self.action_type][0]
        self.ActionPD = policy_util.get_action_pd_cls(self.action_pdtype, self.action_type)

        self.tb_add_graph = False

    def init_part2(self):
        # TODO merge in init
        # dataframes to track data for analysis.analyze_session
        # track training data per episode
        cols = copy.deepcopy(BASIC_COLS)
        # if self.agent is not None :
        #     cols += self.agent.algorithm.extra_training_log_info_col
        # if self.env is not None:
        #     print("self.env.extra_env_log_info_col", self.env.extra_env_log_info_col)
        #     cols += self.env.extra_env_log_info_col
        self.train_df = pd.DataFrame(columns=cols)
        # track eval data within run_eval. the same as train_df except for reward
        if ps.get(self.spec, 'meta.rigorous_eval'):
            self.eval_df = self.train_df.copy()
        else:
            self.eval_df = self.train_df

    def update(self, state, action, reward, next_state, done):
        '''Interface update method for body at agent.update()'''
        if util.get_lab_mode() == 'dev':  # log tensorboard only on dev mode
            self.track_tensorboard(action)

    def __str__(self):
        class_attr = util.get_class_attr(self)
        class_attr.pop('spec')
        return f'body: {util.to_json(class_attr)}'

    def calc_df_row(self, env):
        '''Calculate a row for updating train_df or eval_df.'''
        frame = self.env.clock.frame
        wall_t = self.env.clock.wall_t
        fps = 0 if wall_t == 0 else frame / wall_t
        with warnings.catch_warnings():  # mute np.nanmean warning
            warnings.filterwarnings('ignore')
            if isinstance(env.total_reward, Iterable):
                # print("env.total_reward", env.total_reward)
                ## TODO useless mean ?
                total_reward = np.nanmean(env.total_reward,
                                      axis=tuple(range(1, env.total_reward.ndim, 1)))  # guard for vec env
                # print("total_reward",total_reward)

                if self.agent is None:
                    total_reward = total_reward.sum()
                    # print("world")
                else:
                    total_reward = total_reward[self.agent.agent_idx] if self.agent.playing_agent else np.nan
                #     print("self.agent.agent_idx",self.agent.agent_idx)
                # print("total_reward 2",total_reward)

            else:
                total_reward = np.nanmean(env.total_reward)

        # update debugging variables
        # TODO to be added by algo
        # if net_util.to_check_train_step():
        #     grad_norms = net_util.get_grad_norms(self.agent.algorithm) if self.agent is not None else []
        #     self.mean_grad_norm = np.nan if ps.is_empty(grad_norms) else np.mean(grad_norms)
        if net_util.to_check_train_step():
            if self.agent is not None:
                self.agent.algorithm.log_grad_norm()


        row_dict = {
            # epi and frame are always measured from training env
            'epi': self.env.clock.epi,
            # t and reward are measured from a given env or eval_env
            't': env.clock.t,
            'wall_t': wall_t,
            # 'opt_step': self.env.clock.opt_step,
            'opt_step': np.nan, #self.agent.algorithm.net.opt_step if self.agent is not None else -1,
            'frame': frame,
            'fps': fps,
            # 'reward': total_reward,
            # 'reward_ma': np.nan,  # update outside
            'tot_r': total_reward,
            'tot_r_ma': np.nan,  # update outside
            # 'loss': self.loss,
            # 'lr': self.get_mean_lr(),
            # 'explore_var': self.explore_var,
            # 'expl_var': self.agent.algorithm.explore_var_scheduler.val if
            #                 self.agent is not None and
            #                 hasattr(self.agent.algorithm, 'explore_var_scheduler')
            #                 else np.nan,
            # # 'entropy_coef': self.entropy_coef if hasattr(self, 'entropy_coef') else np.nan,
            # 'entp_coef': self.agent.algorithm.entropy_coef_scheduler.val if
            #                 self.agent is not None and
            #                 hasattr(self.agent.algorithm, 'entropy_coef_scheduler')
            #                 else np.nan,
            # 'entropy': self.mean_entropy,
            # 'grad_norm': self.mean_grad_norm,
        }
        if self.agent is not None :
            row_dict.update(self.agent.algorithm.get_log_values())
        if self.env is not None:
            row_dict.update(self.env.get_extra_training_log_info())
        # print("row_dict",row_dict)
        # for k, v in row_dict.items():
        #     print(k,v, type(v))
        row = pd.Series(row_dict, dtype=np.float32)
        # assert all(col in self.train_df.columns for col in row.index), f'Mismatched row keys: {row.index} vs df columns {self.train_df.columns}'
        return row

    def ckpt(self, env, df_mode):
        '''
        Checkpoint to update body.train_df or eval_df data
        @param OpenAIEnv:env self.env or self.eval_env
        @param str:df_mode 'train' or 'eval'
        '''
        row = self.calc_df_row(env)

        df = getattr(self, f'{df_mode}_df')
        # Dynamicaly add new columns
        for col in row.index:
            if col not in df.columns:
                logger.info(f"Add col {col} in {df_mode}_df")
                df[col] = [np.nan] * len(df)

        df.loc[len(df)] = row  # append efficiently to df
        df.iloc[-1]['tot_r_ma'] = total_reward_ma = df[-viz.PLOT_MA_WINDOW:]['tot_r'].mean()
        self.total_reward_ma = total_reward_ma

    def get_mean_lr(self, algorithm=None):
        '''Gets the average current learning rate of the algorithm's nets.'''
        if algorithm is None:
            if self.agent is None:
                return np.nan
            algorithm = self.agent.algorithm

        if hasattr(algorithm, "algorithms"): # Support nested algorithm
            return np.mean([self.get_mean_lr(algo) for algo in algorithm.algorithms])

        if not hasattr(algorithm, 'net_names'):
            return np.nan

        lrs = []
        for attr, obj in algorithm.__dict__.items():
            if attr.endswith('lr_scheduler'):
                lrs.append(obj.get_lr())
        return np.mean(lrs)

    def get_log_prefix(self):
        '''Get the prefix for logging'''
        # spec = self.agent.spec
        spec = self.spec
        spec_name = spec['name']
        trial_index = spec['meta']['trial']
        session_index = spec['meta']['session']
        prefix = f'T{trial_index}S{session_index}Ag{self.aeb[0]} {spec_name}'
        return prefix

    # def log_metrics(self, metrics, df_mode):
    #     '''Log session metrics'''
    #     # prefix = self.get_log_prefix()
    #     row_str = '  '.join([f'{k}: {v:g}' for k, v in metrics.items()])
    #     # msg = f'{prefix} [{df_mode}_df metrics] {row_str}'
    #     msg = f'[{df_mode} metrics] {row_str}'
    #     logger.info(msg)

    def log_summary(self, df_mode):
        '''
        Log the summary for this body when its environment is done
        @param str:df_mode 'train' or 'eval'
        '''
        # prefix = self.get_log_prefix()
        df = getattr(self, f'{df_mode}_df')
        last_row = {"algo":self.agent.algorithm.name if self.agent is not None else None}
        last_row.update(df.iloc[-1])
        # if isinstance(v, np.float32) else f'{k}: {v}'
        # row_str = '  '.join([f'{k}: {v:g}'  for k, v in last_row.items()])
        # msg = f'{prefix} [{df_mode}] {row_str}'
        # logger.info(msg)
        if util.get_lab_mode() == 'dev' and df_mode == 'train':  # log tensorboard only on dev mode and train df data
            if self.agent is not None:
                self.log_tensorboard()
        return last_row

    def log_tensorboard(self):
        '''
        Log summary and useful info to TensorBoard.
        NOTE this logging is comprehensive and memory-intensive, hence it is used in dev mode only
        '''
        # initialize TensorBoard writer
        if not hasattr(self, 'tb_writer'):
            log_prepath = self.spec['meta']['log_prepath']
            if self.aeb != (0, 0, 0):
                log_prepath = os.path.dirname(log_prepath) + f'aeb{self.aeb}_' + os.path.basename(log_prepath)
                # TODO Improvement: it would be better that this would be created in the world and then shared with each agent (this way it will create only one tensorboard log file)
            self.tb_writer = SummaryWriter(os.path.dirname(log_prepath), filename_suffix=os.path.basename(log_prepath))
            self.tb_actions = []  # store actions for tensorboard
            logger.info(f'Using TensorBoard logging for dev mode. Run `tensorboard --logdir={log_prepath}` to start TensorBoard.')

        trial_index = self.agent.spec['meta']['trial']
        session_index = self.agent.spec['meta']['session']
        if session_index != 0:  # log only session 0
            return
        idx_suffix = f'trial{trial_index}_session{session_index}_agent{self.agent.agent_idx}'
        frame = self.env.clock.frame
        # add summary variables
        last_row = self.train_df.iloc[-1]
        for k, v in last_row.items():
            self.tb_writer.add_scalar(f'{k}/{idx_suffix}', v, frame)

        self._tb_log_net(self.agent.algorithm, idx_suffix, frame)

        # add action histogram and flush
        if not ps.is_empty(self.tb_actions):
            actions = np.array(self.tb_actions)
            if len(actions.shape) == 1:
                self.tb_writer.add_histogram(f'action/{idx_suffix}', actions, frame)
            else:  # multi-action
                for idx, subactions in enumerate(actions.T):
                    self.tb_writer.add_histogram(f'action.{idx}/{idx_suffix}', subactions, frame)
            self.tb_actions = []

    def _tb_log_net(self, algorithm, idx_suffix, frame):
        if hasattr(algorithm, "algorithms"):
            for algo in algorithm.algorithms: # Support nested algorithms
                self._tb_log_net(algo, idx_suffix, frame)
                return
        # add main graph
        if self.tb_add_graph and self.env.clock.frame == 0 and hasattr(algorithm, 'net'):
            # can only log 1 net to tb now, and 8 is a good common length for stacked and rnn inputs
            net = algorithm.net
            self.tb_writer.add_graph(net, torch.rand(ps.flatten([8, net.in_dim])))
        # add network parameters
        for net_name in algorithm.net_names:
            if net_name.startswith('global_') or net_name.startswith('target_'):
                continue
            net = getattr(algorithm, net_name)
            for name, params in net.named_parameters():
                self.tb_writer.add_histogram(f'{net_name}.{name}/{idx_suffix}_algo{algorithm.algo_idx}', params, frame)

    def track_tensorboard(self, action):
        '''Helper to track variables for tensorboard logging'''
        if self.env.is_venv:
            self.tb_actions.extend(action.tolist())
        else:
            self.tb_actions.append(action)
