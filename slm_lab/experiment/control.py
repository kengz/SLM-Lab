'''
The control module
Creates and controls the units of SLM lab: EvolutionGraph, Experiment, Trial, Session
'''
from slm_lab.agent import Agent, AgentSpace
from slm_lab.env import Env, EnvSpace
from slm_lab.experiment.monitor import info_space, AEBSpace, get_body_df_dict
from slm_lab.lib import logger, util, viz
import numpy as np
import pandas as pd
import pydash as _


class Session:
    '''
    The base unit of instantiated RL system.
    Given a spec,
    session creates agent(s) and environment(s),
    run the RL system and collect data, e.g. fitness metrics, till it ends,
    then return the session data.
    '''

    def __init__(self, spec):
        self.spec = spec
        self.coor, self.index = info_space.index_lab_comp(self)
        self.data = pd.DataFrame()
        # TODO put resolved space from spec into monitor.info_space
        self.aeb_space = AEBSpace(self.spec)
        self.env_space = EnvSpace(self.spec, self.aeb_space)
        self.agent_space = AgentSpace(self.spec, self.aeb_space)
        self.aeb_space.init_body_space()
        print(self.aeb_space.body_space)
        self.aeb_space.post_body_init()

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env.
        Prepare self.data.
        '''
        self.agent_space.close()
        self.env_space.close()
        logger.info('Session done, closing.')

    def run_episode(self):
        '''
        Main RL loop, runs a single episode over timesteps, generalized to spaces from singleton.
        Returns episode_data space.
        '''
        self.aeb_space.clock.tick('e')
        logger.info(f'Running episode {self.aeb_space.clock.get("e")}')
        # TODO generalize and make state to include observables
        state_space = self.env_space.reset()
        self.agent_space.reset(state_space)
        # RL steps for SARS
        loss_list = []
        explore_var_list = []
        for t in range(self.env_space.max_timestep):
            self.aeb_space.clock.tick('t')
            # TODO tick body clock
            # TODO common refinement of timestep
            # TODO ability to train more on harder environments, or specify update per timestep per body, ratio of data u use to train. something like train_per_new_mem
            action_space = self.agent_space.act(state_space)
            logger.debug(f'action_space {action_space}')
            (reward_space, state_space,
             done_space) = self.env_space.step(action_space)
            logger.debug(
                f'reward_space: {reward_space}, state_space: {state_space}, done_space: {done_space}')
            # completes cycle of full info for agent_space
            # TODO tmp return, to unify with monitor auto-fetch later
            loss, explore_var = self.agent_space.update(
                action_space, reward_space, state_space, done_space)
            if loss is not None:
                loss_list.append(loss)
            explore_var_list.append(explore_var)
            # TODO hack for a reliable done, otherwise all needs to be coincidental
            # if bool(done_space):
            if done_space.get(a=0)[(0,0)]:
                # TODO refactor: set all to terminate on master termination. Also use the env with longest timestep to prevent being terminated by fast-running env
                done_space.data.fill(1)
                break
        # TODO monitor record all data spaces, including body with body.clock. cuz all data spaces have history
        # split per body, use done as delim (maybe done need body clock now), split, sum each chunk
        mean_loss = np.nanmean(loss_list)
        mean_explore_var = np.nanmean(explore_var_list)
        # print(self.aeb_space.data_spaces['reward'])
        # print(self.aeb_space.data_spaces['reward'].data_history)
        body_df_dict = get_body_df_dict(self.aeb_space)
        # logger.info(
        #     f'epi {self.aeb_space.clock.get("e")}, total_rewards {total_rewards}')
        # TODO compose episode data properly with monitor
        episode_data = {
            'mean_loss': mean_loss,
            'mean_explore_var': mean_explore_var,
            'body_df_dict': body_df_dict,
        }
        # episode_data = pd.DataFrame(
        #     episode_data_list, columns=['rewards', 'total_rewards', 'loss', 'explore_var'])
        # episode_data = {}
        return episode_data

    def run(self):
        body_df_dict = None
        epi_loss_list = []
        epi_explore_var_list = []
        for e in range(_.get(self.spec, 'meta.max_episode')):
            logger.debug(f'episode {e}')
            episode_data = self.run_episode()
            epi_loss_list.append(episode_data['mean_loss'])
            epi_explore_var_list.append(episode_data['mean_explore_var'])
            # collected over absolute time, so just get at epi end
            body_df_dict = episode_data['body_df_dict']
        # TODO tmp hack. fix with monitor data later
        for k, body_df in body_df_dict.items():
            done_list = body_df['done'].tolist()
            # fix offset in cumsum (True entry belongs to the chunk before it)
            done_list.insert(0, False)
            done_list.pop()
            body_df['e'] = pd.Series(done_list).cumsum()
            agg_body_df = body_df[['e', 'reward']].groupby('e').agg('sum')
            body_df_dict[k] = agg_body_df

        loss_df = pd.DataFrame(
            {'loss': epi_loss_list, 'explore_var': epi_explore_var_list})

        fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)

        loss_fig = viz.plot_line(
            loss_df, ['loss'], y2_col=['explore_var'], draw=False)
        fig.append_trace(loss_fig.data[0], 1, 1)
        fig.append_trace(loss_fig.data[1], 2, 1)

        for k, body_df in body_df_dict.items():
            body_fig = viz.plot_line(
                body_df, 'reward', 'e', legend_name=str(k), draw=False)
            fig.append_trace(body_fig.data[0], 3, 1)

        fig.layout['yaxis1'].update(loss_fig.layout['yaxis'])
        fig.layout['yaxis1'].update(domain=[0.55, 1])
        fig.layout['yaxis2'].update(loss_fig.layout['yaxis2'])
        fig.layout['yaxis2'].update(showgrid=False)

        fig.layout['yaxis3'].update(body_fig.layout['yaxis'])
        fig.layout['yaxis3'].update(domain=[0, 0.45])
        fig.layout.update(_.pick(loss_fig.layout, ['legend']))
        fig.layout.update(_.pick(body_fig.layout, ['legend']))
        fig.layout.update(title=self.spec['name'], width=500, height=600)
        viz.plot(fig)
        viz.save_image(fig)

        self.close()
        # TODO session data checker method
        return self.data


class Trial:
    '''
    The base unit of an experiment.
    Given a spec and number s,
    trial creates and runs s sessions,
    gather and aggregate data from sessions as trial data,
    then return the trial data.
    '''

    def __init__(self, spec):
        self.spec = spec
        self.coor, self.index = info_space.index_lab_comp(self)
        self.data = pd.DataFrame()
        self.session = None

    def init_session(self):
        self.session = Session(self.spec)
        return self.session

    def close(self):
        return

    def run(self):
        for s in range(_.get(self.spec, 'meta.max_session')):
            logger.debug(f'session {s}')
            self.init_session().run()
        self.close()
        # TODO trial data checker method
        return self.data


class Experiment:
    '''
    The core high level unit of Lab.
    Given a spec-space/generator of cardinality t,
    a number s,
    a hyper-optimization algorithm hopt(spec, fitness-metric) -> spec_next/null
    experiment creates and runs up to t trials of s sessions each to optimize (maximize) the fitness metric,
    gather the trial data,
    then return the experiment data for analysis and use in evolution graph.
    Experiment data will include the trial data, notes on design, hypothesis, conclusion, analysis data, e.g. fitness metric, evolution link of ancestors to potential descendants.
    An experiment then forms a node containing its data in the evolution graph with the evolution link and suggestion at the adjacent possible new experiments
    On the evolution graph level, an experiment and its neighbors could be seen as test/development of traits.
    '''
    # TODO metaspec to specify specs to run, can be sourced from evolution suggestion

    def __init__(self, spec):
        self.spec = spec
        self.coor, self.index = info_space.index_lab_comp(self)
        self.data = pd.DataFrame()
        self.trial = None

    def init_trial(self):
        self.trial = Trial(self.spec)
        return self.trial

    def close(self):
        return

    def run(self):
        for t in range(_.get(self.spec, 'meta.max_trial')):
            logger.debug(f'trial {t}')
            self.init_trial().run()
        self.close()
        # TODO exp data checker method
        return self.data


class EvolutionGraph:
    '''
    The biggest unit of Lab.
    The evolution graph keeps track of all experiments as nodes of experiment data, with fitness metrics, evolution links, traits,
    which could be used to aid graph analysis on the traits, fitness metrics,
    to suggest new experiment via node creation, mutation or combination (no DAG restriction).
    There could be a high level evolution module that guides and optimizes the evolution graph and experiments to achieve SLM.
    '''
    pass
