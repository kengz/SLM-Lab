'''
The control module
Creates and controls the units of SLM lab: EvolutionGraph, Experiment, Trial, Session
'''
from slm_lab.agent import AgentSpace
from slm_lab.env import EnvSpace
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
        self.aeb_space = AEBSpace(self.spec)
        self.env_space = EnvSpace(self.spec, self.aeb_space)
        self.agent_space = AgentSpace(self.spec, self.aeb_space)
        logger.info(util.self_desc(self))
        self.aeb_space.init_body_space()
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

    def run_all_episodes(self):
        '''
        Run all episodes, where each env can step and reset at its own clock_speed and timeline. Will terminate when all envs done running max_episode.
        '''
        _reward_space, state_space, _done_space = self.env_space.reset()
        self.agent_space.reset(state_space)
        while True:
            end_session = self.aeb_space.tick_clocks()
            if end_session:
                break
            action_space = self.agent_space.act(state_space)
            reward_space, state_space, done_space = self.env_space.step(
                action_space)
            self.agent_space.update(
                action_space, reward_space, state_space, done_space)
        # TODO collect data from different clock speed
        episode_data = {}
        return episode_data

    def run(self):
        self.run_all_episodes()
        # TODO resore viz
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
