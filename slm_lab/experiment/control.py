'''
The control module
Creates and controls the units of SLM lab: EvolutionGraph, Experiment, Trial, Session
'''
from copy import deepcopy
from slm_lab.agent import AgentSpace
from slm_lab.env import EnvSpace
from slm_lab.experiment import analysis, search
from slm_lab.experiment.monitor import AEBSpace, InfoSpace
from slm_lab.lib import logger, util, viz
import importlib
import numpy as np
import os
import pandas as pd
import pydash as _
import torch


def init_thread_vars(info_space, spec, unit):
    '''Initialize thread variables from lab units that do not get carried over properly from master'''
    if info_space.get(unit) is None:
        info_space.tick(unit)
    if logger.to_init(info_space, spec):
        os.environ['PREPATH'] = analysis.get_prepath(info_space, spec)
        importlib.reload(logger)


class Session:
    '''
    The base unit of instantiated RL system.
    Given a spec,
    session creates agent(s) and environment(s),
    run the RL system and collect data, e.g. fitness metrics, till it ends,
    then return the session data.
    '''

    def __init__(self, spec, info_space=None):
        info_space = info_space or InfoSpace()
        init_thread_vars(info_space, spec, unit='session')
        self.spec = spec
        self.info_space = info_space
        self.coor, self.index = self.info_space.get_coor_idx(self)
        # TODO option to set rand_seed. also set np random seed
        self.torch_rand_seed = torch.initial_seed()
        self.data = None
        self.aeb_space = AEBSpace(self.spec, self.info_space)
        self.env_space = EnvSpace(self.spec, self.aeb_space)
        self.agent_space = AgentSpace(self.spec, self.aeb_space)
        logger.info(util.self_desc(self))
        self.aeb_space.init_body_space()
        self.aeb_space.post_body_init()
        logger.info(f'Initialized session {self.index}')

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env.
        Prepare self.df.
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

    def run(self):
        self.run_all_episodes()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data


class Trial:
    '''
    The base unit of an experiment.
    Given a spec and number s,
    trial creates and runs s sessions,
    gather and aggregate data from sessions as trial data,
    then return the trial data.
    '''

    def __init__(self, spec, info_space=None):
        info_space = info_space or InfoSpace()
        init_thread_vars(info_space, spec, unit='trial')
        self.spec = spec
        self.info_space = info_space
        self.coor, self.index = self.info_space.get_coor_idx(self)
        self.session_data_dict = {}
        self.data = None
        analysis.save_spec(info_space, spec, unit='trial')
        logger.info(f'Initialized trial {self.index}')

    def init_session_and_run(self, info_space):
        session = Session(self.spec, info_space)
        session_data = session.run()
        return session_data

    def close(self):
        logger.info('Trial done, closing.')

    def run(self):
        info_spaces = []
        for _s in range(self.spec['meta']['max_session']):
            self.info_space.tick('session')
            info_spaces.append(deepcopy(self.info_space))
        if self.spec['meta']['train_mode'] and len(info_spaces) > 1:
            session_datas = util.parallelize_fn(
                self.init_session_and_run, info_spaces)
        else:  # dont parallelize when debugging to allow render
            session_datas = [
                self.init_session_and_run(info_space) for info_space in info_spaces
            ]
        self.session_data_dict = {
            data.index[0]: data for data in session_datas}
        self.data = analysis.analyze_trial(self)
        self.close()
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

    def __init__(self, spec, info_space=None):
        info_space = info_space or InfoSpace()
        init_thread_vars(info_space, spec, unit='experiment')
        self.spec = spec
        spec['meta']['train_mode'] = True
        self.info_space = info_space
        self.coor, self.index = self.info_space.get_coor_idx(self)
        self.trial_data_dict = {}
        self.data = None
        SearchClass = getattr(search, 'RaySearch')
        self.search = SearchClass(self)
        analysis.save_spec(info_space, spec, unit='experiment')
        logger.info(f'Initialized experiment {self.index}')

    def init_trial_and_run(self, spec, info_space):
        '''
        Method to run trial with the properly updated info_space (trial_index) from experiment.RaySearch.lab_trial.
        Do not tick info_space below, it is already updated when passed from lab_trial.
        '''
        trial = Trial(spec, info_space)
        trial_data = trial.run()
        return trial_data

    def close(self):
        logger.info('Experiment done, closing.')

    def run(self):
        self.trial_data_dict = self.search.run()
        self.data = analysis.analyze_experiment(self)
        self.close()
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
