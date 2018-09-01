'''
The control module
Creates and controls the units of SLM lab: EvolutionGraph, Experiment, Trial, Session
'''
from copy import deepcopy
from slm_lab.agent import AgentSpace, Agent, Body
from slm_lab.env import EnvSpace, make_env
from slm_lab.experiment import analysis, search
from slm_lab.experiment.monitor import AEBSpace, enable_aeb_space
from slm_lab.lib import logger, util
import numpy as np
import os
import pandas as pd
import pydash as ps
import torch
import torch.multiprocessing as mp


class Session:
    '''
    The base unit of instantiated RL system.
    Given a spec,
    session creates agent(s) and environment(s),
    run the RL system and collect data, e.g. fitness metrics, till it ends,
    then return the session data.
    '''

    def __init__(self, spec, info_space, global_nets=None):
        self.spec = spec
        self.info_space = info_space
        self.index = self.info_space.get('session')
        util.set_module_seed(self.info_space.get_random_seed())
        self.data = None

        self.env = make_env(self.spec)
        body = Body(self.env, self.spec['agent'])
        self.agent = Agent(self.spec, self.info_space, body)

        # TODO move outside, and properly pick singleton or multiagent session
        enable_aeb_space(self)
        logger.info(util.self_desc(self))
        logger.info(f'Initialized session {self.index}')

    def save_if_ckpt(cls, agent, env):
        '''Save for agent, env if episode is at checkpoint'''
        agent.body.log_summary()
        epi = env.clock.get('epi')
        save_this_epi = epi > 0 and hasattr(env, 'save_epi_frequency') and epi % env.save_epi_frequency == 0
        if save_this_epi:
            agent.save(epi=epi)

    def run_episode(self):
        self.env.clock.tick('epi')
        reward, state, done = self.env.reset()
        self.agent.reset(state)
        while not done:
            self.env.clock.tick('t')
            action = self.agent.act(state)
            reward, state, done = self.env.step(action)
            self.agent.update(action, reward, state, done)
        self.save_if_ckpt(self.agent, self.env)

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env.
        '''
        self.agent.close()
        self.env.close()
        logger.info('Session done, closing.')

    def run(self):
        while self.env.clock.get('epi') <= self.env.max_episode:
            self.run_episode()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data


class SpaceSession:
    '''
    The base unit of instantiated RL system.
    Given a spec,
    session creates agent(s) and environment(s),
    run the RL system and collect data, e.g. fitness metrics, till it ends,
    then return the session data.
    '''

    def __init__(self, spec, info_space, global_nets=None):
        self.spec = spec
        self.info_space = info_space
        self.coor, self.index = self.info_space.get_coor_idx(self)
        self.random_seed = 100 * (info_space.get('trial') or 0) + self.index
        util.set_module_seed(self.random_seed)
        self.data = None
        self.aeb_space = AEBSpace(self.spec, self.info_space)
        self.env_space = EnvSpace(self.spec, self.aeb_space)
        self.agent_space = AgentSpace(self.spec, self.aeb_space, global_nets)
        logger.info(util.self_desc(self))
        self.aeb_space.init_body_space()
        self.aeb_space.post_body_init()
        logger.info(f'Initialized session {self.index}')

    def run_all_episodes(self):
        '''
        Run all episodes, where each env can step and reset at its own clock_speed and timeline. Will terminate when all envs done running max_episode.
        '''
        _reward_space, state_space, _done_space = self.env_space.reset()
        _action_space = self.agent_space.reset(state_space)  # nan action at t=0 for bookkeeping in data_space
        while True:
            # TODO epi now stats from self.epi = 0 instead of 1. use the same ticking scheme as singleton
            end_session = self.aeb_space.tick_clocks(self)
            if end_session:
                break
            action_space = self.agent_space.act(state_space)
            reward_space, state_space, done_space = self.env_space.step(action_space)
            self.agent_space.update(action_space, reward_space, state_space, done_space)

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env.
        Prepare self.df.
        '''
        self.agent_space.close()
        self.env_space.close()
        logger.info('Session done, closing.')

    def run(self):
        self.run_all_episodes()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data


class DistSession(mp.Process):
    '''Distributed Session for distributed training'''

    def __init__(self, spec, info_space, global_nets):
        super(DistSession, self).__init__()
        self.name = f'w{info_space.get("session")}'
        self.session = Session(spec, info_space, global_nets)

    def run(self):
        return self.session.run()


class Trial:
    '''
    The base unit of an experiment.
    Given a spec and number s,
    trial creates and runs s sessions,
    gather and aggregate data from sessions as trial data,
    then return the trial data.
    '''

    def __init__(self, spec, info_space):
        self.spec = spec
        self.info_space = info_space
        self.coor, self.index = self.info_space.get_coor_idx(self)
        self.session_data_dict = {}
        self.data = None
        analysis.save_spec(spec, info_space, unit='trial')
        logger.info(f'Initialized trial {self.index}')

    def init_session_and_run(self, info_space):
        session = Session(self.spec, info_space)
        session_data = session.run()
        return session_data

    def run_serial_sessions(self):
        logger.info('Running serial sessions')
        num_cpus = ps.get(self.spec['meta'], 'resources.num_cpus', util.NUM_CPUS)
        info_spaces = []
        for _s in range(self.spec['meta']['max_session']):
            self.info_space.tick('session')
            info_spaces.append(deepcopy(self.info_space))
        if util.get_lab_mode() == 'train' and len(info_spaces) > 1:
            session_datas = util.parallelize_fn(self.init_session_and_run, info_spaces, num_cpus)
        else:  # dont parallelize when debugging to allow render
            session_datas = []
            for info_space in info_spaces:
                session_data = self.init_session_and_run(info_space)
                session_datas.append(session_data)
                if analysis.is_unfit(session_data):
                    break
        return session_datas

    def init_global_nets(self):
        global_session = Session(deepcopy(self.spec), deepcopy(self.info_space))
        global_agent = global_session.agent
        global_session.env.close()
        global_nets = {}
        for net_name in global_agent.algorithm.net_names:
            g_net = getattr(global_agent.algorithm, net_name)
            g_net.share_memory()  # make global sharable
            # TODO also create shared optimizer here
            global_nets[net_name] = g_net
        return global_nets

    def run_distributed_sessions(self):
        logger.info('Running distributed sessions')
        global_nets = self.init_global_nets()
        workers = []
        for s in range(self.spec['meta']['max_session']):
            self.info_space.tick('session')
            w = DistSession(deepcopy(self.spec), self.info_space, global_nets)
            w.start()
            workers.append(w)
        for w in workers:
            w.join()

        prepath = util.get_prepath(self.spec, self.info_space)
        predir = util.prepath_to_predir(prepath)
        session_datas = analysis.session_data_dict_from_file(predir, self.info_space.get('trial'))
        session_datas = [session_datas[k] for k in sorted(session_datas.keys())]
        return session_datas

    def close(self):
        logger.info('Trial done, closing.')

    def run(self):
        if self.spec['meta'].get('distributed'):
            session_datas = self.run_distributed_sessions()
        else:
            session_datas = self.run_serial_sessions()
        self.session_data_dict = {data.index[0]: data for data in session_datas}
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

    def __init__(self, spec, info_space):
        self.spec = spec
        self.info_space = info_space
        self.coor, self.index = self.info_space.get_coor_idx(self)
        self.trial_data_dict = {}
        self.data = None
        SearchClass = getattr(search, spec['meta'].get('search'))
        self.search = SearchClass(self)
        analysis.save_spec(spec, info_space, unit='experiment')
        logger.info(f'Initialized experiment {self.index}')

    def init_trial_and_run(self, spec, info_space):
        '''
        Method to run trial with the properly updated info_space (trial_index) from experiment.search.lab_trial.
        Do not tick info_space below, it is already updated when passed from lab_trial.
        '''
        trial = Trial(spec, info_space)
        trial_data = trial.run()
        return trial_data

    def close(self):
        reload(search)  # to fix ray consecutive run crash due to bad cleanup
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
