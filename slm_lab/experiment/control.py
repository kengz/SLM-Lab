'''
The control module
Creates and controls the units of SLM lab: EvolutionGraph, Experiment, Trial, Session
'''
from copy import deepcopy
from importlib import reload
from slm_lab.agent import AgentSpace, Agent
from slm_lab.env import EnvSpace, make_env
from slm_lab.experiment import analysis, search
from slm_lab.experiment.monitor import AEBSpace, Body, enable_aeb_space
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
        os.environ['PREPATH'] = util.get_prepath(spec, info_space, unit='session')
        reload(logger)  # to set session-specific logger
        self.spec = spec
        self.info_space = info_space
        self.index = self.info_space.get('session')
        util.try_set_cuda_id(self.spec, self.info_space)
        util.set_module_seed(self.info_space.get_random_seed())
        self.data = None

        # init singleton agent and env
        self.env = make_env(self.spec)
        body = Body(self.env, self.spec['agent'])
        self.agent = Agent(self.spec, self.info_space, body=body, global_nets=global_nets)

        enable_aeb_space(self)  # to use lab's data analysis framework
        logger.info(util.self_desc(self))
        logger.info(f'Initialized session {self.index}')

    def save_if_ckpt(self, agent, env):
        '''Save for agent, env if episode is at checkpoint'''
        epi = env.clock.get('epi')
        save_this_epi = epi > 0 and hasattr(env, 'save_epi_frequency') and epi % env.save_epi_frequency == 0
        if save_this_epi:
            agent.save(ckpt='last')
            analysis.analyze_session(self)

    def run_episode(self):
        self.env.clock.tick('epi')
        logger.info(f'Running trial {self.info_space.get("trial")} session {self.index} episode {self.env.clock.get("epi")}')
        reward, state, done = self.env.reset()
        self.agent.reset(state)
        while not done:
            self.env.clock.tick('t')
            action = self.agent.act(state)
            reward, state, done = self.env.step(action)
            self.agent.update(action, reward, state, done)
        self.agent.body.log_summary()
        self.save_if_ckpt(self.agent, self.env)

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env.
        '''
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self):
        while self.env.clock.get('epi') <= self.env.max_episode:
            self.run_episode()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data


class SpaceSession(Session):
    '''Session for multi-agent/env setting'''

    def __init__(self, spec, info_space, global_nets=None):
        os.environ['PREPATH'] = util.get_prepath(spec, info_space, unit='session')
        reload(logger)  # to set session-specific logger
        self.spec = spec
        self.info_space = info_space
        self.index = self.info_space.get('session')
        util.try_set_cuda_id(self.spec, self.info_space)
        util.set_module_seed(self.info_space.get_random_seed())
        self.data = None

        self.aeb_space = AEBSpace(self.spec, self.info_space)
        self.env_space = EnvSpace(self.spec, self.aeb_space)
        self.aeb_space.init_body_space()
        self.agent_space = AgentSpace(self.spec, self.aeb_space, global_nets)

        logger.info(util.self_desc(self))
        logger.info(f'Initialized session {self.index}')

    def save_if_ckpt(self, agent_space, env_space):
        '''Save for agent, env if episode is at checkpoint'''
        for agent in agent_space.agents:
            for body in agent.nanflat_body_a:
                env = body.env
                super(SpaceSession, self).save_if_ckpt(agent, env)

    def run_all_episodes(self):
        '''
        Continually run all episodes, where each env can step and reset at its own clock_speed and timeline.
        Will terminate when all envs done are done.
        '''
        all_done = self.aeb_space.tick('epi')
        reward_space, state_space, done_space = self.env_space.reset()
        self.agent_space.reset(state_space)
        while not all_done:
            all_done = self.aeb_space.tick()
            action_space = self.agent_space.act(state_space)
            reward_space, state_space, done_space = self.env_space.step(action_space)
            self.agent_space.update(action_space, reward_space, state_space, done_space)
            self.save_if_ckpt(self.agent_space, self.env_space)

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env.
        '''
        self.agent_space.close()
        self.env_space.close()
        logger.info('Session done and closed.')

    def run(self):
        self.run_all_episodes()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data


class DistSession(mp.Process):
    '''Distributed Session for distributed training'''

    def __init__(self, DistSessionClass, spec, info_space, global_nets):
        super(DistSession, self).__init__()
        self.name = f'w{info_space.get("session")}'
        self.session = DistSessionClass(spec, info_space, global_nets)
        logger.info(f'Initialized DistSession {self.session.index}')

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
        self.index = self.info_space.get('trial')
        self.session_data_dict = {}
        self.data = None
        analysis.save_spec(spec, info_space, unit='trial')
        self.is_singleton = len(spec['agent']) == 1 and len(spec['env']) == 1 and spec['body']['num'] == 1  # singleton mode as opposed to multi-agent-env space
        self.SessionClass = Session if self.is_singleton else SpaceSession
        logger.info(f'Initialized trial {self.index}')

    def init_session_and_run(self, info_space):
        session = self.SessionClass(self.spec, info_space)
        session_data = session.run()
        return session_data

    def run_sessions(self):
        logger.info('Running sessions')
        info_spaces = []
        for _s in range(self.spec['meta']['max_session']):
            self.info_space.tick('session')
            info_spaces.append(deepcopy(self.info_space))

        if util.get_lab_mode() == 'train' and len(info_spaces) > 1:
            # when training a single spec over multiple sessions
            session_datas = util.parallelize_fn(self.init_session_and_run, info_spaces, ps.get(self.spec['meta'], 'resources.num_cpus', util.NUM_CPUS))
        else:
            session_datas = []
            for info_space in info_spaces:
                session_data = self.init_session_and_run(info_space)
                session_datas.append(session_data)
                if analysis.is_unfit(session_data):
                    break
        return session_datas

    def make_global_nets(self, agent):
        global_nets = {}
        for net_name in agent.algorithm.net_names:
            g_net = getattr(agent.algorithm, net_name)
            g_net.share_memory()  # make net global
            # TODO also create shared optimizer here
            global_nets[net_name] = g_net
        return global_nets

    def init_global_nets(self):
        session = self.SessionClass(deepcopy(self.spec), deepcopy(self.info_space))
        if self.is_singleton:
            session.env.close()  # safety
            global_nets = self.make_global_nets(session.agent)
        else:
            session.env_space.close()  # safety
            global_nets = [self.make_global_nets(agent) for agent in session.agent_space.agents]
        return global_nets

    def run_distributed_sessions(self):
        logger.info('Running distributed sessions')
        global_nets = self.init_global_nets()
        workers = []
        for _s in range(self.spec['meta']['max_session']):
            self.info_space.tick('session')
            w = DistSession(self.SessionClass, deepcopy(self.spec), self.info_space, global_nets)
            w.start()
            workers.append(w)
        for w in workers:
            w.join()
        session_datas = analysis.session_data_dict_for_dist(self.spec, self.info_space)
        return session_datas

    def close(self):
        logger.info('Trial done and closed.')

    def run(self):
        if self.spec['meta'].get('distributed'):
            session_datas = self.run_distributed_sessions()
        else:
            session_datas = self.run_sessions()
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

    def __init__(self, spec, info_space):
        self.spec = spec
        self.info_space = info_space
        self.index = self.info_space.get('experiment')
        self.trial_data_dict = {}
        self.data = None
        analysis.save_spec(spec, info_space, unit='experiment')
        SearchClass = getattr(search, spec['meta'].get('search'))
        self.search = SearchClass(self)
        logger.info(f'Initialized experiment {self.index}')

    def init_trial_and_run(self, spec, info_space):
        '''
        Method to run trial with the properly updated info_space (trial_index) from experiment.search.lab_trial.
        '''
        trial = Trial(spec, info_space)
        trial_data = trial.run()
        return trial_data

    def close(self):
        reload(search)  # fixes ray consecutive run crashing due to bad cleanup
        logger.info('Experiment done and closed.')

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

    def __init__(self, spec, info_space):
        raise NotImplementedError
