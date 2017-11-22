'''
The control module
Creates and controls the units of SLM lab: EvolutionGraph, Experiment, Trial, Session
'''
import numpy as np
import pandas as pd
import pydash as _
# TODO resolve spec module name conflict, naming env_spec
from slm_lab.spec import spec_util
from slm_lab.agent import Agent, AgentSpace
from slm_lab.env import Body, Env, EnvSpace
from slm_lab.experiment.monitor import data_space
from slm_lab.lib import logger, util, viz


class Session:
    '''
    The base unit of instantiated RL system.
    Given a spec,
    session creates agent(s) and environment(s),
    run the RL system and collect data, e.g. fitness metrics, till it ends,
    then return the session data.
    TODO only experiment_spec, agent_spec, agent_spec
    auto-resolve param space spec for trial, copy for session with idx
    '''
    spec = None
    data = None
    aeb_coor_arr = None
    env_space = None
    agent_space = None

    def __init__(self, spec):
        data_space.init_lab_comp_coor(self, spec)
        self.data = pd.DataFrame()

        # TODO init AEB space by resolving from data_space
        # TODO put resolved space from spec into monitor.dataspace
        self.aeb_coor_arr = spec_util.resolve_aeb(self.spec)

        self.env_space = EnvSpace(self.spec)
        self.agent_space = AgentSpace(self.spec)
        self.env_space.set_agent_space(self.agent_space)
        self.agent_space.set_env_space(self.env_space)

        self.init_bodies()

    def init_bodies(self):
        # TODO at init after AEB resolution and projection, check if all bodies can fit in env
        # TODO prolly need proxy body object to link from agent batch output index to bodies in generalized number of environments
        # AEB stores resolved AEB coordinates to linking bodies
        for (a, e, b) in self.aeb_coor_arr:
            body = Body(a, e, b)
            # TODO set upper reference to retrieve objects quickly
            self.env_space.add_body(body)
            self.agent_space.add_body(body)

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
        TODO still WIP
        sys_vars is now session_data, should collect silently from agent and env (fully observable anyways with full access)
        preprocessing shd belong to agent internal, analogy: a lens
        any rendering goes to env
        make env observable to agent, vice versa. useful for memory
        '''
        # TODO generalize and make state to include observables
        state_space = self.env_space.reset()
        self.agent_space.reset()
        # RL steps for SARS
        for t in range(self.env_space.max_timestep):
            # TODO create an AEB data carrier for SARS
            # state space from env is AEB on E, need to transpose
            # ok do these internally to agent.
            # or no need to transpose, just collect and fill up AEB space
            # for every production, collect, aim aeb coor by bodies x A or E, fill in AEB cube
            # TODO or could have a central storage with hashed index, and the AEB space element would jsut carry that index, so no real data is transformed
            # so it's just a flattened list from both spaces
            # this is all proxied by a space container carrying the data with the AEB index
            # TODO needa class for taht
            action_space = self.agent_space.act(state_space)
            # at this point, grouped by agent, but need to reproject and regroup by env. ensure all lies along AEB tho
            # use transpose
            logger.debug(f'action_space {action_space}')
            reward_space, state_space, done_space = self.env_space.step(
                action_space)
            # at this point, grouped by env. still lying along AEB
            logger.debug(f'reward_space: {reward_space}, state_space: {state_space}, done_space: {done_space}')
            # fully observable SARS from env_space, memory and training internally
            self.agent_space.update(reward_space, state_space)
            if np.all(done_space):
                break
        # TODO compose episode data
        episode_data = {}
        return episode_data

    def run(self):
        for e in range(_.get(self.spec, 'meta.max_episode')):
            logger.debug(f'episode {e}')
            self.run_episode()
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
    spec = None
    data = None
    session = None

    def __init__(self, spec):
        data_space.init_lab_comp_coor(self, spec)
        self.data = pd.DataFrame()

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
    spec = None
    data = None
    trial = None

    def __init__(self, spec):
        data_space.init_lab_comp_coor(self, spec)
        return

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
