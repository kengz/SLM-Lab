'''
The control module
Creates and controls the units of SLM lab: EvolutionGraph, Experiment, Trial, Session
'''
import pandas as pd
import pydash as _
from slm_lab.agent import Agent
from slm_lab.env import Env
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
    agent = None
    env = None
    envs = []
    agents = []

    def __init__(self, spec):
        data_space.init_lab_comp_coor(self, spec)
        self.data = pd.DataFrame()

        self.init_AEB()
        self.init_envs()
        self.init_agents()
        self.init_bodies()

    def init_AEB(self):
        # TODO init AEB space by resolving from data_space
        return

    def init_envs(self):
        for env_spec in self.spec['env']:
            env = Env(env_spec, self.spec['meta'])
            self.envs.append(env)
        return self.envs

    def init_agents(self):
        for agent_spec in self.spec['agent']:
            agent = Agent(agent_spec)
            self.agents.append(agent)
        return self.agents

    def init_bodies(self):
        # TODO at init after AEB resolution and projection, check if all bodies can fit in env
        # TODO prolly need proxy body object to link from agent batch output index to bodies in generalized number of environments
        # AEB stores resolved AEB coordinates to linking bodies
        # for (a, e, b) in self.AEB
        #     body = self.create_body(a, e, b)
        #     self.agents[a].add_body(body)
        #     self.env[e].add_body(body)
        # TODO tmp base base, use above later
        self.agents[0].set_env(self.envs[0])
        self.envs[0].set_agent(self.agents[0])
        # TODO tmp set default singleton to make singleton logic work
        self.agent = self.agents[0]
        self.env = self.envs[0]

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env.
        Prepare self.data.
        '''
        # TODO catch all to close Unity when py runtime fails
        self.agent.close()
        self.env.close()
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
        state = self.env.reset()
        logger.debug(f'reset state {state}')

        self.agent.reset()
        # RL steps for SARS
        for t in range(self.env.max_timestep):
            action = self.agent.act(state)
            logger.debug(f'action {action}')
            reward, state, done = self.env.step(action)
            logger.debug(f'reward: {reward}, state: {state}, done: {done}')
            # fully observable SARS from env, memory and training internally
            self.agent.update(reward, state)
            if done:
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
