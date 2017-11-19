'''
The control module
Creates and controls the units of SLM lab: EvolutionGraph, Experiment, Trial, Session
'''
import pandas as pd
import pydash as _
from slm_lab.agent import Agent
from slm_lab.env import Env
from slm_lab.experiment import data_space
from slm_lab.lib import logger, util, viz


class Monitor:
    '''
    Monitors agents, environments, sessions, trials, experiments, evolutions.
    Has standardized input/output data structure, methods.
    Persists data to DB, and to viz module for plots or Tensorboard.
    Pipes data to Controller for evolution.
    TODO Possibly unify this with logger module.
    '''

    def __init__(self, spec):
        logger.debug('Monitor initialized.')
        self.data_coor = data_space.create_data_coor()

    def update_stage(self, axis):
        data_space.update_data_coor(self.data_coor, axis)
        return self.data_coor

    def update(self):
        # TODO hook monitor to agent, env, then per update, auto fetches all that is in background
        # TODO call update in session, trial, experiment loops to collect data visible there too, for unit_data
        return


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

    def __init__(self, spec, monitor):
        # TODO monitor has to be top level outside of Experiment
        # TODO probably is a good idea to not have global Monitor
        # that case if one session runtime fails, others wont
        # or monitor has to be copied in parallel runs too
        self.monitor = monitor

        self.monitor.update_stage('session')
        print(self.monitor.data_coor['session'])

        self.spec = spec
        self.data = pd.DataFrame()

        self.env = self.init_env()
        self.agent = self.init_agent()

    def init_agent(self):
        # TODO absorb into class init?
        self.monitor.update_stage('agent')
        print(self.monitor.data_coor['agent'])
        agent_spec = self.spec['agent']
        self.agent = Agent(agent_spec, self.monitor.data_coor)
        # TODO link in AEB space properly
        self.agent.set_env(self.env)
        self.env.set_agent(self.agent)
        return self.agent

    def init_env(self):
        # TODO absorb into class init?
        self.monitor.update_stage('env')
        print(self.monitor.data_coor['env'])
        env_spec = _.merge(self.spec['env'], self.spec['meta'])
        self.env = Env(env_spec, self.monitor.data_coor)
        return self.env

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env. Update monitor.
        Prepare self.data.
        '''
        # TODO catch all to close Unity when py runtime fails
        self.agent.close()
        self.env.close()
        self.monitor.update()
        logger.info('Session done, closing.')

    def run_episode(self):
        '''
        TODO still WIP
        sys_vars is now session_data, should collect silently from agent and env (fully observable anyways with full access)
        preprocessing shd belong to agent internal, analogy: a lens
        any rendering goes to env
        make env observable to agent, vice versa. useful for memory
        '''
        # TODO substitute singletons for spaces later
        self.monitor.update_stage('episode')
        print(self.monitor.data_coor['episode'])
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
            self.monitor.update()
            # TODO monitor shd update session data from episode data
            if done:
                break
        # TODO compose episode data from monitor update, is just session_data[episode]
        episode_data = {}
        return episode_data

    def run(self):
        for e in range(_.get(self.spec, 'meta.max_episode')):
            logger.debug(f'episode {e}')
            self.run_episode()
            self.monitor.update()
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
    spec = None
    data = None
    session = None

    def __init__(self, spec, monitor):
        self.monitor = monitor
        self.monitor.update_stage('trial')
        print(self.monitor.data_coor['trial'])
        self.spec = spec
        self.data = pd.DataFrame()

    def init_session(self):
        self.session = Session(self.spec, self.monitor)
        return self.session

    def close(self):
        return

    def run(self):
        for s in range(_.get(self.spec, 'meta.max_session')):
            logger.debug(f'session {s}')
            self.init_session().run()
            self.monitor.update()
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

    def __init__(self, spec):
        return

    def init_session(self):
        return

    def close(self):
        return

    def run(self):
        for s in range(_.get(self.spec, 'meta.max_session')):
            logger.debug(f'session {e}')
            self.sess.run()
            self.monitor.update()
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


# TODO detach data_coor from monitor
# TODO universal index in spec: experiment, trial, session, then agent, env, bodies
# TODO spec resolver for params per trial
# TODO spec key checker and defaulting mechanism, by merging a dict of congruent shape with default values
# TODO AEB space resolver
