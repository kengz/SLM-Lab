'''
The control module
Creates and controls the units of SLM lab
EvolutionGraph, Experiment, Trial, Session
'''
import pandas as pd
import pydash as _
from slm_lab import agent
from slm_lab.env import Env
from slm_lab.experiment import hyperdata
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
        self.hyperindex = hyperdata.create_hyperindex()

    def update_stage(self, axis):
        hyperdata.update_hyperindex(self.hyperindex, axis)
        return self.hyperindex

    def update(self):
        # TODO hook monitor to agent, env, then per update, auto fetches all that is in background
        # TODO call update in session, trial, experiment loops to collect data visible there too, for unit_data
        return


class Controller:
    '''
    Controls agents, environments, sessions, trials, experiments, evolutions.
    Though many things run independently without needing a controller.
    Has standardized input/output data structure, methods.
    Uses data from Monitor for evolution.
    '''
    pass


class Session:
    '''
    The base unit of instantiated RL system.
    Given a spec,
    session creates agent(s) and environment(s),
    run the RL system and collect data, e.g. fitness metrics, till it ends,
    then return the session data.
    noo only experiment_spec, agent_spec, agent_spec
    auto-resolve param space spec for trial, copy for session with idx
    TODO rewrite
    spec = {
    agent_spec: {} or [], list instantiate classes
    env_spec: {} or [], list instantiate classes
    body_spec: {}, with keyword like '{inner, outer}, body_num', or custom (a,e): body_num
    }
    param space further outer-products this AEB space - an AEB space exists for a trial, and when trying different param for a different trial, we create a whole new AEB space

    need to extract params from agent_spec, and do a (with warning)
    new param = new Agent, could do super speedy training in parallel
    how do u enumerate the param space onto the AEB space, acting on A?
    '''
    spec = None
    data = None
    agent = None
    env = None

    def __init__(self, spec):
        monitor.update_stage('session')
        print(monitor.hyperindex['session'])

        self.spec = spec
        self.data = pd.DataFrame()

        self.agent = self.init_agent()
        self.env = self.init_env()

    def init_agent(self):
        agent_spec = self.spec['agent']
        # TODO missing: index in AEB space
        agent_spec['index'] = 0
        agent_name = agent_spec['name']
        AgentClass = agent.__dict__.get(agent_name)
        self.agent = AgentClass(agent_spec)
        # TODO absorb into class init?
        monitor.update_stage('agent')
        print(monitor.hyperindex['agent'])
        return self.agent

    def init_env(self):
        env_spec = _.merge(self.spec['env'], self.spec['meta'])
        # TODO also missing: index in AEB space, train_mode
        env_spec['index'] = 0
        self.env = Env(env_spec)
        # TODO absorb into class init?
        monitor.update_stage('env')
        print(monitor.hyperindex['env'])
        # TODO link in AEB space properly
        self.agent.set_env(self.env)
        self.env.set_agent(self.agent)
        return self.env

    def close(self):
        '''
        Close session and clean up.
        Save agent, close env. Update monitor.
        Prepare self.data.
        '''
        # TODO save agent and shits
        # TODO catch all to close Unity when py runtime fails
        self.agent.close()
        self.env.close()
        monitor.update()
        logger.info('Session done, closing.')

    def run_episode(self):
        '''
        sys_vars is now session_data, should collect silently from agent and env (fully observable anyways with full access)
        preprocessing shd belong to agent internal, analogy: a lens
        any rendering goes to env
        make env observable to agent, vice versa. useful for memory
        TODO substitute singletons for spaces later
        '''
        monitor.update_stage('episode')
        print(monitor.hyperindex['episode'])
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
            monitor.update()
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
            monitor.update()
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
    pass


class Experiment:
    '''
    The core high level unit of Lab.
    Given a spec-space/generator of cardinality t,
    a number s,
    a hyper-optimization algorithm hopt(spec, fitness-metric) -> spec_next/null
    experiment creates and runs up to t trials of s sessions each
    to optimize (maximize) the fitness metric,
    gather the trial data,
    then return the experiment data for analysis and use in evolution graph.
    Experiment data will include the trial data,
    notes on design, hypothesis, conclusion,
    analysis data, e.g. fitness metric,
    evolution link of ancestors to potential descendants.
    An experiment then forms a node containing its data in the evolution graph
    with the evolution link and
    suggestion at the adjacent possible new experiments
    On the evolution graph level, an experiment and its neighbors
    could be seen as test/development of traits.
    '''
    pass


class EvolutionGraph:
    '''
    The biggest unit of Lab.
    The evolution graph keeps track of all experiments
    as nodes of experiment data,
    with fitness metrics, evolution links, traits, which could be used to
    aid graph analysis on the traits, fitness metrics,
    to suggest new experiment via node creation, mutation or combination
    (no DAG restriction).
    There could be a high level evolution module that guides and optimizes
    the evolution graph and experiments to achieve SLM.
    '''
    pass


# TODO universal index in spec: experiment, trial, session, then agent, env, bodies
# TODO spec resolver for params per trial
# TODO spec key checker and defaulting mechanism, by merging a dict of congruent shape with default values
# TODO AEB space resolver

# Ghetto ass run method for now, only runs base case (1 agent 1 env 1 body)
logger.set_level('DEBUG')
demo_spec = util.read('slm_lab/spec/demo.json')
monitor = Monitor(demo_spec)
session_spec = demo_spec['base_case']
sess = Session(session_spec)
session_data = sess.run()
