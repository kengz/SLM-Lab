'''
The experiment module
Handles experimentation logic: control, design, monitoring, analysis, evolution
'''
from slm_lab.lib import logger, util, viz


class Monitor:
    '''
    Monitors agents, environments, sessions, trials, experiments, evolutions.
    Has standardized input/output data structure, methods.
    Persists data to DB, and to viz module for plots or Tensorboard.
    Pipes data to Controller for evolution.
    TODO Possibly unify this with logger module.
    '''
    pass


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
    '''
    spec = None
    agent = None
    env = None
    data = None
    monitor = None

    def __init__(self, spec):
        # init agent and env as necesary
        return

    def init_agent(self):
        return

    def init_env(self):
        return

    def run_episode(self):
        # TODO multi-brain resolver without tampering below
        # note: an Agent here is the same unit as a Brain in Unity
        # sys_vars is now session_data, should collect silently from agent and env (fully observable anyways with full access)
        # preprocessing shd belong to agent internal, analogy: a lens
        # any rendering goes to env
        # make env observable to agent, vice versa. useful for memory
        # foresee multi agent, multi environment in a session
        self.agent.reset()
        state = self.env.reset()
        # RL steps for SARS
        for t in range(self.env.max_timestep):
            action = self.agent.act(state)
            reward, state, done = self.env.step(action)
            # fully observable SARS from env, memory and training internally
            self.agent.update(reward, state)
            self.monitor.update()
            if done:
                break
        return

    def run(self):
        for e in range(self.spec.max_episode):
            self.run_episode()
        # TODO tie up things like close env and save agent, package data
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
