'''
The experiment module
Handles experimentation logic: control, design, monitoring, analysis, evolution
'''


class Session:
    '''
    The base unit of instantiated RL system.
    Given a spec,
    session creates agent(s) and environment(s),
    run the RL system and collect data, e.g. fitness metrics, till it ends,
    then return the session data.
    '''
    pass


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
