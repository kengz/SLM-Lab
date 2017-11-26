'''
The monitor module with data_space
Monitors agents, environments, sessions, trials, experiments, evolutions, and handles all the data produced by the Lab components.
DataSpace handles the unified hyperdimensional data for SLM Lab, used for analysis and experiment planning. Sources data from monitor.
Each dataframe resolves from the coarsest dimension to the finest, with data coordinates coor in the form: (evolution,experiment,trial,session,agent,env,body)
The resolution after session is the AEB space, hence it is a subspace.
AEB space is not necessarily tabular, and hence the data is NoSQL.

The data_space is congruent to the coor, with proper resolution.
E.g. (evolution,experiment,trial,session) specifies the session_data of a session, ran over multiple episodes on the AEB space.

Space ordering:
DataSpace: the general space for complete data

AEBSpace: subspace of DataSpace for a specific session

AgentSpace: space agent instances, subspace of AEBSpace

EnvSpace: space of env instances, subspace of AEBSpace

AEBDataSpace: a data space for a type of data inside AEBSpace, e.g. action_space, reward_space. Each (a,e,b) coordinate maps to a projection (a or e axis) of the data of the body (at a timestep). The map, `aeb_idx_space` is a copy of the AEBSpace, and its scalar value at (a,e,b) is the projected index (ab_idx, eb_idx) of the data in `data_proj`.
E.g. `action_proj` collected from agent_space has the congruence of aeb_space projected on the a-axis, `a_eb_proj = [[(0, 0)]]` with shape [a, [(e, b)]]. First flat index is from the first agent, and the data there is for the multiple bodies of the agent, belonging to (e,b). Vice versa (swap a-e) for `data_proj` collected from env_space.
'''
# TODO - plug to NoSQL graph db, using graphql notation, and data backup
# TODO - data_space viewer and stats method for evaluating and planning experiments
# TODO change to ensure coorlist is of tuples, add assert to coor usage
import numpy as np
import pydash as _
from slm_lab.lib import util
from slm_lab.spec import spec_util

# These correspond to the control unit classes, lower cased
COOR_AXES = [
    'evolution',
    'experiment',
    'trial',
    'session',
    'agent',
    'env',
    'body',
]
COOR_AXES_ORDER = {
    axis: idx for idx, axis in enumerate(COOR_AXES)
}
COOR_DIM = len(COOR_AXES)
AGENT_DATA_NAMES = ['action']
ENV_DATA_NAMES = ['state', 'reward', 'done']

# TODO need to assert when accessing index data_proj[idx] idx != -1
# TODO at init after AEB resolution and projection, check if all bodies can fit in env
# TODO AEB needs to check agent output dim is sufficient


class AEBDataSpace:
    '''
    AEB data space - data container with an AEB space hashed to index of a flat list of stored data
    '''
    # TODO prolly keep episodic, timestep historic data series
    data_name = None
    proj_aeb_idx_space = None
    proj_axis = None
    data_proj = None
    # TODO how to u know the axis of proj?

    def __init__(self, data_name, proj_aeb_idx_space):
        self.data_name = data_name
        self.proj_aeb_idx_space = proj_aeb_idx_space
        self.proj_axis = 'a' if data_name in AGENT_DATA_NAMES else 'e'

    def add(self, data_proj):
        # TODO might wanna keep a history before replacement, shove to DB
        self.data_proj = data_proj

    def get(self, a=None, e=None):
        # TODO cache projection mode, if proj_axis is the same as the existing one, just get
        # TODO resolve projection shape by reprojecting a_eb to e_ab, with proper eb, ab indices. vice versa. use proj_aeb_idx_space map from 'a' to 'e'
        # get for a, then flip to e, vice versa
        proj_idx = a if a is not None else e
        assert proj_idx > -1
        return self.data_proj[proj_idx]


class AEBSpace:
    coor_arr = None
    aeb_shape = None
    proj_aeb_idx_space = {
        'a': None,
        'e': None,
    }
    agent_space = None
    env_space = None
    data_spaces = {
        data_name: None for data_name in _.concat(AGENT_DATA_NAMES, ENV_DATA_NAMES)
    }

    def __init__(self, spec):
        self.coor_arr = spec_util.resolve_aeb(spec)
        self.aeb_shape = np.amax(self.coor_arr, axis=0) + 1
        self.init_data_spaces()

    def init_data_spaces(self):
        self.init_aeb_idx_spaces()
        for data_name in self.data_spaces:
            data_space = AEBDataSpace(data_name, self.proj_aeb_idx_space)
            self.data_spaces[data_name] = data_space

    def init_aeb_idx_spaces(self):
        # agent_space output data_proj, shape [a, [(e, b)]]
        # TODO construct the AEB space proj to A, E from spec
        a_eb_proj = [
            [(0, 0)]
        ]
        a_eb_idx_space = np.full(self.aeb_shape, -1, dtype=int)
        for a, eb_proj in enumerate(a_eb_proj):
            for eb_idx, (e, b) in enumerate(eb_proj):
                aeb = (a, e, b)
                a_eb_idx_space.itemset(aeb, eb_idx)
        self.proj_aeb_idx_space['a'] = a_eb_idx_space

        # env_space output data_proj shape [e, [(a, b)]]
        # TODO construct the AEB space proj to A, E from spec
        e_ab_proj = [
            [(0, 0)]
        ]
        e_ab_idx_space = np.swapaxes(a_eb_idx_space, 0, 1)
        for e, ab_proj in enumerate(e_ab_proj):
            for ab_idx, (a, b) in enumerate(ab_proj):
                aeb = (a, e, b)
                e_ab_idx_space.itemset(aeb, ab_idx)
        self.proj_aeb_idx_space['e'] = e_ab_idx_space

    def add(self, data_name, data_proj):
        data_space = self.data_spaces[data_name]
        data_space.add(data_proj)
        return data_space

    def set_space_ref(self, agent_space, env_space):
        '''Set symmetric references from aeb_space to agent_space and env_space. Called from control.'''
        self.agent_space = agent_space
        self.env_space = env_space
        self.agent_space.set_space_ref(self)
        self.env_space.set_space_ref(self)


class DataSpace:
    coor = None
    covered_space = []

    def __init__(self, last_coor=None):
        '''
        Initialize the coor, the global point in data space that will advance according to experiment progress.
        The coor starts with null first since the coor may not start at the origin.
        TODO In general, when we parallelize to use multiple coor on a data space, keep a covered space and multiple coors to advance without conflicts.
        TODO logic to resume from given last_coor
        '''
        self.coor = last_coor or {k: None for k in COOR_AXES}

    def reset_lower_axes(cls, coor, axis):
        '''Reset the axes lower than the given axis in coor'''
        axis_idx = COOR_AXES_ORDER[axis]
        for post_idx in range(axis_idx + 1, COOR_DIM):
            post_axis = COOR_AXES[post_idx]
            coor[post_axis] = None
        return coor

    def advance_coor(self, axis):
        '''
        Advance the coor to the next point in axis (control unit class).
        If the axis value has been reset, update to 0, else increment. For all axes lower than the specified axis, reset to None.
        Note this will not skip coor in space, even though the covered space may not be rectangular.
        TODO careful with reset on AEB under the same session
        '''
        assert axis in self.coor
        new_coor = self.coor.copy()
        if new_coor[axis] is None:
            new_coor[axis] = 0
        else:
            new_coor[axis] += 1
        new_coor = self.reset_lower_axes(new_coor, axis)
        self.covered_space.append(self.coor)
        self.coor = new_coor
        return self.coor

    def init_lab_comp_coor(self, lab_comp, spec):
        '''
        Update data space coor when initializing lab component, and set its self.spec.
        @example

        class Session:
            def __init__(self, spec):
                data_space.init_lab_comp_coor(self, spec)
        '''
        axis = util.get_class_name(lab_comp, lower=True)
        self.advance_coor(axis)
        lab_comp.coor = self.coor.copy()
        lab_comp.index = lab_comp.coor[axis]
        # for agent and env with list specs
        if isinstance(spec, list):
            comp_spec = spec[lab_comp.index]
        else:
            comp_spec = spec
        lab_comp.spec = comp_spec


class Monitor:
    '''
    Monitors agents, environments, sessions, trials, experiments, evolutions.
    Has standardized input/output data structure, methods.
    Persists data to DB, and to viz module for plots or Tensorboard.
    Pipes data to Controller for evolution.
    TODO Possibly unify this with logger module.
    TODO shove monitor reporting into control loop
    '''

    def __init__(self):
        logger.debug('Monitor initialized.')

    def update_stage(self, axis):
        pass

    def update(self):
        # TODO hook monitor to agent, env, then per update, auto fetches all that is in background
        # TODO call update in session, trial, experiment loops to collect data visible there too, for unit_data
        return


# TODO create like monitor, for experiment level
data_space = DataSpace()
