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
- DataSpace: the general space for complete data
- AEBSpace: subspace of DataSpace for a specific session
- AgentSpace: space agent instances, subspace of AEBSpace
- EnvSpace: space of env instances, subspace of AEBSpace
- AEBDataSpace: a data space for a type of data inside AEBSpace, e.g. action_space, reward_space. Each (a,e,b) coordinate maps to a flat list of the data of the body (at a timestep). The map, `aeb_idx_space` is a copy of the AEBSpace, and its scalar value at (a,e,b) is the index of the data in `data_list`.
'''
# TODO - plug to NoSQL graph db, using graphql notation, and data backup
# TODO - data_space viewer and stats method for evaluating and planning experiments
# TODO change to ensure coorlist is of tuples, add assert to coor usage
import numpy as np
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

# # these indices are permanent
# a_data_idx_map = np.full(aeb_shape, -1, dtype=int)
# a_data_idx_map.shape
# # construct first, then transpose
# e_data_idx_map = np.swapaxes(a_data_idx_map, 0, 1)
# e_data_idx_map.shape
# # e0.state = by brain and body

# TODO at init after AEB resolution and projection, check if all bodies can fit in env
# TODO AEB needs to check agent output dim is sufficient
# np.amax([[1,2], [3,4]], axis=0)
# np.amax([(1,2), (3,4)], axis=0)


class AEBDataSpace:
    '''
    AEB data space - data container with an AEB space hashed to index of a flat list of stored data
    '''
    data_name = None
    aeb_idx_space = None
    data_list = None

    def __init__(self, data_name, coor_size):
        self.data_name = data_name
        self.aeb_idx_space = np.empty(coor_size, dtype=int)

    # TODO below is predictable right
    # so we shd be able to just batch input at one time per session step, make it the data_list. And so this class is only init once per session
    # TODO also make auto resolver method
    def construct_aeb_idx_space(self):
        return

    def add(self, data_list):
        # TODO might wanna keep a history before replacement
        self.data_list = data_list

    def get(self, idx):
        return self.data_list[idx]

    def resolve_a_eb(self):
        return

    def resolve_e_ab(self):
        return

    # def add(self, data, aeb_coor):
    #     self.data_list.append(data)
    #     idx = len(self.data_list) - 1
    #     self.aeb_idx_space.itemset(aeb_coor, idx)
    #
    # def get(self, aeb_coor):
    #     # TODO assert aeb_coor is tuple by construction
    #     idx = self.aeb_idx_space[aeb_coor]
    #     return data[idx]


class AEBSpace:
    coor_arr = None
    coor_size = None
    agent_space = None
    env_space = None
    data_space_dict = {
        'state': None,
        'action': None,
        'reward': None,
        'done': None,
    }

    def __init__(self, spec):
        self.coor_arr = spec_util.resolve_aeb(spec)
        self.coor_size = np.amax(self.coor_arr, axis=0) + 1
        # TODO tmp placement here, but shd be after set_space_ref
        self.init_data_space('TODO')

    def set_space_ref(self, agent_space, env_space):
        '''Set symmetric references from aeb_space to agent_space and env_space'''
        self.agent_space = agent_space
        self.env_space = env_space
        self.agent_space.set_space_ref(self)
        self.env_space.set_space_ref(self)

    def init_data_space(self, data_name):
        # TODO also create resolver lookup from env, agent, by defining an AEB projection to A, E, which can be used to init A, E too
        # TODO tmp
        for data_name in self.data_space_dict:
            data_space = AEBDataSpace(data_name, self.coor_size)
            self.data_space_dict[data_name] = data_space

        # assert data_name in self.data_space_dict
        # TODO pending logic of lookup hash from A, E spaces and spec

    def add(self, data_name, data_list):
        data_space = self.data_space_dict[data_name]
        data_space.add(data_list)
        return data_space


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
