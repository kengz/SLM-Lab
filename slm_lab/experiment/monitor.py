'''
The monitor module with data_space
Monitors agents, environments, sessions, trials, experiments, evolutions, and handles all the data produced by the Lab components.
DataSpace handles the unified hyperdimensional data for SLM Lab, used for analysis and experiment planning. Sources data from monitor.
Each dataframe resolves from the coarsest dimension to the finest, with data coordinates coor in the form: (evolution,experiment,trial,session,agent,env,body)
The resolution after session is the AEB space, hence it is a subspace.
AEB space is not necessarily tabular, and hence the data is NoSQL.

The data_space is congruent to the coor, with proper resolution.
E.g. (evolution,experiment,trial,session) specifies the session_data of a session, ran over multiple episodes on the AEB space.

DataSpace Components:
- AEB resolver
- coor resolver
- data_space resolver
- monitor data sourcer
- data_space getter and setter for running experiments
- plug to NoSQL graph db, using graphql notation, and data backup
- data_space viewer and stats method for evaluating and planning experiments
'''
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

# a = np.arange(4).reshape((2, 2))
# a
# a.itemset((1, 1), 100)
# a
# a = np.array([(1, 20, 3), (4, 5, 6)])
# # a
# np.amax(a, axis=0)
# # a[:, 0]
# # a[:, 0].max()
# # a[:, 1]
# spec = spec_util.get('base.json', 'general_custom')
# coor_arr = spec_util.resolve_aeb(spec)
# aeb_shape = np.amax(coor_arr, axis=0) + 1
#
# # these indices are permanent
# a_data_idx_map = np.full(aeb_shape, -1, dtype=int)
# a_data_idx_map.shape
# # construct first, then transpose
# e_data_idx_map = np.swapaxes(a_data_idx_map, 0, 1)
# e_data_idx_map.shape
# # e0.state = by brain and body
#
# # base loop:
# # bodies per agent
# action_space = agent_space.act(state_space)
# # want subspace to refer super space, use the resolver to always return data in AEB format
# # space class internal
# # dummy first, with warning
# raw_action_space = [agent.act(state_space) for agent in agents]
# action_space = to_aeb_data_space(raw_action_space, a_data_idx_map)
# # => into aeb shape
# (reward_space, state_space,
#  done_space) = self.env_space.step(action_space)
# # same internals from raw to_aeb_data_space
# # => take standardized format
# # TODO also reshaping only changes the aeb_idx_space


class AEBDataSpace:
    aeb_idx_space = None
    data_list = None

    def __init__(self, coor_size):
        self.aeb_idx_space = np.empty(coor_size, dtype=int)

    def add(self, data, aeb_coor):
        self.data_list.append(data)
        idx = len(self.data_list) - 1
        self.aeb_idx_space.itemset(aeb_coor, idx)

    def get(self, aeb_coor):
        return

# TODO at init after AEB resolution and projection, check if all bodies can fit in env
# TODO AEB needs to check agent output dim is sufficient


def resolve_a_eb(space, a):
    return space[0]


def resolve_e_ab(space, e):
    return space[0]


class AEBSpace:
    # will create AEBDataSpace
    coor_arr = None
    coor_size = None
    # a registry of AEBDataSpace
    # nice to have for monitor I guess
    data_space_dict = {
        'state': None,
        'action': None,
        'reward': None,
        'done': None,
    }

    def __init__(self, spec):
        self.coor_arr = spec_util.resolve_aeb(spec)
        self.coor_size = np.amax(self.coor_arr, axis=0) + 1

    def get_box(self):
        return AEBDataSpace(self.coor_size)

    def use(self):
        '''
        box = aeb_space.get_box()
        for a, agent in enumerate(agents):
            actions = agent.act()
            for a_act_idx, action in enumerate(actions):
                e, b = lookup(a, a_act_idx)
                aeb_coor = (a,e,b)
                box.add(action, aeb_coor)
        # refactor
        construct lookup of projector, for A, reduce to EB
        pass agent x actions, with lookup to body,
        an act is a body
        every data producer must carry (full) signature in AEB space
        e.g. per agent, in act(), the agent must have a mapper from action index to (a,e,b)
        likewise for, per env, in (wrapped) step(), have a mapper from each dataspace to aeb,
        it suffices to have one of each agent or env carry the AEB mapper
        also when returning the data, the external resolver should just pick up the data as is, with the signature AEB mapper, and embed data into AEB space
        AEB space needs to see A, E then, and should actually just store those signatures.
        Session needs to init AEB space from spec, then A, E using the resolver for auto-architecture
        the map is just projection
        suppress dim to just EB, AB. flatten space, unflatten space. bodies are the true anchors.
        suppress for each e in E then fill lookup by body
        likewise for a in A
        '''
        return

    def embed_data(self, data):
        # needa be efficient
        AEBDataSpace(coor_size)
        return


# TODO debate, is it more effective to use named tuple as coor
# TODO check AEB_space body index increasing for the same AE pair


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
