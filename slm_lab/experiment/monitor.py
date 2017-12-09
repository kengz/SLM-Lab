'''
The monitor module with data_space
Monitors agents, environments, sessions, trials, experiments, evolutions, and handles all the data produced by the Lab components.
InfoSpace handles the unified hyperdimensional data for SLM Lab, used for analysis and experiment planning. Sources data from monitor.
Each dataframe resolves from the coarsest dimension to the finest, with data coordinates coor in the form: (evolution,experiment,trial,session,agent,env,body)
The resolution after session is the AEB space, hence it is a subspace.
AEB space is not necessarily tabular, and hence the data is NoSQL.

The data_space is congruent to the coor, with proper resolution.
E.g. (evolution,experiment,trial,session) specifies the session_data of a session, ran over multiple episodes on the AEB space.

Space ordering:
InfoSpace: the general space for complete information
AEBSpace: subspace of InfoSpace for a specific session
AgentSpace: space agent instances, subspace of AEBSpace
EnvSpace: space of env instances, subspace of AEBSpace
DataSpace: a data space storing an AEB data projected to a-axis, and its dual projected to e-axis. This is so that a-proj data like action_space from agent_space can be used by env_space, which requires e-proj data, and vice versa.

Object reference (for agent to access env properties, vice versa):
Agents - AgentSpace - AEBSpace - EnvSpace - Envs
'''
# TODO - plug to NoSQL graph db, using graphql notation, and data backup
# TODO - data_space viewer and stats method for evaluating and planning experiments
from copy import deepcopy
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import numpy as np
import pydash as _

# These correspond to the control unit classes, lower cased
COOR_AXES = [
    'evolution',
    'experiment',
    'trial',
    'session',
]
COOR_AXES_ORDER = {
    axis: idx for idx, axis in enumerate(COOR_AXES)
}
COOR_DIM = len(COOR_AXES)
AGENT_DATA_NAMES = ['action']
ENV_DATA_NAMES = ['state', 'reward', 'done']


class Body:
    '''
    Body, helpful info reference unit under AEBSpace for sharing info between agent and env.
    '''

    def __init__(self, aeb, agent, env):
        self.coor = aeb
        self.a, self.e, self.b = aeb
        self.agent = agent
        self.env = env
        self.observable_dim = self.env.get_observable_dim(self.a)
        # TODO use tuples for state_dim for pixel-based in the future
        self.state_dim = self.observable_dim['state']
        self.action_dim = self.env.get_action_dim(self.a)
        self.is_discrete = self.env.is_discrete(self.a)

    def __str__(self):
        return 'body: ' + util.to_json(util.get_class_attr(self))


class DataSpace:
    '''
    AEB data space - data container with an AEB space hashed to index of a flat list of stored data
    '''
    # TODO prolly keep episodic, timestep historic data series, to DB per episode

    def __init__(self, data_name, aeb_proj_dual_map, aeb_space):
        self.data_name = data_name
        self.aeb_proj_dual_map = aeb_proj_dual_map
        self.aeb_space = aeb_space
        if data_name in AGENT_DATA_NAMES or data_name == 'body':
            self.proj_axis = 'a'
            self.dual_proj_axis = 'e'
        else:
            self.proj_axis = 'e'
            self.dual_proj_axis = 'a'
        self.data_proj = None
        self.dual_data_proj = None

    def __str__(self):
        s = '['
        x_map = self.aeb_proj_dual_map[self.dual_proj_axis]
        x_yb_proj = self.aeb_space.a_eb_proj if self.dual_proj_axis == 'a' else self.aeb_space.e_ab_proj
        for x, y_map_idx_list in enumerate(x_map):
            s += f'\n  {self.proj_axis}:{x} ['
            for x_idx, (y, xb_idx) in enumerate(y_map_idx_list):
                b = x_yb_proj[x][x_idx][1]
                data = self.data_proj[y][xb_idx]
                s += f'({self.dual_proj_axis}:{y},b:{b}) {data} '
            s += ']'
        s += '\n]'
        return s

    def __bool__(self):
        '''Get AND of all entries, reduce 2 times on 2D axes of AE, then convert to py bool'''
        return bool(np.all(np.all(self.data_proj)))

    def create_dual_data_proj(self, data_proj):
        '''
        Every data_proj from agent will be used by env, and vice versa.
        Hence, on receiving data_proj, construct and cache the dual for fast access later.
        @param {[y: [xb_idx:[body_data]]} data_proj, where x, y could be a, e interchangeably.
        @returns {[x: [yb_idx:[body_data]]} dual_data_proj, with axes flipped.
        '''
        x_map = self.aeb_proj_dual_map[self.dual_proj_axis]
        x_data_proj = []
        try:
            for _x, y_map_idx_list in enumerate(x_map):
                x_data_proj_x = []
                for _x_idx, (y, xb_idx) in enumerate(y_map_idx_list):
                    data = data_proj[y][xb_idx]
                    x_data_proj_x.append(data)
                x_data_proj.append(x_data_proj_x)
        except Exception as e:
            logger.exception(
                f'Dimension mismatch of data space {self.data_name}. Expected aeb_proj_map: {x_map}, received input data_proj: {data_proj}')
            raise e
        return x_data_proj

    def add(self, data_proj):
        '''
        Add a new instance of data projection to data_space from agent_space or env_space. Creates a dual_data_proj.
        @param {[x: [yb_idx:[body_data]]} data_proj, where x, y could be a, e interchangeably.
        '''
        # TODO might wanna keep a history before replacement, shove to DB
        self.data_proj = data_proj
        self.dual_data_proj = self.create_dual_data_proj(data_proj)

    def get(self, a=None, e=None):
        '''
        Get the data_proj for a or e axis to be used by agent_space, env_space respectively, automatically projected and resolved.
        @param {int} a The index a of an agent in agent_space
        @param {int} e The index e of an env in env_space
        @returns {[yb_idx:[body_data]_x} data_proj[x], where x, y could be a, e interchangeably.
        '''
        if a is not None:
            proj_axis = 'a'
            proj_idx = a
        else:
            proj_axis = 'e'
            proj_idx = e
        assert proj_idx > -1
        if proj_axis == self.proj_axis:
            return self.data_proj[proj_idx]
        else:
            return self.dual_data_proj[proj_idx]


class AEBSpace:

    def __init__(self, spec):
        # TODO shove
        # self.info_space = info_space
        self.spec = spec
        self.agent_space = None
        self.env_space = None
        self.body_space = None
        self.coor_list = spec_util.resolve_aeb(self.spec)
        self.aeb_shape, self.a_eb_proj = self.compute_aeb_dims(self.coor_list)
        self.e_ab_proj = None
        self.aeb_proj_dual_map = {
            'a': None,
            'e': None,
        }
        self.data_spaces = self.init_data_spaces()
        self.clock = {
            unit: 0 for unit in ['t', 'total_t', 'e']
        }

    def compute_aeb_dims(self, coor_list):
        '''
        Compute the aeb_shape and a_eb_proj from coor_list, which are used to resolve agent_space and env_space.
        @param {[(a, e, b)]} coor_list The array of aeb coors
        @returns {array([a, e, b]), [a: [(e, b)]]} aeb_shape, a_eb_proj
        '''
        aeb_shape = util.get_aeb_shape(coor_list)
        a_aeb_groups = _.group_by(coor_list, lambda aeb: aeb[0])
        a_eb_proj = []
        for a, aeb_list in a_aeb_groups.items():
            a_eb_proj.append(
                util.to_tuple_list(np.array(aeb_list)[:, 1:]))
        assert len(a_eb_proj) == len(self.spec['agent'])
        return aeb_shape, a_eb_proj

    def compute_dual_map(cls, a_eb_proj):
        '''
        Compute the direct dual map and dual proj of the given proj by swapping a, e
        @param {[a: [(e, b)]]} a_eb_proj The aeb space projected onto a-axis
        @returns {[e: [(a, eb_idx)]], [e: [(a, b)]]} e_ab_dual_map, e_ab_proj
        '''
        flat_aeb_list = []
        for a, eb_list in enumerate(a_eb_proj):
            for eb_idx, (e, b) in enumerate(eb_list):
                flat_aeb_list.append((a, e, b, eb_idx))
        flat_aeb_list = sorted(flat_aeb_list)

        e_ab_dual_map = []
        e_ab_proj = []
        e_aeb_groups = _.group_by(flat_aeb_list, lambda row: row[1])
        for e, eab_list in e_aeb_groups.items():
            e_ab_dual_map.append(util.to_tuple_list(
                np.array(eab_list)[:, (0, 3)]))
            e_ab_proj.append(util.to_tuple_list(np.array(eab_list)[:, (0, 2)]))
        return e_ab_dual_map, e_ab_proj

    def init_aeb_proj_dual_map(self):
        '''
        Initialize the AEB projection dual map to map aeb_data_space between agent space and env space.
        agent_space output data_proj, shape [a, [(e, b)]]
        env_space output data_proj shape [e, [(a, b)]]
        '''
        e_ab_dual_map, e_ab_proj = self.compute_dual_map(self.a_eb_proj)
        a_eb_dual_map, check_a_eb_proj = self.compute_dual_map(e_ab_proj)
        assert np.array_equal(self.a_eb_proj, check_a_eb_proj)

        self.e_ab_proj = e_ab_proj
        self.aeb_proj_dual_map['a'] = a_eb_dual_map
        self.aeb_proj_dual_map['e'] = e_ab_dual_map

    def init_data_spaces(self):
        '''Initialize the data_space that contains all the data for the Lab.'''
        self.data_spaces = {
            data_name: None for data_name in _.concat(AGENT_DATA_NAMES, ENV_DATA_NAMES)
        }
        self.init_aeb_proj_dual_map()
        for data_name in self.data_spaces:
            data_space = DataSpace(data_name, self.aeb_proj_dual_map, self)
            self.data_spaces[data_name] = data_space
        return self.data_spaces

    def tick_clock(self, unit):
        '''Tick the clock a unit into future time'''
        if unit == 't':  # timestep
            self.clock['t'] += 1
            self.clock['total_t'] += 1
        elif unit == 'e':  # episode, reset timestep
            self.clock['t'] = 0
            self.clock['e'] += 1
        else:
            raise KeyError
        return self.clock

    def init_body_space(self):
        '''Initialize the body_space (same class as data_space) used for AEB body resolution, and set reference in agents and envs'''
        self.body_space = DataSpace('body', self.aeb_proj_dual_map, self)
        data_proj = deepcopy(self.a_eb_proj)
        for a, eb_list in enumerate(self.a_eb_proj):
            for eb_idx, (e, b) in enumerate(eb_list):
                agent = self.agent_space.get(a)
                env = self.env_space.get(e)
                body = Body((a, e, b), agent, env)
                data_proj[a][eb_idx] = body
        self.body_space.add(data_proj)

        for agent in self.agent_space.agents:
            agent.bodies = self.body_space.get(a=agent.index)
        for env in self.env_space.envs:
            env.bodies = self.body_space.get(e=env.index)
        return self.body_space

    def post_body_init(self):
        '''Run init for agent, env components that need bodies to exist first, e.g. memory or architecture.'''
        self.agent_space.post_body_init()
        self.env_space.post_body_init()

    def add(self, data_name, data_proj):
        '''
        Add a data projection to a data space, e.g. data_proj actions collected per body, per agent, from agent_space, with AEB shape projected on a-axis, added to action_space.
        @param {str} data_name
        @param {[x: [yb_idx:[body_data]]} data_proj, where x, y could be a, e interchangeably.
        @returns {DataSpace} data_space (aeb is implied)
        '''
        data_space = self.data_spaces[data_name]
        data_space.add(data_proj)
        return data_space


# TODO put AEBSpace into InfoSpace, propagate method usage, shove into DB
class InfoSpace:
    def __init__(self, last_coor=None):
        '''
        Initialize the coor, the global point in info space that will advance according to experiment progress.
        The coor starts with null first since the coor may not start at the origin.
        TODO In general, when we parallelize to use multiple coor on a info space, keep a covered space and multiple coors to advance without conflicts.
        TODO logic to resume from given last_coor
        '''
        self.coor = last_coor or {k: None for k in COOR_AXES}
        self.covered_space = []

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

    def index_lab_comp(self, lab_comp):
        '''
        Update info space coor when initializing lab component, and return its coor and index.
        Does not apply to AEB entities.
        @returns {tuple, int} data_coor, index
        @example

        class Session:
            def __init__(self, spec):
                self.coor, self.index = info_space.index_lab_comp(self)
        '''
        axis = util.get_class_name(lab_comp, lower=True)
        self.advance_coor(axis)
        coor = self.coor.copy()
        index = coor[axis]
        return coor, index


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
info_space = InfoSpace()
