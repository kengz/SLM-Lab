'''
The data_space module
Handles the unified hyperdimensional data for SLM Lab, used for analysis and experiment planning.
Sources data from monitor.
Each dataframe resolves from the coarsest dimension to the finest, with data coordinates data_coor in the form: (evolution,experiment,trial,session,agent,env,body,episode,timestep)
The resolution after session is the AEB space, hence it is a subspace.
AEB space is not necessarily tabular, and hence the data is NoSQL.

The data_space is congruent to the data_coor, with proper resolution.
E.g. (evolution,experiment,trial,session) specifies the session_data of a session, ran over multiple episodes on the AEB space.

Components:
- AEB resolver
- data_coor resolver
- data_space resolver
- monitor data sourcer
- data_space getter and setter for running experiments
- plug to NoSQL graph db, using graphql notation, and data backup
- data_space viewer and stats method for evaluating and planning experiments
'''

# TODO maybe stop at timestep since episode is the lowest sensible refinement of data without bloat
DATA_COOR_AXES = [
    'evolution',
    'experiment',
    'trial',
    'session',
    'agent',
    'env',
    'body',
    'episode',
    'timestep',
]
DATA_COOR_AXES_ORDER = {
    axis: idx for idx, axis in enumerate(DATA_COOR_AXES)
}
DATA_COOR_DIM = len(DATA_COOR_AXES)

# TODO maybe not create new dict per episode and timestep for complexity


def create_data_coor():
    '''Create a new data_coor dict'''
    return {k: None for k in DATA_COOR_AXES}


# TODO or every stage the class instance shd hold its own data_coor that ends at its axis
def update_data_coor(data_coor, axis):
    '''
    Update a given data_coor at axis (name).
    If the axis value has been reset, update to 0, else increment.
    For all axes post the specified axis, reset to None.
    '''
    # TODO increment pattern in break down in AEB
    assert axis in data_coor
    if data_coor[axis] is None:
        data_coor[axis] = 0
    else:
        data_coor[axis] += 1

    axis_idx = DATA_COOR_AXES_ORDER[axis]
    for post_idx in range(axis_idx + 1, DATA_COOR_DIM):
        post_axis = DATA_COOR_AXES[post_idx]
        data_coor[post_axis] = None

    return data_coor

# data_coor = create_data_coor()
# print(data_coor)
# update_data_coor(data_coor, 'trial')
# print(data_coor)
