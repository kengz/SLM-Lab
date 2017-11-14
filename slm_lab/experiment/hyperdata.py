'''
The hyperdata module
Handles the unified hyperdimensional data for SLM Lab,
used for analysis and experiment planning.
Sources data from monitor.
Each dataframe resolves from the coarsest dimension to the finest,
with hyperindex in the form:
(evolution,experiment,trial,session,agent,env,body,episode,timestep)
The resolution after session is the AEB space, hence it is a subspace.
AEB space is not necessarily tabular, and hence the data is NoSQL.

The hyperdata is congruent to the hyperindex, with proper resolution.
E.g. (evolution,experiment,trial,session) specifies the session_data of a session, ran over multiple episodes on the AEB space.

Components:
- AEB resolver
- hyperindex resolver
- hyperdata resolver
- monitor data sourcer
- hyperdata getter and setter for running experiments
- plug to NoSQL graph db, using graphql notation, and data backup
- hyperdata viewer and stats method for evaluating and planning experiments
'''

# TODO rename hyper-everthing without hyper
# TODO maybe stop at timestep since episode is the lowest sensible refinement of data without bloat
HYPERINDEX_AXES = [
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
HYPERINDEX_AXES_ORDER = {
    axis: idx for idx, axis in enumerate(HYPERINDEX_AXES)
}
HYPERINDEX_DIM = len(HYPERINDEX_AXES)

# TODO maybe not create new dict per episode and timestep for complexity


def create_hyperindex():
    '''Create a new hyperindex dict'''
    return {k: None for k in HYPERINDEX_AXES}


# TODO or every stage the class instance shd hold its own hyperindex that ends at its axis
def update_hyperindex(hyperindex, axis):
    '''
    Update a given hyperindex at axis (name).
    If the axis value has been reset, update to 0, else increment.
    For all axes post the specified axis, reset to None.
    '''
    # TODO increment pattern in break down in AEB
    assert axis in hyperindex
    if hyperindex[axis] is None:
        hyperindex[axis] = 0
    else:
        hyperindex[axis] += 1

    axis_idx = HYPERINDEX_AXES_ORDER[axis]
    for post_idx in range(axis_idx + 1, HYPERINDEX_DIM):
        post_axis = HYPERINDEX_AXES[post_idx]
        hyperindex[post_axis] = None

    return hyperindex

# hyperindex = create_hyperindex()
# print(hyperindex)
# update_hyperindex(hyperindex, 'trial')
# print(hyperindex)
