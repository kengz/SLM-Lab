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
from collections import namedtuple

HYPERINDEX_ORDER = [
    'evolution',
    'experiment',
    'trial',
    'session',
    'agent',
    'env',
    'body',
    'episode',
    'timestep'
]
Hyperindex = namedtuple('Hyperindex', HYPERINDEX_ORDER)


hyperindex = Hyperindex(*list(range(len(HYPERINDEX_ORDER))))
hyperindex
