'''
The hyperdata module
Handles the unified hyperdimensional data for SLM Lab,
used for analysis and experiment planning.
Sources data from monitor.
Each dataframe resolves from the coarsest dimension to the finest,
with hyperindex in the form:
(evolution,experiment,trial,session,episode,timestep,agent,env,body)
Note that the finest resolution is the AEB space,
hence AEB is in fact the subindex of the hyperindex.
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
