'''
The analysis module
Handles the analyses of the info and data space for experiment evaluation and design.
'''
from slm_lab.agent import AGENT_DATA_NAMES
from slm_lab.env import ENV_DATA_NAMES
from slm_lab.lib import logger, util, viz
import colorlover as cl
import numpy as np
import pandas as pd
import pydash as _

DATA_AGG_FNS = {
    't': 'sum',
    'reward': 'sum',
    'loss': 'mean',
    'explore_var': 'mean',
}
FITNESS_COLS = ['strength', 'speed', 'stability', 'consistency']
FITNESS_STD = util.read('slm_lab/experiment/fitness_std.json')
MA_WINDOW = 100


def get_session_data(session):
    '''
    Gather data from session: MDP, Agent, Env data, hashed by aeb; then aggregate.
    @returns {dict, dict} session_mdp_data, session_data
    '''
    data_names = AGENT_DATA_NAMES + ENV_DATA_NAMES
    agg_data_names = ['epi'] + list(DATA_AGG_FNS.keys())
    data_h_v_dict = {
        data_name: session.aeb_space.get_history_v(data_name) for data_name in data_names}
    session_mdp_data, session_data = {}, {}
    for aeb in session.aeb_space.aeb_list:
        data_h_dict = {
            data_name: data_h_v[aeb] for data_name, data_h_v in data_h_v_dict.items()}
        # remove any incomplete session timesteps from tail (due to multienv termination)
        complete_done_h = np.trim_zeros(data_h_dict['done'], 'b')
        data_len = len(complete_done_h)
        reset_idx = np.isnan(complete_done_h)
        nonreset_idx = ~reset_idx
        data_h_dict['t'] = np.ones(reset_idx.shape)
        data_h_dict['epi'] = reset_idx.astype(int).cumsum()
        mdp_df = pd.DataFrame({
            data_name: data_h_dict[data_name][:data_len][nonreset_idx]
            for data_name in ['t', 'epi'] + data_names})
        aeb_df = mdp_df[agg_data_names].groupby('epi').agg(DATA_AGG_FNS)
        aeb_df.reset_index(drop=False, inplace=True)
        session_mdp_data[aeb], session_data[aeb] = mdp_df, aeb_df
    logger.debug(f'{session_data}')
    return session_mdp_data, session_data


def plot_session(session_spec, session_data):
    '''Plot the session graph, 2 panes: reward, loss & explore_var. Each aeb_df gets its own color'''
    aeb_count = len(session_data)
    if aeb_count <= 8:
        palette = cl.scales[str(max(3, aeb_count))]['qual']['Set2']
    else:
        palette = util.interp(cl.scales['8']['qual']['Set2'], aeb_count)
    fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)
    for idx, (a, e, b) in enumerate(session_data):
        aeb_str = f'{a}{e}{b}'
        aeb_df = session_data[(a, e, b)]
        fig_1 = viz.plot_line(
            aeb_df, 'reward', 'epi', legend_name=aeb_str, draw=False, trace_kwargs={'legendgroup': aeb_str, 'line': {'color': palette[idx]}})
        fig.append_trace(fig_1.data[0], 1, 1)

        fig_2 = viz.plot_line(
            aeb_df, ['loss'], 'epi', y2_col=['explore_var'], trace_kwargs={'legendgroup': aeb_str, 'showlegend': False, 'line': {'color': palette[idx]}}, draw=False)
        fig.append_trace(fig_2.data[0], 2, 1)
        fig.append_trace(fig_2.data[1], 3, 1)

    fig.layout['xaxis1'].update(title='epi', zerolinewidth=1)
    fig.layout['yaxis1'].update(fig_1.layout['yaxis'])
    fig.layout['yaxis1'].update(domain=[0.55, 1])
    fig.layout['yaxis2'].update(fig_2.layout['yaxis'])
    fig.layout['yaxis2'].update(showgrid=False, domain=[0, 0.45])
    fig.layout['yaxis3'].update(fig_2.layout['yaxis2'])
    fig.layout['yaxis3'].update(overlaying='y2', anchor='x2')
    fig.layout.update(_.pick(fig_1.layout, ['legend']))
    fig.layout.update(
        title=f'session graph: {session_spec["name"]}', width=500, height=600)
    viz.plot(fig)
    return fig


def calc_session_fitness_df(session, session_data):
    '''Calculate the session fitness df'''
    session_fitness_data = {}
    for aeb in session_data:
        aeb_df = session_data[aeb]
        body = session.aeb_space.body_space.data[aeb]
        aeb_fitness_sr = calc_aeb_fitness_sr(aeb_df, body.env.name)
        aeb_fitness_df = pd.DataFrame([aeb_fitness_sr], index=[session.index])
        session_fitness_data[aeb] = aeb_fitness_df
    # form multiindex df, then take mean across all bodies
    session_fitness_df = pd.concat(session_fitness_data, axis=1)
    mean_fitness_df = session_fitness_df.mean(axis=1, level=3)
    session_fitness = calc_fitness(mean_fitness_df)
    logger.info(f'Session mean fitness: {session_fitness}\n{mean_fitness_df}')
    return session_fitness_df


# TODO persist each session's full data to DB from here
def save_session_data(info_space, session_spec, session_mdp_data, session_data, session_fitness_df, session_fig):
    '''
    Save the session data: session_mdp_df, session_df, session_fitness_df, session_graph.
    session_data is saved as session_df; multi-indexed with (a,e,b), 3 extra levels
    to read, use:
    session_df = util.read(filepath, header=[0, 1, 2, 3])
    session_data = util.session_df_to_data(session_df)
    Likewise for session_mdp_df
    '''
    spec_name = session_spec['name']
    trial_index = info_space.coor['trial']
    session_index = info_space.coor['session']
    predir = f'data/{spec_name}_{info_space.experiment_ts}'
    prename = f'{spec_name}_t{trial_index}_s{session_index}'
    logger.info(
        f'Saving trial {trial_index} session {session_index} data to {predir}')
    session_mdp_df = pd.concat(session_mdp_data, axis=1)
    session_df = pd.concat(session_data, axis=1)
    util.write(session_mdp_df, f'{predir}/{prename}_session_mdp_df.csv')
    util.write(session_df, f'{predir}/{prename}_session_df.csv')
    util.write(session_fitness_df,
               f'{predir}/{prename}_session_fitness_df.csv')
    # TODO replaced by plot_best_sessions until Feb 2018
    # viz.save_image(session_fig, f'{predir}/{prename}_session_graph.png')


def analyze_session(session):
    '''
    Gather session data, plot, and return fitness df for high level agg.
    @returns {DataFrame} session_fitness_df Single-row df of session fitness vector (avg over aeb), indexed with session index.
    '''
    session_mdp_data, session_data = get_session_data(session)
    session_fitness_df = calc_session_fitness_df(session, session_data)
    session_fig = plot_session(session.spec, session_data)
    save_session_data(
        session.info_space, session.spec, session_mdp_data, session_data, session_fitness_df, session_fig)
    return session_fitness_df


def calc_trial_fitness_df(trial):
    '''
    Calculate the trial fitness df by aggregating from the collected session_data_dict (session_fitness_df's).
    Adds a consistency dimension to fitness vector.
    '''
    trial_fitness_data = {}
    all_session_fitness_df = pd.concat(
        list(trial.session_data_dict.values()))
    for aeb in util.get_df_aeb_list(all_session_fitness_df):
        aeb_df = all_session_fitness_df.loc[:, aeb]
        aeb_fitness_sr = aeb_df.mean()
        consistency = calc_consistency(aeb_df.values)
        aeb_fitness_sr = aeb_fitness_sr.append(
            pd.Series({'consistency': consistency}))
        aeb_fitness_df = pd.DataFrame([aeb_fitness_sr], index=[trial.index])
        trial_fitness_data[aeb] = aeb_fitness_df
    # form multiindex df, then take mean across all bodies
    trial_fitness_df = pd.concat(trial_fitness_data, axis=1)
    mean_fitness_df = trial_fitness_df.mean(axis=1, level=3)
    trial_fitness_df = mean_fitness_df
    trial_fitness = calc_fitness(mean_fitness_df)
    logger.info(f'Trial mean fitness: {trial_fitness}\n{mean_fitness_df}')
    return trial_fitness_df


def save_trial_data(info_space, trial_spec, trial_fitness_df):
    '''Save the trial data: spec, trial_fitness_df.'''
    spec_name = trial_spec['name']
    trial_index = info_space.coor['trial']
    predir = f'data/{spec_name}_{info_space.experiment_ts}'
    prename = f'{spec_name}_t{trial_index}'
    logger.info(f'Saving trial {trial_index} data to {predir}')
    util.write(trial_spec, f'{predir}/{prename}_spec.json')
    # TODO trial data is composed of saved session data files
    util.write(trial_fitness_df, f'{predir}/{prename}_trial_fitness_df.csv')


def analyze_trial(trial):
    '''
    Gather trial data, plot, and return trial df for high level agg.
    @returns {DataFrame} trial_fitness_df Single-row df of trial fitness vector (avg over aeb, sessions), indexed with trial index.
    '''
    trial_fitness_df = calc_trial_fitness_df(trial)
    save_trial_data(trial.info_space, trial.spec, trial_fitness_df)
    return trial_fitness_df


def plot_experiment(experiment_spec, experiment_df):
    '''
    Plot the variable specs vs fitness vector of an experiment, where each point is a trial.
    ref colors: https://plot.ly/python/heatmaps-contours-and-2dhistograms-tutorial/#plotlys-predefined-color-scales
    '''
    y_cols = ['fitness'] + FITNESS_COLS
    x_cols = _.difference(experiment_df.columns.tolist(), y_cols)

    fig = viz.tools.make_subplots(
        rows=len(y_cols), cols=len(x_cols), shared_xaxes=True, shared_yaxes=True)
    fitness_sr = experiment_df['fitness']
    min_fitness = fitness_sr.values.min()
    max_fitness = fitness_sr.values.max()
    for row_idx, y in enumerate(y_cols):
        for col_idx, x in enumerate(x_cols):
            x_sr = experiment_df[x]
            guard_cat_x = x_sr.astype(str) if x_sr.dtype == 'object' else x_sr
            trace = viz.go.Scatter(
                y=experiment_df[y], yaxis=f'y{row_idx+1}',
                x=guard_cat_x, xaxis=f'x{col_idx+1}',
                showlegend=False, mode='markers',
                marker={
                    'symbol': 'circle-open-dot', 'color': experiment_df['fitness'],
                    # dump first quarter of colorscale that is too bright
                    'cmin': min_fitness - 0.25 * (max_fitness - min_fitness), 'cmax': max_fitness,
                    'colorscale': 'YIGnBu', 'reversescale': True
                },
            )
            fig.append_trace(trace, row_idx + 1, col_idx + 1)
            fig.layout[f'xaxis{col_idx+1}'].update(
                title='<br>'.join(_.chunk(x, 20)), zerolinewidth=1, categoryarray=sorted(guard_cat_x.unique()))
        fig.layout[f'yaxis{row_idx+1}'].update(title=y, rangemode='tozero')
    fig.layout.update(
        title=f'experiment graph: {experiment_spec["name"]}', width=len(x_cols) * 200, height=700)
    viz.plot(fig)
    return fig


def save_experiment_data(info_space, best_spec, experiment_df, experiment_fig):
    '''Save the experiment data: best_spec, experiment_df, experiment_graph.'''
    spec_name = best_spec['name']
    predir = f'data/{spec_name}_{info_space.experiment_ts}'
    prename = f'{spec_name}'
    logger.info(f'Saving experiment data to {predir}')
    util.write(best_spec, f'{predir}/{prename}_best_spec.json')
    util.write(experiment_df, f'{predir}/{prename}_experiment_df.csv')
    viz.save_image(experiment_fig, f'{predir}/{prename}_experiment_graph.png')
    plot_best_sessions(experiment_df, predir, prename)


def analyze_experiment(experiment):
    '''
    Gather experiment trial_data_dict as experiment_df, plot.
    Search module must return best_spec and experiment_data with format {trial_index: exp_trial_data},
    where trial_data = {**var_spec, **fitness_vec, fitness}.
    This is then made into experiment_df.
    @returns {DataFrame} experiment_df Of var_specs, fitness_vec, fitness for all trials.
    '''
    experiment_df = pd.DataFrame(experiment.trial_data_dict).transpose()
    cols = FITNESS_COLS + ['fitness']
    config_cols = sorted(_.difference(
        experiment_df.columns.tolist(), cols))
    sorted_cols = config_cols + cols
    experiment_df = experiment_df.reindex(sorted_cols, axis=1)
    experiment_df.sort_values(by=['fitness'], ascending=False, inplace=True)
    logger.info(f'Experiment data:\n{experiment_df}')
    experiment_fig = plot_experiment(experiment.spec, experiment_df)
    best_config = experiment_df.iloc[0][config_cols].to_dict()
    best_spec = _.merge(experiment.spec, best_config)
    save_experiment_data(
        experiment.info_space, best_spec, experiment_df, experiment_fig)
    return experiment_df


def plot_session_from_file(session_df_filepath):
    '''
    Method to plot session from its session_df file
    @example

    from slm_lab.experiment import analysis

    filepath = 'data/reinforce_cartpole_2018_01_22_211751/reinforce_cartpole_t0_s0_session_df.csv'
    analysis.plot_session_from_file(filepath)
    '''
    spec_name = '_'.join(session_df_filepath.split('/')[1].split('_')[:-4])
    session_spec = {'name': spec_name}
    session_df = util.read(session_df_filepath, header=[0, 1, 2, 3])
    session_data = util.session_df_to_data(session_df)
    session_fig = plot_session(session_spec, session_data)
    viz.save_image(session_fig, session_df_filepath.replace(
        '_session_df.csv', '_session_graph.png'))


def plot_experiment_from_file(experiment_df_filepath):
    '''
    Method to plot experiment from its experiment_df file
    @example

    from slm_lab.experiment import analysis

    filepath = 'data/reinforce_cartpole_2018_01_22_190720/reinforce_cartpole_experiment_df.csv'
    analysis.plot_experiment_from_file(filepath)
    '''
    spec_name = '_'.join(experiment_df_filepath.split('/')[1].split('_')[:-4])
    experiment_spec = {'name': spec_name}
    experiment_df = util.read(experiment_df_filepath)
    experiment_fig = plot_experiment(experiment_spec, experiment_df)
    viz.save_image(experiment_fig, experiment_df_filepath.replace(
        '_experiment_df.csv', '_experiment_graph.png'))


def plot_best_sessions(experiment_df, predir, prename):
    '''
    Plot the session graphs from the best trials.
    TODO retire and plot all when Plotly allows unlimited plotting in Feb 2018
    '''
    for trial_index in experiment_df.index[:5]:
        session_prename = f'{prename}_t{trial_index}_s{0}'
        session_df_filepath = f'{predir}/{session_prename}_session_df.csv'
        plot_session_from_file(session_df_filepath)


'''
Fitness analysis
'''


def calc_strength(aeb_df, rand_epi_reward, std_epi_reward):
    '''
    Calculate the strength for each episode:
    strength_epi = (epi_reward - rand_epi_reward) / (std_epi_reward - rand_epi_reward)
    Propeties:
    - random agent has strength ~0, baseline agent has strength ~1.
    - if an agent achieve x2 rewards, the strength is ~x2, and so on.
    - strength of learning agent always tends toward positive regardless of the sign of rewards
    - scale of strength is always standard at 1 and its multiplies, regardless of the scale of actual rewards. Strength stays invariant even as reward gets rescaled.
    This allows for standard comparison between agents on the same problem using an intuitive measurement of strength. With proper scaling by a difficulty factor, we can compare across problems of different difficulties.
    '''
    # use lower clip 0 for noise in reward to dip slighty below rand
    return (aeb_df['reward'] - rand_epi_reward).clip(0) / (std_epi_reward - rand_epi_reward)


def calc_stable_idx(aeb_df):
    '''Calculate the index (epi) when strength first becomes stable (using moving mean and working backward)'''
    std_strength = 1
    above_std_strength_sr = (aeb_df['strength_ma'] >= std_strength)
    if above_std_strength_sr.any():
        # if it achieved stable (ma) std_strength at some point, the index when
        std_strength_ra_idx = above_std_strength_sr.idxmax()
        stable_idx = std_strength_ra_idx - (MA_WINDOW - 1)
    else:
        stable_idx = np.nan
    return stable_idx


def calc_std_strength_timestep(aeb_df):
    '''
    Calculate the timestep needed to achieve stable (within window) std_strength.
    For agent failing to achieve std_strength 1, it is meaningless to measure speed or give false interpolation, so set as inf (never).
    '''
    std_strength = 1
    stable_idx = calc_stable_idx(aeb_df)
    if np.isnan(stable_idx):
        std_strength_timestep = np.inf
    else:
        std_strength_timestep = aeb_df.loc[
            stable_idx, 'total_t'] / std_strength
    return std_strength_timestep


def calc_speed(aeb_df, std_timestep):
    '''
    Calculate the speed (using absolute timesteps) to attain std_strength 1:
    speed = std_timestep / agent_timestep
    Propeties:
    - random agent has speed ~0, baseline agent has speed ~1
    - if an agent takes x2 timesteps to read std_strength, we can it is 2x slower.
    - speed of learning agent always tends toward positive regardless of the shape of rewards curve
    - scale of speed is always standard at 1 and its multiplies, regardless of absolute timestep.
    This allows an intuitive measurement of learning speed and the standard comparison between agents on the same problem. Absolute timestep also measures the bits of new information given to the agent, which is a more grounded metric. With proper scaling of timescale (or bits scale), we can compare across problems of different difficulties.
    For agent failing to achieve std_strength 1, it is meaningless to measure speed or give false interpolation, so speed is 0.
    '''
    agent_timestep = calc_std_strength_timestep(aeb_df)
    speed = std_timestep / agent_timestep
    return speed


def is_noisy_mono_inc(sr):
    '''Check if sr is monotonically increasing, within noise = 5% * std_strength = 0.05 * 1'''
    zero_noise = -0.05
    mono_inc_sr = np.diff(sr) >= zero_noise
    # restore sr to same length
    mono_inc_sr = np.insert(mono_inc_sr, 0, np.nan)
    return mono_inc_sr


def calc_stability(aeb_df):
    '''
    Calculate the stability at maintaining std_strength and higher:
    stability = ratio of times strength is monotonically increasing with 5% allowance for noise since becoming stable.
    Propeties:
    - considers once strength becomes stable (note, stable does not imply stability = 1)
    - allows for drop in strength of 5% of std_strength, which is invariant to the scale of rewards
    - if strength is monotonically increasing (with 5% noise), then it is stable
    - sharp gain in strength is considered stable
    - works even for partial solution (not attaining std_strength), due to how stable_idx is calculated
    When an agent fails to achieve std_strength, it is meaningless to measure stability or give false interpolation, so stability is 0.
    '''
    stable_idx = calc_stable_idx(aeb_df)
    if np.isnan(stable_idx):
        stability = 0
    else:
        stable_df = aeb_df.loc[stable_idx:, 'strength_mono_inc']
        stability = stable_df.sum() / len(stable_df)
    return stability


def calc_consistency(fitness_vecs):
    '''
    Calculate the consistency of trial by the fitness_vectors of its sessions:
    consistency = ratio of non-outlier vectors
    Properties:
    - outliers are calculated using MAD modified z-score
    - if all the fitness vectors are zero, consistency = 0
    - works for all sorts of session fitness vectors, with the standard scale
    '''
    if ~np.any(fitness_vecs):
        # no consistency if vectors all 0
        consistency = 0
    else:
        is_outlier_arr = util.is_outlier(fitness_vecs)
        consistency = (~is_outlier_arr).sum() / len(is_outlier_arr)
    return consistency


def calc_fitness(fitness_vec):
    '''
    Takes a vector of qualifying standardized dimensions of fitness and compute the normalized length as fitness
    L2 norm because it diminishes lower values but amplifies higher values for comparison.
    '''
    if isinstance(fitness_vec, pd.Series):
        fitness_vec = fitness_vec.values
    elif isinstance(fitness_vec, pd.DataFrame):
        fitness_vec = fitness_vec.iloc[0].values
    std_fitness_vector = np.ones(len(fitness_vec))
    fitness = np.linalg.norm(fitness_vec) / np.linalg.norm(std_fitness_vector)
    return fitness


def calc_aeb_fitness_sr(aeb_df, env_name):
    '''Top level method to calculate fitness vector for AEB level data (strength, speed, stability)'''
    logger.info('Dev feature: fitness computation')
    no_fitness_sr = pd.Series({
        'strength': 0, 'speed': 0, 'stability': 0})
    if len(aeb_df) < MA_WINDOW:
        logger.warn(
            f'Run more than {MA_WINDOW} episodes to compute proper fitness')
        return no_fitness_sr
    if env_name not in FITNESS_STD:
        logger.warn(
            f'The fitness standard for env {env_name} is not built yet. Contact author.')
        return no_fitness_sr
    std = FITNESS_STD.get(env_name)
    aeb_df['total_t'] = aeb_df['t'].cumsum()
    aeb_df['strength'] = calc_strength(
        aeb_df, std['rand_epi_reward'], std['std_epi_reward'])
    aeb_df['strength_ma'] = aeb_df['strength'].rolling(MA_WINDOW).mean()
    aeb_df['strength_mono_inc'] = is_noisy_mono_inc(
        aeb_df['strength']).astype(int)

    strength = aeb_df['strength_ma'].max()
    speed = calc_speed(aeb_df, std['std_timestep'])
    stability = calc_stability(aeb_df)
    aeb_fitness_sr = pd.Series({
        'strength': strength, 'speed': speed, 'stability': stability})
    return aeb_fitness_sr
