'''
The analysis module
Handles the analyses of the info and data space for experiment evaluation and design.
'''
from slm_lab.agent import AGENT_DATA_NAMES
from slm_lab.env import ENV_DATA_NAMES
from slm_lab.lib import logger, util, viz
import colorlover as cl
import numpy as np
import os
import pandas as pd
import pydash as ps

DATA_AGG_FNS = {
    't': 'sum',
    'reward': 'sum',
    'loss': 'mean',
    'explore_var': 'mean',
}
FITNESS_COLS = ['strength', 'speed', 'stability', 'consistency']
FITNESS_STD = util.read('slm_lab/spec/_fitness_std.json')
NOISE_WINDOW = 0.05
MA_WINDOW = 100
logger = logger.get_logger(__name__)


def calc_mean_fitness(fitness_df):
    '''Method to calculated mean over all bodies for a fitness_df'''
    return fitness_df.mean(axis=1, level=3)


def get_session_data(session):
    '''
    Gather data from session: MDP, Agent, Env data, hashed by aeb; then aggregate.
    @returns {dict, dict} session_mdp_data, session_data
    '''
    data_names = AGENT_DATA_NAMES + ENV_DATA_NAMES
    mdp_data_names = ['t', 'epi'] + data_names
    agg_data_names = ['epi'] + list(DATA_AGG_FNS.keys())
    data_h_v_dict = {data_name: session.aeb_space.get_history_v(data_name) for data_name in data_names}
    session_mdp_data, session_data = {}, {}
    for aeb in session.aeb_space.aeb_list:
        data_h_dict = {data_name: data_h_v[aeb] for data_name, data_h_v in data_h_v_dict.items()}
        # trim back to remove any incomplete sessions due to multienv termination
        complete_done_h = np.trim_zeros(data_h_dict['done'], 'b')
        # offset properly to bin separate episodes
        reset_bin = np.concatenate([[0.], complete_done_h[:-1]])
        data_len = len(reset_bin)
        reset_idx = reset_bin.astype('bool')
        nonreset_idx = ~reset_idx
        data_h_dict['t'] = np.ones(reset_idx.shape)
        data_h_dict['epi'] = reset_idx.astype(int).cumsum()
        mdp_df = pd.DataFrame({
            data_name: data_h_dict[data_name][:data_len]
            for data_name in mdp_data_names})
        mdp_df = mdp_df.reindex(mdp_data_names, axis=1)
        aeb_df = mdp_df[agg_data_names].groupby('epi').agg(DATA_AGG_FNS)
        aeb_df.reset_index(drop=False, inplace=True)
        session_mdp_data[aeb], session_data[aeb] = mdp_df, aeb_df
    logger.debug(f'{session_data}')
    data_size_in_bytes = util.memory_size(session_mdp_data)
    logger.debug(f'Size of session data: {data_size_in_bytes} MB')
    if data_size_in_bytes > 25:
        logger.warn(f'Session data > 25 MB')
    return session_mdp_data, session_data


def calc_session_fitness_df(session, session_data):
    '''Calculate the session fitness df'''
    session_fitness_data = {}
    for aeb in session_data:
        aeb_df = session_data[aeb]
        util.downcast_float32(aeb_df)
        body = session.aeb_space.body_space.data[aeb]
        aeb_fitness_sr = calc_aeb_fitness_sr(aeb_df, body.env.name)
        aeb_fitness_df = pd.DataFrame([aeb_fitness_sr], index=[session.index])
        aeb_fitness_df = aeb_fitness_df.reindex(FITNESS_COLS[:3], axis=1)
        session_fitness_data[aeb] = aeb_fitness_df
    # form multiindex df, then take mean across all bodies
    session_fitness_df = pd.concat(session_fitness_data, axis=1)
    mean_fitness_df = calc_mean_fitness(session_fitness_df)
    session_fitness = calc_fitness(mean_fitness_df)
    logger.info(f'Session mean fitness: {session_fitness}\n{mean_fitness_df}')
    return session_fitness_df


def is_unfit(fitness_df):
    '''Check if a fitness_df is unfit. Used to determine of trial should stop running more sessions'''
    mean_fitness_df = calc_mean_fitness(fitness_df)
    return mean_fitness_df['strength'].iloc[0] < NOISE_WINDOW


def calc_trial_fitness_df(trial):
    '''
    Calculate the trial fitness df by aggregating from the collected session_data_dict (session_fitness_df's).
    Adds a consistency dimension to fitness vector.
    '''
    trial_fitness_data = {}
    all_session_fitness_df = pd.concat(list(trial.session_data_dict.values()))
    for aeb in util.get_df_aeb_list(all_session_fitness_df):
        aeb_fitness_df = all_session_fitness_df.loc[:, aeb]
        aeb_fitness_sr = aeb_fitness_df.mean()
        consistency = calc_consistency(aeb_fitness_df)
        aeb_fitness_sr = aeb_fitness_sr.append(pd.Series({'consistency': consistency}))
        aeb_fitness_df = pd.DataFrame([aeb_fitness_sr], index=[trial.index])
        aeb_fitness_df = aeb_fitness_df.reindex(FITNESS_COLS, axis=1)
        trial_fitness_data[aeb] = aeb_fitness_df
    # form multiindex df, then take mean across all bodies
    trial_fitness_df = pd.concat(trial_fitness_data, axis=1)
    mean_fitness_df = calc_mean_fitness(trial_fitness_df)
    trial_fitness_df = mean_fitness_df
    trial_fitness = calc_fitness(mean_fitness_df)
    logger.info(f'Trial mean fitness: {trial_fitness}\n{mean_fitness_df}')
    return trial_fitness_df


def get_palette(aeb_count):
    '''Get the suitable palette to plot for some number of aeb graphs, where each aeb is a color.'''
    if aeb_count <= 8:
        palette = cl.scales[str(max(3, aeb_count))]['qual']['Set2']
    else:
        palette = util.interp(cl.scales['8']['qual']['Set2'], aeb_count)
    return palette


def lower_opacity(rgb, opacity):
    return rgb.replace('rgb(', 'rgba(').replace(')', f',{opacity})')


def plot_session(session_spec, info_space, session_data):
    '''Plot the session graph, 2 panes: reward, loss & explore_var. Each aeb_df gets its own color'''
    aeb_count = len(session_data)
    palette = get_palette(aeb_count)
    fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)
    for idx, (a, e, b) in enumerate(session_data):
        aeb_str = f'{a}{e}{b}'
        aeb_df = session_data[(a, e, b)]
        aeb_df.fillna(0, inplace=True)  # for saving plot, cant have nan
        fig_1 = viz.plot_line(aeb_df, 'reward', 'epi', legend_name=aeb_str, draw=False, trace_kwargs={'legendgroup': aeb_str, 'line': {'color': palette[idx]}})
        fig.append_trace(fig_1.data[0], 1, 1)

        fig_2 = viz.plot_line(aeb_df, ['loss'], 'epi', y2_col=['explore_var'], trace_kwargs={'legendgroup': aeb_str, 'showlegend': False, 'line': {'color': palette[idx]}}, draw=False)
        fig.append_trace(fig_2.data[0], 2, 1)
        fig.append_trace(fig_2.data[1], 3, 1)

    fig.layout['xaxis1'].update(title='epi', zerolinewidth=1)
    fig.layout['yaxis1'].update(fig_1.layout['yaxis'])
    fig.layout['yaxis1'].update(domain=[0.55, 1])
    fig.layout['yaxis2'].update(fig_2.layout['yaxis'])
    fig.layout['yaxis2'].update(showgrid=False, domain=[0, 0.45])
    fig.layout['yaxis3'].update(fig_2.layout['yaxis2'])
    fig.layout['yaxis3'].update(overlaying='y2', anchor='x2')
    fig.layout.update(ps.pick(fig_1.layout, ['legend']))
    fig.layout.update(title=f'session graph: {session_spec["name"]} t{info_space.get("trial")} s{info_space.get("session")}', width=500, height=600)
    viz.plot(fig)
    return fig


def gather_aeb_rewards_df(aeb, session_datas):
    '''Gather rewards from each session for a body into a df'''
    aeb_session_rewards = {}
    for s, session_data in session_datas.items():
        aeb_df = session_data[aeb]
        aeb_reward_sr = aeb_df['reward']
        aeb_session_rewards[s] = aeb_reward_sr
    aeb_rewards_df = pd.DataFrame(aeb_session_rewards)
    return aeb_rewards_df


def build_aeb_reward_fig(aeb_rewards_df, aeb_str, color):
    '''Build the aeb_reward envelope figure'''
    mean_sr = aeb_rewards_df.mean(axis=1)
    std_sr = aeb_rewards_df.std(axis=1).fillna(0)
    max_sr = mean_sr + std_sr
    min_sr = mean_sr - std_sr
    x = aeb_rewards_df.index.tolist()
    max_y = max_sr.tolist()
    min_y = min_sr.tolist()

    envelope_trace = viz.go.Scatter(
        x=x + x[::-1],
        y=max_y + min_y[::-1],
        fill='tozerox',
        fillcolor=lower_opacity(color, 0.2),
        line=dict(color='transparent'),
        showlegend=False,
        legendgroup=aeb_str,
    )
    df = pd.DataFrame({'epi': x, 'mean_reward': mean_sr})
    fig = viz.plot_line(
        df, ['mean_reward'], ['epi'], legend_name=aeb_str, draw=False, trace_kwargs={'legendgroup': aeb_str, 'line': {'color': color}}
    )
    fig.data.append(envelope_trace)
    return fig


def plot_trial(trial_spec, info_space):
    '''Plot the trial graph, 1 pane: mean and error envelope of reward graphs from all sessions. Each aeb_df gets its own color'''
    prepath = util.get_prepath(trial_spec, info_space)
    predir = util.prepath_to_predir(prepath)
    session_datas = session_datas_from_file(predir, trial_spec, info_space.get('trial'))

    aeb_count = len(session_datas[0])
    palette = get_palette(aeb_count)
    fig = None
    for idx, (a, e, b) in enumerate(session_datas[0]):
        aeb = (a, e, b)
        aeb_str = f'{a}{e}{b}'
        color = palette[idx]
        aeb_rewards_df = gather_aeb_rewards_df(aeb, session_datas)
        aeb_fig = build_aeb_reward_fig(aeb_rewards_df, aeb_str, color)
        if fig is None:
            fig = aeb_fig
        else:
            fig.data.extend(aeb_fig.data)
    fig.layout.update(title=f'trial graph: {trial_spec["name"]} t{info_space.get("trial")}', width=500, height=600)
    viz.plot(fig)
    return fig


def plot_experiment(experiment_spec, experiment_df):
    '''
    Plot the variable specs vs fitness vector of an experiment, where each point is a trial.
    ref colors: https://plot.ly/python/heatmaps-contours-and-2dhistograms-tutorial/#plotlys-predefined-color-scales
    '''
    y_cols = ['fitness'] + FITNESS_COLS
    x_cols = ps.difference(experiment_df.columns.tolist(), y_cols)

    fig = viz.tools.make_subplots(rows=len(y_cols), cols=len(x_cols), shared_xaxes=True, shared_yaxes=True)
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
                    'symbol': 'circle-open-dot', 'color': experiment_df['fitness'], 'opacity': 0.5,
                    # dump first quarter of colorscale that is too bright
                    'cmin': min_fitness - 0.50 * (max_fitness - min_fitness), 'cmax': max_fitness,
                    'colorscale': 'YIGnBu', 'reversescale': True
                },
            )
            fig.append_trace(trace, row_idx + 1, col_idx + 1)
            fig.layout[f'xaxis{col_idx+1}'].update(title='<br>'.join(ps.chunk(x, 20)), zerolinewidth=1, categoryarray=sorted(guard_cat_x.unique()))
        fig.layout[f'yaxis{row_idx+1}'].update(title=y, rangemode='tozero')
    fig.layout.update(title=f'experiment graph: {experiment_spec["name"]}', width=max(600, len(x_cols) * 300), height=700)
    viz.plot(fig)
    return fig


# TODO persist each session's full data to DB from here
def save_session_data(spec, info_space, session_mdp_data, session_data, session_fitness_df, session_fig):
    '''
    Save the session data: session_mdp_df, session_df, session_fitness_df, session_graph.
    session_data is saved as session_df; multi-indexed with (a,e,b), 3 extra levels
    to read, use:
    session_df = util.read(filepath, header=[0, 1, 2, 3])
    session_data = util.session_df_to_data(session_df)
    Likewise for session_mdp_df
    '''
    prepath = util.get_prepath(spec, info_space, unit='session')
    logger.info(f'Saving session data to {prepath}')
    if session_mdp_data is not None:  # not from retro analysis
        session_mdp_df = pd.concat(session_mdp_data, axis=1)
        session_df = pd.concat(session_data, axis=1)
        # TODO reactivate saving when get to the transition matrix research
        # util.write(session_mdp_df, f'{prepath}_session_mdp_df.csv')
        util.write(session_df, f'{prepath}_session_df.csv')
    util.write(session_fitness_df, f'{prepath}_session_fitness_df.csv')
    viz.save_image(session_fig, f'{prepath}_session_graph.png')


def save_trial_data(spec, info_space, trial_fitness_df, trial_fig):
    '''Save the trial data: spec, trial_fitness_df.'''
    prepath = util.get_prepath(spec, info_space, unit='trial')
    logger.info(f'Saving trial data to {prepath}')
    util.write(trial_fitness_df, f'{prepath}_trial_fitness_df.csv')
    viz.save_image(trial_fig, f'{prepath}_trial_graph.png')


def save_experiment_data(spec, info_space, experiment_df, experiment_fig):
    '''Save the experiment data: best_spec, experiment_df, experiment_graph.'''
    prepath = util.get_prepath(spec, info_space, unit='experiment')
    logger.info(f'Saving experiment data to {prepath}')
    util.write(experiment_df, f'{prepath}_experiment_df.csv')
    viz.save_image(experiment_fig, f'{prepath}_experiment_graph.png')


def analyze_session(session, session_data=None):
    '''
    Gather session data, plot, and return fitness df for high level agg.
    @returns {DataFrame} session_fitness_df Single-row df of session fitness vector (avg over aeb), indexed with session index.
    '''
    logger.info('Analyzing session')
    if session_data is None:
        session_mdp_data, session_data = get_session_data(session)
    else:  # from retro analysis
        session_mdp_data = None
    session_fitness_df = calc_session_fitness_df(session, session_data)
    session_fig = plot_session(session.spec, session.info_space, session_data)
    save_session_data(session.spec, session.info_space, session_mdp_data, session_data, session_fitness_df, session_fig)
    return session_fitness_df


def analyze_trial(trial):
    '''
    Gather trial data, plot, and return trial df for high level agg.
    @returns {DataFrame} trial_fitness_df Single-row df of trial fitness vector (avg over aeb, sessions), indexed with trial index.
    '''
    logger.info('Analyzing trial')
    trial_fitness_df = calc_trial_fitness_df(trial)
    trial_fig = plot_trial(trial.spec, trial.info_space)
    save_trial_data(trial.spec, trial.info_space, trial_fitness_df, trial_fig)
    return trial_fitness_df


def analyze_experiment(experiment):
    '''
    Gather experiment trial_data_dict as experiment_df, plot.
    Search module must return best_spec and experiment_data with format {trial_index: exp_trial_data},
    where trial_data = {**var_spec, **fitness_vec, fitness}.
    This is then made into experiment_df.
    @returns {DataFrame} experiment_df Of var_specs, fitness_vec, fitness for all trials.
    '''
    logger.info('Analyzing experiment')
    experiment_df = pd.DataFrame(experiment.trial_data_dict).transpose()
    cols = FITNESS_COLS + ['fitness']
    config_cols = sorted(ps.difference(experiment_df.columns.tolist(), cols))
    sorted_cols = config_cols + cols
    experiment_df = experiment_df.reindex(sorted_cols, axis=1)
    experiment_df.sort_values(by=['fitness'], ascending=False, inplace=True)
    logger.info(f'Experiment data:\n{experiment_df}')
    experiment_fig = plot_experiment(experiment.spec, experiment_df)
    save_experiment_data(experiment.spec, experiment.info_space, experiment_df, experiment_fig)
    return experiment_df


def save_spec(spec, info_space, unit='experiment'):
    '''Save spec to proper path. Called at Experiment or Trial init.'''
    prepath = util.get_prepath(spec, info_space, unit)
    util.write(spec, f'{prepath}_spec.json')


def session_data_from_file(predir, trial_index, session_index):
    '''Build session.session_data from file'''
    for filename in os.listdir(predir):
        if filename.endswith(f'_t{trial_index}_s{session_index}_session_df.csv'):
            filepath = f'{predir}/{filename}'
            session_df = util.read(filepath, header=[0, 1, 2, 3], index_col=0)
            session_data = util.session_df_to_data(session_df)
            return session_data


def session_datas_from_file(predir, trial_spec, trial_index):
    '''Return a dict of {session_index: session_data} for a trial'''
    session_datas = {}
    for s in range(trial_spec['meta']['max_session']):
        session_data = session_data_from_file(predir, trial_index, s)
        if session_data is not None:
            session_datas[s] = session_data
    return session_datas


def session_data_dict_from_file(predir, trial_index):
    '''Build trial.session_data_dict from file'''
    session_data_dict = {}
    for filename in os.listdir(predir):
        if f'_t{trial_index}_' in filename and filename.endswith('_session_fitness_df.csv'):
            filepath = f'{predir}/{filename}'
            fitness_df = util.read(filepath, header=[0, 1, 2, 3], index_col=0, dtype=np.float32)
            util.fix_multiindex_dtype(fitness_df)
            session_index = fitness_df.index[0]
            session_data_dict[session_index] = fitness_df
    return session_data_dict


def trial_data_dict_from_file(predir):
    '''Build experiment.trial_data_dict from file'''
    trial_data_dict = {}
    for filename in os.listdir(predir):
        if filename.endswith('_trial_data.json'):
            filepath = f'{predir}/{filename}'
            exp_trial_data = util.read(filepath)
            trial_index = exp_trial_data.pop('trial_index')
            trial_data_dict[trial_index] = exp_trial_data
    return trial_data_dict


# TODO unify with util method
def mock_info_space_spec(predir, trial_index=None, session_index=None):
    '''Helper for retro analysis to build mock info_space and spec'''
    from slm_lab.experiment.monitor import InfoSpace
    spec_name = util.prepath_to_spec_name(predir)
    experiment_ts = util.prepath_to_experiment_ts(predir)
    info_space = InfoSpace()
    info_space.experiment_ts = experiment_ts
    info_space.set('experiment', 0)
    if trial_index is None:
        filepath = f'{predir}/{spec_name}_spec.json'
    else:
        info_space.set('trial', trial_index)
        filepath = f'{predir}/{spec_name}_t{trial_index}_spec.json'
    if session_index is not None:
        info_space.set('session', session_index)
    spec = util.read(filepath)
    return spec, info_space


def retro_analyze_sessions(predir):
    '''Retro-analyze all session level datas.'''
    logger.info('Retro-analyzing sessions from file')
    from slm_lab.experiment.control import Session
    for filename in os.listdir(predir):
        if filename.endswith('_session_df.csv'):
            tn, sn = filename.replace('_session_df.csv', '').split('_')[-2:]
            trial_index, session_index = int(tn[1:]), int(sn[1:])
            # mock session
            spec, info_space = mock_info_space_spec(predir, trial_index, session_index)
            session = Session(spec, info_space)
            session_data = session_data_from_file(predir, trial_index, session_index)
            analyze_session(session, session_data)


def retro_analyze_trials(predir):
    '''Retro-analyze all trial level datas.'''
    logger.info('Retro-analyzing trials from file')
    from slm_lab.experiment.control import Trial
    for filename in os.listdir(predir):
        if filename.endswith('_trial_data.json'):
            filepath = f'{predir}/{filename}'
            tn = filename.replace('_trial_data.json', '').split('_')[-1]
            trial_index = int(tn[1:])
            # mock trial
            spec, info_space = mock_info_space_spec(predir, trial_index)
            trial = Trial(spec, info_space)
            session_data_dict = session_data_dict_from_file(predir, trial_index)
            trial.session_data_dict = session_data_dict
            trial_fitness_df = analyze_trial(trial)
            # write trial_data that was written from ray search
            fitness_vec = trial_fitness_df.iloc[0].to_dict()
            fitness = calc_fitness(trial_fitness_df)
            trial_data = util.read(filepath)
            trial_data.update({
                **fitness_vec, 'fitness': fitness, 'trial_index': trial_index,
            })
            util.write(trial_data, filepath)


def retro_analyze_experiment(predir):
    '''Retro-analyze all experiment level datas.'''
    logger.info('Retro-analyzing experiment from file')
    from slm_lab.experiment.control import Experiment
    # mock experiment
    spec, info_space = mock_info_space_spec(predir)
    experiment = Experiment(spec, info_space)
    trial_data_dict = trial_data_dict_from_file(predir)
    experiment.trial_data_dict = trial_data_dict
    return analyze_experiment(experiment)


def retro_analyze(predir):
    '''
    Method to analyze experiment from file after it ran.
    Read from files, constructs lab units, run retro analyses on all lab units.
    This method has no side-effects, i.e. doesn't overwrite data it should not.
    @example

    from slm_lab.experiment import analysis
    predir = 'data/reinforce_cartpole_2018_01_22_211751'
    analysis.retro_analyze(predir)
    '''
    os.environ['PREPATH'] = f'{predir}/retro_analyze'  # to prevent overwriting log file
    logger.info(f'Retro-analyzing {predir}')
    retro_analyze_sessions(predir)
    retro_analyze_trials(predir)
    retro_analyze_experiment(predir)


def plot_session_from_file(session_df_filepath):
    '''
    Method to plot session from its session_df file
    @example

    from slm_lab.experiment import analysis
    filepath = 'data/reinforce_cartpole_2018_01_22_211751/reinforce_cartpole_t0_s0_session_df.csv'
    analysis.plot_session_from_file(filepath)
    '''
    from slm_lab.experiment.monitor import InfoSpace
    spec_name = util.prepath_to_spec_name(session_df_filepath)
    session_spec = {'name': spec_name}
    session_df = util.read(session_df_filepath, header=[0, 1, 2, 3], index_col=0, dtype=np.float32)
    session_data = util.session_df_to_data(session_df)
    tn, sn = session_df_filepath.replace('_session_df.csv', '').split('_')[-2:]
    info_space = InfoSpace()
    info_space.set('experiment', 0)
    info_space.set('trial', int(tn[1:]))
    info_space.set('session', int(sn[1:]))
    session_fig = plot_session(session_spec, info_space, session_data)
    viz.save_image(session_fig, session_df_filepath.replace('_session_df.csv', '_session_graph.png'))


'''
Fitness analysis
'''


def calc_strength(aeb_df, rand_epi_reward, std_epi_reward):
    '''
    For each episode, use the total rewards to calculate the strength as
    strength_epi = (reward_epi - reward_rand) / (reward_std - reward_rand)
    **Properties:**
    - random agent has strength 0, standard agent has strength 1.
    - if an agent achieve x2 rewards, the strength is ~x2, and so on.
    - strength of learning agent always tends toward positive regardless of the sign of rewards (some environments use negative rewards)
    - scale of strength is always standard at 1 and its multiplies, regardless of the scale of actual rewards. Strength stays invariant even as reward gets rescaled.
    This allows for standard comparison between agents on the same problem using an intuitive measurement of strength. With proper scaling by a difficulty factor, we can compare across problems of different difficulties.
    '''
    # use lower clip 0 for noise in reward to dip slighty below rand
    return (aeb_df['reward'] - rand_epi_reward).clip(0.) / (std_epi_reward - rand_epi_reward)


def calc_stable_idx(aeb_df, min_strength_ma):
    '''Calculate the index (epi) when strength first becomes stable (using moving mean and working backward)'''
    above_std_strength_sr = (aeb_df['strength_ma'] >= min_strength_ma)
    if above_std_strength_sr.any():
        # if it achieved stable (ma) min_strength_ma at some point, the index when
        std_strength_ra_idx = above_std_strength_sr.idxmax()
        stable_idx = std_strength_ra_idx - (MA_WINDOW - 1)
    else:
        stable_idx = np.nan
    return stable_idx


def calc_std_strength_timestep(aeb_df):
    '''
    Calculate the timestep needed to achieve stable (within NOISE_WINDOW) std_strength.
    For agent failing to achieve std_strength 1, it is meaningless to measure speed or give false interpolation, so set as inf (never).
    '''
    std_strength = 1.
    stable_idx = calc_stable_idx(aeb_df, min_strength_ma=std_strength - NOISE_WINDOW)
    if np.isnan(stable_idx):
        std_strength_timestep = np.inf
    else:
        std_strength_timestep = aeb_df.loc[stable_idx, 'total_t'] / std_strength
    return std_strength_timestep


def calc_speed(aeb_df, std_timestep):
    '''
    For each session, measure the moving average for strength with interval = 100 episodes.
    Next, measure the total timesteps up to the first episode that first surpasses standard strength, allowing for noise of 0.05.
    Finally, calculate speed as
    speed = timestep_std / timestep_solved
    **Properties:**
    - random agent has speed 0, standard agent has speed 1.
    - if an agent takes x2 timesteps to exceed standard strength, we can say it is 2x slower.
    - the speed of learning agent always tends toward positive regardless of the shape of the rewards curve
    - the scale of speed is always standard at 1 and its multiplies, regardless of the absolute timesteps.
    For agent failing to achieve standard strength 1, it is meaningless to measure speed or give false interpolation, so the speed is 0.
    This allows an intuitive measurement of learning speed and the standard comparison between agents on the same problem.
    '''
    agent_timestep = calc_std_strength_timestep(aeb_df)
    speed = std_timestep / agent_timestep
    return speed


def is_noisy_mono_inc(sr):
    '''Check if sr is monotonically increasing, (given NOISE_WINDOW = 5%) within noise = 5% * std_strength = 0.05 * 1'''
    zero_noise = -NOISE_WINDOW
    mono_inc_sr = np.diff(sr) >= zero_noise
    # restore sr to same length
    mono_inc_sr = np.insert(mono_inc_sr, 0, np.nan)
    return mono_inc_sr


def calc_stability(aeb_df):
    '''
    Find a baseline =
    - 0. + noise for very weak solution
    - max(strength_ma_epi) - noise for partial solution weak solution
    - 1. - noise for solution achieving standard strength and beyond
    So we get:
    - weak_baseline = 0. + noise
    - strong_baseline = min(max(strength_ma_epi), 1.) - noise
    - baseline = max(weak_baseline, strong_baseline)

    Let epi_baseline be the episode where baseline is first attained. Consider the episodes starting from epi_baseline, let #epi_+ be the number of episodes, and #epi_>= the number of episodes where strength_ma_epi is monotonically increasing.
    Calculate stability as
    stability = #epi_>= / #epi_+
    **Properties:**
    - stable agent has value 1, unstable agent < 1, and non-solution = 0.
    - allows for drops strength MA of 5% to account for noise, which is invariant to the scale of rewards
    - if strength is monotonically increasing (with 5% noise), then it is stable
    - sharp gain in strength is considered stable
    - monotonically increasing implies strength can keep growing and as long as it does not fall much, it is considered stable
    '''
    weak_baseline = 0. + NOISE_WINDOW
    strong_baseline = min(aeb_df['strength_ma'].max(), 1.) - NOISE_WINDOW
    baseline = max(weak_baseline, strong_baseline)
    stable_idx = calc_stable_idx(aeb_df, min_strength_ma=baseline)
    if np.isnan(stable_idx):
        stability = 0.
    else:
        stable_df = aeb_df.loc[stable_idx:, 'strength_mono_inc']
        stability = stable_df.sum() / len(stable_df)
    return stability


def calc_consistency(aeb_fitness_df):
    '''
    Calculate the consistency of trial by the fitness_vectors of its sessions:
    consistency = ratio of non-outlier vectors
    **Properties:**
    - outliers are calculated using MAD modified z-score
    - if all the fitness vectors are zero or all strength are zero, consistency = 0
    - works for all sorts of session fitness vectors, with the standard scale
    When an agent fails to achieve standard strength, it is meaningless to measure consistency or give false interpolation, so consistency is 0.
    '''
    fitness_vecs = aeb_fitness_df.values
    if ~np.any(fitness_vecs) or ~np.any(aeb_fitness_df['strength']):
        # no consistency if vectors all 0
        consistency = 0.
    elif len(fitness_vecs) == 2:
        # if only has 2 vectors, check norm_diff
        diff_norm = np.linalg.norm(np.diff(fitness_vecs, axis=0)) / np.linalg.norm(np.ones(len(fitness_vecs[0])))
        consistency = diff_norm <= NOISE_WINDOW
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
    no_fitness_sr = pd.Series({
        'strength': 0., 'speed': 0., 'stability': 0.})
    if len(aeb_df) < MA_WINDOW:
        logger.warn(f'Run more than {MA_WINDOW} episodes to compute proper fitness')
        return no_fitness_sr
    std = FITNESS_STD.get(env_name)
    if std is None:
        std = FITNESS_STD.get('template')
        logger.warn(f'The fitness standard for env {env_name} is not built yet. Contact author. Using a template standard for now.')
    aeb_df['total_t'] = aeb_df['t'].cumsum()
    aeb_df['strength'] = calc_strength(aeb_df, std['rand_epi_reward'], std['std_epi_reward'])
    aeb_df['strength_ma'] = aeb_df['strength'].rolling(MA_WINDOW).mean()
    aeb_df['strength_mono_inc'] = is_noisy_mono_inc(aeb_df['strength']).astype(int)

    strength = aeb_df['strength_ma'].max()
    speed = calc_speed(aeb_df, std['std_timestep'])
    stability = calc_stability(aeb_df)
    aeb_fitness_sr = pd.Series({
        'strength': strength, 'speed': speed, 'stability': stability})
    return aeb_fitness_sr
