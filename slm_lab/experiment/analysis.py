'''
The analysis module
Handles the analyses of the info and data space for experiment evaluation and design.
'''
from slm_lab.agent import AGENT_DATA_NAMES
from slm_lab.env import ENV_DATA_NAMES
from slm_lab.lib import logger, math_util, util, viz
from slm_lab.spec import spec_util
import numpy as np
import os
import pandas as pd
import pydash as ps
import regex as re
import shutil

FITNESS_COLS = ['strength', 'speed', 'stability', 'consistency']
# TODO improve to make it work with any reward mean
FITNESS_STD = util.read('slm_lab/spec/_fitness_std.json')
NOISE_WINDOW = 0.05
NORM_ORDER = 1  # use L1 norm in fitness vector norm
MA_WINDOW = 100
logger = logger.get_logger(__name__)

'''
Fitness analysis
'''


def calc_strength_sr(aeb_df, rand_reward, std_reward):
    '''
    Calculate strength for each reward as
    strength = (reward - rand_reward) / (std_reward - rand_reward)
    '''
    return (aeb_df['reward'] - rand_reward) / (std_reward - rand_reward)


def calc_strength(aeb_df):
    '''
    Strength of an agent in fitness is its maximum strength_ma. Moving average is used to denoise signal.
    For an agent total reward at a time, calculate strength by normalizing it with a given baseline rand_reward and solution std_reward, i.e.
    strength = (reward - rand_reward) / (std_reward - rand_reward)

    **Properties:**
    - random agent has strength 0, standard agent has strength 1.
    - strength is standardized to be independent of the actual sign and scale of raw reward
    - scales relative to std_reward: if an agent achieve x2 std_reward, the strength is x2, and so on.
    This allows for standard comparison between agents on the same problem using an intuitive measurement of strength. With proper scaling by a difficulty factor, we can compare across problems of different difficulties.
    '''
    strength = aeb_df['strength_ma'].max()
    return max(0.0, strength)


def calc_speed(aeb_df, std_timestep):
    '''
    Find the maximum strength_ma, and the time to first reach it. Then the strength/time divided by the standard std_strength/std_timestep is speed, i.e.
    speed = (max_strength_ma / timestep_to_first_reach) / (std_strength / std_timestep)
    **Properties:**
    - random agent has speed 0, standard agent has speed 1.
    - if both agents reach the same max strength_ma, and one reaches it in half the timesteps, it is twice as fast.
    - speed is standardized regardless of the scaling of absolute timesteps, or even the max strength attained
    This allows an intuitive measurement of learning speed and the standard comparison between agents on the same problem.
    '''
    first_max_idx = aeb_df['strength_ma'].idxmax()  # this returns the first max
    max_row = aeb_df.loc[first_max_idx]
    std_strength = 1.
    if max_row['total_t'] == 0:  # especially for random agent
        speed = 0.
    else:
        speed = (max_row['strength_ma'] / max_row['total_t']) / (std_strength / std_timestep)
    return max(0., speed)


def calc_stability(aeb_df):
    '''
    Stability = fraction of monotonically increasing elements in the denoised series of strength_ma, or 0 if strength_ma is all <= 0.
    **Properties:**
    - stable agent has value 1, unstable agent < 1, and non-solution = 0.
    - uses strength_ma to be more robust to noise
    - sharp gain in strength is considered stable
    - monotonically increasing implies strength can keep growing and as long as it does not fall much, it is considered stable
    '''
    if (aeb_df['strength_ma'].values <= 0.).all():
        stability = 0.
    else:
        mono_inc_sr = np.diff(aeb_df['strength_ma']) >= 0.
        stability = mono_inc_sr.sum() / mono_inc_sr.size
    return max(0., stability)


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
        diff_norm = np.linalg.norm(np.diff(fitness_vecs, axis=0), NORM_ORDER) / np.linalg.norm(np.ones(len(fitness_vecs[0])), NORM_ORDER)
        consistency = diff_norm <= NOISE_WINDOW
    else:
        is_outlier_arr = math_util.is_outlier(fitness_vecs)
        consistency = (~is_outlier_arr).sum() / len(is_outlier_arr)
    return consistency


def calc_epi_reward_ma(aeb_df, ckpt=None):
    '''Calculates the episode reward moving average with the MA_WINDOW'''
    rewards = aeb_df['reward']
    if ckpt == 'eval':
        # online eval mode reward is reward_ma from avg
        aeb_df['reward_ma'] = rewards
    else:
        aeb_df['reward_ma'] = rewards.rolling(window=MA_WINDOW, min_periods=0, center=False).mean()
    return aeb_df


def calc_fitness(fitness_vec):
    '''
    Takes a vector of qualifying standardized dimensions of fitness and compute the normalized length as fitness
    use L1 norm for simplicity and intuititveness of linearity
    '''
    if isinstance(fitness_vec, pd.Series):
        fitness_vec = fitness_vec.values
    elif isinstance(fitness_vec, pd.DataFrame):
        fitness_vec = fitness_vec.iloc[0].values
    std_fitness_vector = np.ones(len(fitness_vec))
    fitness = np.linalg.norm(fitness_vec, NORM_ORDER) / np.linalg.norm(std_fitness_vector, NORM_ORDER)
    return fitness


def calc_aeb_fitness_sr(aeb_df, env_name):
    '''Top level method to calculate fitness vector for AEB level data (strength, speed, stability)'''
    std = FITNESS_STD.get(env_name)
    if std is None:
        std = FITNESS_STD.get('template')
        logger.warn(f'The fitness standard for env {env_name} is not built yet. Contact author. Using a template standard for now.')

    # calculate the strength sr and the moving-average (to denoise) first before calculating fitness
    aeb_df['strength'] = calc_strength_sr(aeb_df, std['rand_epi_reward'], std['std_epi_reward'])
    aeb_df['strength_ma'] = aeb_df['strength'].rolling(MA_WINDOW, min_periods=0, center=False).mean()

    strength = calc_strength(aeb_df)
    speed = calc_speed(aeb_df, std['std_timestep'])
    stability = calc_stability(aeb_df)
    aeb_fitness_sr = pd.Series({
        'strength': strength, 'speed': speed, 'stability': stability})
    return aeb_fitness_sr


'''
Checkpoint and early termination analysis
'''


def get_reward_mas(agent, name='eval_reward_ma'):
    '''Return array of the named reward_ma for all of an agent's bodies.'''
    bodies = getattr(agent, 'nanflat_body_a', [agent.body])
    return np.array([getattr(body, name) for body in bodies], dtype=np.float16)


def get_std_epi_rewards(agent):
    '''Return array of std_epi_reward for each of the environments.'''
    bodies = getattr(agent, 'nanflat_body_a', [agent.body])
    return np.array([ps.get(FITNESS_STD, f'{body.env.name}.std_epi_reward') for body in bodies], dtype=np.float16)


def new_best(agent):
    '''Check if algorithm is now the new best result, then update the new best'''
    best_reward_mas = get_reward_mas(agent, 'best_reward_ma')
    eval_reward_mas = get_reward_mas(agent, 'eval_reward_ma')
    best = (eval_reward_mas >= best_reward_mas).all()
    if best:
        bodies = getattr(agent, 'nanflat_body_a', [agent.body])
        for body in bodies:
            body.best_reward_ma = body.eval_reward_ma
    return best


def all_solved(agent):
    '''Check if envs have all been solved using std from slm_lab/spec/_fitness_std.json'''
    eval_reward_mas = get_reward_mas(agent, 'eval_reward_ma')
    std_epi_rewards = get_std_epi_rewards(agent)
    solved = (
        not np.isnan(std_epi_rewards).any() and
        (eval_reward_mas >= std_epi_rewards).all()
    )
    return solved


def is_unfit(fitness_df, session):
    '''Check if a fitness_df is unfit. Used to determine of trial should stop running more sessions'''
    if FITNESS_STD.get(session.spec['env'][0]['name']) is None:
        return False  # fitness not known
    mean_fitness_df = calc_mean_fitness(fitness_df)
    return mean_fitness_df['strength'].iloc[0] <= NOISE_WINDOW


'''
Analysis interface methods
'''


def save_spec(spec, unit='experiment'):
    '''Save spec to proper path. Called at Experiment or Trial init.'''
    prepath = util.get_prepath(spec, unit)
    util.write(spec, f'{prepath}_spec.json')


def calc_mean_fitness(fitness_df):
    '''Method to calculated mean over all bodies for a fitness_df'''
    return fitness_df.mean(axis=1, level=3)


def get_session_data(session, body_df_kind='eval', tmp_space_session_sub=False):
    '''
    Gather data from session from all the bodies
    Depending on body_df_kind, will use eval_df or train_df
    '''
    session_data = {}
    for aeb, body in util.ndenumerate_nonan(session.aeb_space.body_space.data):
        aeb_df = body.eval_df if body_df_kind == 'eval' else body.train_df
        # TODO tmp substitution since SpaceSession does not have run_eval yet
        if tmp_space_session_sub:
            aeb_df = body.train_df
        session_data[aeb] = aeb_df.copy()
    return session_data


def calc_session_fitness_df(session, session_data):
    '''Calculate the session fitness df'''
    session_fitness_data = {}
    for aeb in session_data:
        aeb_df = session_data[aeb]
        aeb_df = calc_epi_reward_ma(aeb_df, session.spec['meta']['ckpt'])
        util.downcast_float32(aeb_df)
        body = session.aeb_space.body_space.data[aeb]
        aeb_fitness_sr = calc_aeb_fitness_sr(aeb_df, body.env.name)
        aeb_fitness_df = pd.DataFrame([aeb_fitness_sr], index=[session.index])
        aeb_fitness_df = aeb_fitness_df.reindex(FITNESS_COLS[:3], axis=1)
        session_fitness_data[aeb] = aeb_fitness_df
    # form multi_index df, then take mean across all bodies
    session_fitness_df = pd.concat(session_fitness_data, axis=1)
    mean_fitness_df = calc_mean_fitness(session_fitness_df)
    session_fitness = calc_fitness(mean_fitness_df)
    return session_fitness_df


def calc_trial_fitness_df(trial):
    '''
    Calculate the trial fitness df by aggregating from the collected session_data_dict (session_fitness_df's).
    Adds a consistency dimension to fitness vector.
    '''
    trial_fitness_data = {}
    try:
        all_session_fitness_df = pd.concat(list(trial.session_data_dict.values()))
    except ValueError as e:
        logger.exception('Sessions failed, no data to analyze. Check stack trace above')
    for aeb in util.get_df_aeb_list(all_session_fitness_df):
        aeb_fitness_df = all_session_fitness_df.loc[:, aeb]
        aeb_fitness_sr = aeb_fitness_df.mean()
        consistency = calc_consistency(aeb_fitness_df)
        aeb_fitness_sr = aeb_fitness_sr.append(pd.Series({'consistency': consistency}))
        aeb_fitness_df = pd.DataFrame([aeb_fitness_sr], index=[trial.index])
        aeb_fitness_df = aeb_fitness_df.reindex(FITNESS_COLS, axis=1)
        trial_fitness_data[aeb] = aeb_fitness_df
    # form multi_index df, then take mean across all bodies
    trial_fitness_df = pd.concat(trial_fitness_data, axis=1)
    mean_fitness_df = calc_mean_fitness(trial_fitness_df)
    trial_fitness_df = mean_fitness_df
    trial_fitness = calc_fitness(mean_fitness_df)
    return trial_fitness_df


def plot_session(session_spec, session_data):
    '''Plot the session graph, 2 panes: reward, loss & explore_var. Each aeb_df gets its own color'''
    max_tick_unit = ps.get(session_spec, 'meta.max_tick_unit')
    aeb_count = len(session_data)
    palette = viz.get_palette(aeb_count)
    fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True, print_grid=False)
    for idx, (a, e, b) in enumerate(session_data):
        aeb_str = f'{a}{e}{b}'
        aeb_df = session_data[(a, e, b)]
        aeb_df.fillna(0, inplace=True)  # for saving plot, cant have nan
        fig_1 = viz.plot_line(aeb_df, 'reward_ma', max_tick_unit, legend_name=aeb_str, draw=False, trace_kwargs={'legendgroup': aeb_str, 'line': {'color': palette[idx]}})
        fig.append_trace(fig_1.data[0], 1, 1)

        fig_2 = viz.plot_line(aeb_df, ['loss'], max_tick_unit, y2_col=['explore_var'], trace_kwargs={'legendgroup': aeb_str, 'showlegend': False, 'line': {'color': palette[idx]}}, draw=False)
        fig.append_trace(fig_2.data[0], 2, 1)
        fig.append_trace(fig_2.data[1], 3, 1)

    fig.layout['xaxis1'].update(title=max_tick_unit, zerolinewidth=1)
    fig.layout['yaxis1'].update(fig_1.layout['yaxis'])
    fig.layout['yaxis1'].update(domain=[0.55, 1])
    fig.layout['yaxis2'].update(fig_2.layout['yaxis'])
    fig.layout['yaxis2'].update(showgrid=False, domain=[0, 0.45])
    fig.layout['yaxis3'].update(fig_2.layout['yaxis2'])
    fig.layout['yaxis3'].update(overlaying='y2', anchor='x2')
    fig.layout.update(ps.pick(fig_1.layout, ['legend']))
    fig.layout.update(title=f'session graph: {session_spec["name"]} t{session_spec["meta"]["trial"]} s{session_spec["meta"]["session"]}', width=500, height=600)
    viz.plot(fig)
    return fig


def gather_aeb_rewards_df(aeb, session_datas, max_tick_unit):
    '''Gather rewards from each session for a body into a df'''
    aeb_session_rewards = {}
    for s, session_data in session_datas.items():
        aeb_df = session_data[aeb]
        aeb_reward_sr = aeb_df['reward_ma']
        aeb_reward_sr.index = aeb_df[max_tick_unit]
        # guard for duplicate eval result
        aeb_reward_sr = aeb_reward_sr[~aeb_reward_sr.index.duplicated()]
        if util.in_eval_lab_modes():
            # guard for eval appending possibly not ordered
            aeb_reward_sr.sort_index(inplace=True)
        aeb_session_rewards[s] = aeb_reward_sr
    aeb_rewards_df = pd.DataFrame(aeb_session_rewards)
    return aeb_rewards_df


def build_aeb_reward_fig(aeb_rewards_df, aeb_str, color, max_tick_unit):
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
        fillcolor=viz.lower_opacity(color, 0.2),
        line=dict(color='rgba(0, 0, 0, 0)'),
        showlegend=False,
        legendgroup=aeb_str,
    )
    df = pd.DataFrame({max_tick_unit: x, 'mean_reward': mean_sr})
    fig = viz.plot_line(
        df, ['mean_reward'], [max_tick_unit], legend_name=aeb_str, draw=False, trace_kwargs={'legendgroup': aeb_str, 'line': {'color': color}}
    )
    fig.add_traces([envelope_trace])
    return fig


def calc_trial_df(trial_spec):
    '''Calculate trial_df as mean of all session_df'''
    from slm_lab.experiment import retro_analysis
    prepath = util.get_prepath(trial_spec)
    predir, _, _, _, _, _ = util.prepath_split(prepath)
    session_datas = retro_analysis.session_datas_from_file(predir, trial_spec)
    aeb_transpose = {aeb: [] for aeb in session_datas[list(session_datas.keys())[0]]}
    max_tick_unit = ps.get(trial_spec, 'meta.max_tick_unit')
    for s, session_data in session_datas.items():
        for aeb, aeb_df in session_data.items():
            aeb_transpose[aeb].append(aeb_df.sort_values(by=[max_tick_unit]).set_index(max_tick_unit, drop=False))

    trial_data = {}
    for aeb, df_list in aeb_transpose.items():
        trial_data[aeb] = pd.concat(df_list).groupby(level=0).mean().reset_index(drop=True)

    trial_df = pd.concat(trial_data, axis=1)
    return trial_df


def plot_trial(trial_spec):
    '''Plot the trial graph, 1 pane: mean and error envelope of reward graphs from all sessions. Each aeb_df gets its own color'''
    from slm_lab.experiment import retro_analysis
    prepath = util.get_prepath(trial_spec)
    predir, _, _, _, _, _ = util.prepath_split(prepath)
    session_datas = retro_analysis.session_datas_from_file(predir, trial_spec)
    rand_session_data = session_datas[list(session_datas.keys())[0]]
    max_tick_unit = ps.get(trial_spec, 'meta.max_tick_unit')
    aeb_count = len(rand_session_data)
    palette = viz.get_palette(aeb_count)
    fig = None
    for idx, (a, e, b) in enumerate(rand_session_data):
        aeb = (a, e, b)
        aeb_str = f'{a}{e}{b}'
        color = palette[idx]
        aeb_rewards_df = gather_aeb_rewards_df(aeb, session_datas, max_tick_unit)
        aeb_fig = build_aeb_reward_fig(aeb_rewards_df, aeb_str, color, max_tick_unit)
        if fig is None:
            fig = aeb_fig
        else:
            fig.add_traces(aeb_fig.data)
    fig.layout.update(title=f'trial graph: {trial_spec["name"]} t{trial_spec["meta"]["trial"]}, {len(session_datas)} sessions', width=500, height=600)
    viz.plot(fig)
    return fig


def plot_experiment(experiment_spec, experiment_df):
    '''
    Plot the variable specs vs fitness vector of an experiment, where each point is a trial.
    ref colors: https://plot.ly/python/heatmaps-contours-and-2dhistograms-tutorial/#plotlys-predefined-color-scales
    '''
    y_cols = ['fitness'] + FITNESS_COLS
    x_cols = ps.difference(experiment_df.columns.tolist(), y_cols)

    fig = viz.tools.make_subplots(rows=len(y_cols), cols=len(x_cols), shared_xaxes=True, shared_yaxes=True, print_grid=False)
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
                    'colorscale': 'YlGnBu', 'reversescale': True
                },
            )
            fig.append_trace(trace, row_idx + 1, col_idx + 1)
            fig.layout[f'xaxis{col_idx+1}'].update(title='<br>'.join(ps.chunk(x, 20)), zerolinewidth=1, categoryarray=sorted(guard_cat_x.unique()))
        fig.layout[f'yaxis{row_idx+1}'].update(title=y, rangemode='tozero')
    fig.layout.update(title=f'experiment graph: {experiment_spec["name"]}', width=max(600, len(x_cols) * 300), height=700)
    viz.plot(fig)
    return fig


def save_session_df(session_data, filepath, spec):
    '''Save session_df, and if is in eval mode, modify it and save with append'''
    if util.in_eval_lab_modes():
        ckpt = util.find_ckpt(spec['meta']['eval_model_prepath'])
        epi = int(re.search('epi(\d+)', ckpt)[1])
        totalt = int(re.search('totalt(\d+)', ckpt)[1])
        session_df = pd.concat(session_data, axis=1)
        mean_sr = session_df.mean()
        mean_sr.name = totalt  # set index to prevent all being the same
        eval_session_df = pd.DataFrame(data=[mean_sr])
        # set sr name too, to total_t
        for aeb in util.get_df_aeb_list(eval_session_df):
            eval_session_df.loc[:, aeb + ('epi',)] = epi
            eval_session_df.loc[:, aeb + ('total_t',)] = totalt
        # if eval, save with append mode
        header = not os.path.exists(filepath)
        with open(filepath, 'a') as f:
            eval_session_df.to_csv(f, header=header)
    else:
        session_df = pd.concat(session_data, axis=1)
        util.write(session_df, filepath)


def save_session_data(spec, session_data, session_fitness_df, session_fig, body_df_kind='eval'):
    '''
    Save the session data: session_df, session_fitness_df, session_graph.
    session_data is saved as session_df; multi-indexed with (a,e,b), 3 extra levels
    to read, use:
    session_df = util.read(filepath, header=[0, 1, 2, 3], index_col=0)
    session_data = util.session_df_to_data(session_df)
    '''
    prepath = util.get_prepath(spec, unit='session')
    prefix = 'train' if body_df_kind == 'train' else ''
    if 'retro_analyze' not in os.environ['PREPATH']:
        save_session_df(session_data, f'{prepath}_{prefix}session_df.csv', spec)
    util.write(session_fitness_df, f'{prepath}_{prefix}session_fitness_df.csv')
    viz.save_image(session_fig, f'{prepath}_{prefix}session_graph.png')
    logger.debug(f'Saved {body_df_kind} session data and graphs to {prepath}*')


def save_trial_data(spec, trial_df, trial_fitness_df, trial_fig, zip=True):
    '''Save the trial data: spec, trial_fitness_df.'''
    prepath = util.get_prepath(spec, unit='trial')
    util.write(trial_df, f'{prepath}_trial_df.csv')
    util.write(trial_fitness_df, f'{prepath}_trial_fitness_df.csv')
    viz.save_image(trial_fig, f'{prepath}_trial_graph.png')
    logger.debug(f'Saved trial data and graphs to {prepath}*')
    if util.get_lab_mode() == 'train' and zip:
        predir, _, _, _, _, _ = util.prepath_split(prepath)
        shutil.make_archive(predir, 'zip', predir)
        logger.info(f'All trial data zipped to {predir}.zip')


def save_experiment_data(spec, experiment_df, experiment_fig):
    '''Save the experiment data: best_spec, experiment_df, experiment_graph.'''
    prepath = util.get_prepath(spec, unit='experiment')
    util.write(experiment_df, f'{prepath}_experiment_df.csv')
    viz.save_image(experiment_fig, f'{prepath}_experiment_graph.png')
    logger.debug(f'Saved experiment data to {prepath}')
    # zip for ease of upload
    predir, _, _, _, _, _ = util.prepath_split(prepath)
    shutil.make_archive(predir, 'zip', predir)
    logger.info(f'All experiment data zipped to {predir}.zip')


def _analyze_session(session, session_data, body_df_kind='eval'):
    '''Helper method for analyze_session to run using eval_df and train_df'''
    session_fitness_df = calc_session_fitness_df(session, session_data)
    session_fig = plot_session(session.spec, session_data)
    save_session_data(session.spec, session_data, session_fitness_df, session_fig, body_df_kind)
    return session_fitness_df


def analyze_session(session, eager_analyze_trial=False, tmp_space_session_sub=False):
    '''
    Gather session data, plot, and return fitness df for high level agg.
    @returns {DataFrame} session_fitness_df Single-row df of session fitness vector (avg over aeb), indexed with session index.
    '''
    session_data = get_session_data(session, body_df_kind='train')
    session_fitness_df = _analyze_session(session, session_data, body_df_kind='train')
    session_data = get_session_data(session, body_df_kind='eval', tmp_space_session_sub=tmp_space_session_sub)
    session_fitness_df = _analyze_session(session, session_data, body_df_kind='eval')
    if eager_analyze_trial:
        # for live trial graph, analyze trial after analyzing session, this only takes a second
        from slm_lab.experiment import retro_analysis
        prepath = util.get_prepath(session.spec, unit='session')
        # use new ones to prevent side effects
        spec = util.prepath_to_spec(prepath)
        predir, _, _, _, _, _ = util.prepath_split(prepath)
        retro_analysis.analyze_eval_trial(spec, predir)
    return session_fitness_df


def analyze_trial(trial, zip=True):
    '''
    Gather trial data, plot, and return trial df for high level agg.
    @returns {DataFrame} trial_fitness_df Single-row df of trial fitness vector (avg over aeb, sessions), indexed with trial index.
    '''
    trial_df = calc_trial_df(trial.spec)
    trial_fitness_df = calc_trial_fitness_df(trial)
    trial_fig = plot_trial(trial.spec)
    save_trial_data(trial.spec, trial_df, trial_fitness_df, trial_fig, zip)
    return trial_fitness_df


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
    config_cols = sorted(ps.difference(experiment_df.columns.tolist(), cols))
    sorted_cols = config_cols + cols
    experiment_df = experiment_df.reindex(sorted_cols, axis=1)
    experiment_df.sort_values(by=['fitness'], ascending=False, inplace=True)
    logger.info(f'Experiment data:\n{experiment_df}')
    experiment_fig = plot_experiment(experiment.spec, experiment_df)
    save_experiment_data(experiment.spec, experiment_df, experiment_fig)
    return experiment_df
