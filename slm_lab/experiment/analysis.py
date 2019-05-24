'''
The analysis module
Handles the analyses of the info and data space for experiment evaluation and design.
'''
from itertools import product
from slm_lab.agent import AGENT_DATA_NAMES
from slm_lab.env import ENV_DATA_NAMES
from slm_lab.lib import logger, math_util, util, viz
from slm_lab.spec import random_baseline, spec_util
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
NUM_EVAL = 4

logger = logger.get_logger(__name__)


# methods to generate returns (total rewards)

def gen_return(agent, env):
    '''Generate return for an agent and an env in eval mode'''
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward


def gen_avg_return(agent, env, num_eval=NUM_EVAL):
    '''Generate average return for agent and an env'''
    with util.ctx_lab_mode('eval'):  # enter eval context
        agent.algorithm.update()  # set explore_var etc. to end_val under ctx
        returns = [gen_return(agent, env) for i in range(num_eval)]
    # exit eval context, restore variables simply by updating
    agent.algorithm.update()
    return np.mean(returns)


# metrics calculation methods

def calc_strength(mean_returns, mean_rand_returns):
    '''
    Calculate strength for metric
    str &= \frac{1}{N} \sum_{i=0}^N \overline{R}_i - \overline{R}_{rand}
    @param Series:mean_returns A series of mean returns from each checkpoint
    @param float:mean_rand_rets The random baseline
    @returns float:str, Series:local_strs
    '''
    local_strs = mean_returns - mean_rand_returns
    str_ = local_strs.mean()
    return str_, local_strs


def calc_efficiency(local_strs, ts):
    '''
    Calculate efficiency for metric
    e &= \frac{\sum_{i=0}^N \frac{1}{t_i} str_i}{\sum_{i=0}^N \frac{1}{t_i}}
    @param Series:local_strs A series of local strengths
    @param Series:ts A series of times units (total_t or opt_steps)
    @returns float:eff, Series:local_effs
    '''
    eff = (local_strs / ts).sum() / local_strs.sum()
    local_effs = (local_strs / ts).cumsum() / local_strs.cumsum()
    return eff, local_effs


def calc_stability(local_strs):
    '''
    Calculate stability for metric
    sta &= 1 - \left| \frac{\sum_{i=0}^{N-1} \min(str_{i+1} - str_i, 0)}{\sum_{i=0}^{N-1} str_i} \right|
    @param Series:local_strs A series of local strengths
    @returns float:sta, Series:local_stas
    '''
    # shift to keep indices for division
    drops = local_strs.diff().shift(-1).iloc[:-1].clip(upper=0.0)
    denoms = local_strs.iloc[:-1]
    local_stas = 1 - (drops / denoms).abs()
    sum_drops = drops.sum()
    sum_denom = denoms.sum()
    sta = 1 - np.abs(sum_drops / sum_denom)
    return sta, local_stas


def calc_consistency(local_strs_list):
    '''
    Calculate consistency for metric
    con &= 1 - \frac{\sum_{i=0}^N 2 stdev_j(str_{i,j})}{\sum_{i=0}^N avg_j(str_{i,j})}
    @param Series:local_strs_list A list of multiple series of local strengths from different sessions
    @returns float:con, Series:local_cons
    '''
    mean_local_strs, std_local_strs = util.calc_srs_mean_std(local_strs_list)
    local_cons = 1 - 2 * std_local_strs / mean_local_strs
    con = 1 - 2 * std_local_strs.sum() / mean_local_strs.sum()
    return con, local_cons


def calc_session_metrics(session_df, env_name, prepath=None):
    '''
    Calculate the session metrics: strength, efficiency, stability
    @param DataFrame:session_df Dataframe containing reward, total_t, opt_step
    @param str:env_name Name of the environment to get its random baseline
    @param str:prepath Optional prepath to auto-save the output to
    @returns dict:metrics Consists of scalar metrics and series local metrics
    '''
    rand_bl = random_baseline.get_random_baseline(env_name)
    mean_rand_returns = rand_bl['mean']
    mean_returns = session_df['reward']
    frames = session_df['total_t']
    opt_steps = session_df['opt_step']

    str_, local_strs = calc_strength(mean_returns, mean_rand_returns)
    min_str, max_str = local_strs.min(), local_strs.max()
    sample_eff, local_sample_effs = calc_efficiency(local_strs, frames)
    train_eff, local_train_effs = calc_efficiency(local_strs, opt_steps)
    sta, local_stas = calc_stability(local_strs)

    # all the scalar session metrics
    scalar = {
        'strength': str_,
        'min_strength': min_str,
        'max_strength': max_str,
        'sample_efficiency': sample_eff,
        'training_efficiency': train_eff,
        'stability': sta,
    }
    # all the session local metrics
    local = {
        'strengths': local_strs,
        'sample_efficiencies': local_sample_effs,
        'training_efficiencies': local_train_effs,
        'stabilities': local_stas,
        'mean_returns': mean_returns,
        'frames': frames,
        'opt_steps': opt_steps,
    }
    metrics = {
        'scalar': scalar,
        'local': local,
    }

    # auto-save if prepath is given
    if prepath is not None:
        util.write(metrics, f'{prepath}_session_metrics.pkl')
        util.write(scalar, f'{prepath}_session_metrics_scalar.json')
    return metrics


def calc_trial_metrics(session_metrics_list, prepath=None):
    '''
    Calculate the trial metrics: mean(strength), mean(efficiency), mean(stability), consistency
    @param list:session_metrics_list The metrics collected from each session; format: {session_index: {'scalar': {...}, 'local': {...}}}
    @param str:prepath Optional prepath to auto-save the output to
    @returns dict:metrics Consists of scalar metrics and series local metrics
    '''
    # calculate mean of session metrics
    scalar_list = [sm['scalar'] for sm in session_metrics_list]
    mean_scalar = pd.DataFrame(scalar_list).mean().to_dict()

    local_strs_list = [sm['local']['strengths'] for sm in session_metrics_list]
    local_se_list = [sm['local']['sample_efficiencies'] for sm in session_metrics_list]
    local_te_list = [sm['local']['training_efficiencies'] for sm in session_metrics_list]
    local_sta_list = [sm['local']['stabilities'] for sm in session_metrics_list]
    mean_returns_list = [sm['local']['mean_returns'] for sm in session_metrics_list]
    frames = session_metrics_list[0]['local']['frames']
    opt_steps = session_metrics_list[0]['local']['opt_steps']
    # calculate consistency
    con, local_cons = calc_consistency(local_strs_list)

    # all the scalar trial metrics
    scalar = {
        'strength': mean_scalar['strength'],
        'min_strength': mean_scalar['min_strength'],
        'max_strength': mean_scalar['max_strength'],
        'sample_efficiency': mean_scalar['sample_efficiency'],
        'training_efficiency': mean_scalar['training_efficiency'],
        'stability': mean_scalar['stability'],
        'consistency': con,
    }
    # for plotting: gather all local series of sessions
    local = {
        'strengths': local_strs_list,
        'sample_efficiencies': local_se_list,
        'training_efficiencies': local_te_list,
        'stabilities': local_sta_list,
        'consistencies': local_cons,  # this is a list
        'mean_returns': mean_returns_list,
        'frames': frames,
        'opt_steps': opt_steps,
    }
    metrics = {
        'scalar': scalar,
        'local': local,
    }

    # auto-save if prepath is given
    if prepath is not None:
        util.write(metrics, f'{prepath}_trial_metrics.pkl')
        util.write(scalar, f'{prepath}_trial_metrics_scalar.json')
    return metrics


# plotting methods

def plot_session(session_spec, session_df, df_mode='eval'):
    '''Plot the session graph, 2 panes: reward, loss & explore_var.'''
    meta_spec = session_spec['meta']
    prepath = meta_spec['prepath']
    max_tick_unit = ps.get(session_spec, 'meta.max_tick_unit')
    # TODO iterate for vector rewards later
    color = viz.get_palette(1)[0]
    fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True, print_grid=False)
    session_df = session_df.fillna(0)  # for saving plot, cant have nan
    fig_1 = viz.plot_line(session_df, 'reward', max_tick_unit, draw=False, trace_kwargs={'line': {'color': color}})
    fig.add_trace(fig_1.data[0], 1, 1)

    fig_2 = viz.plot_line(session_df, ['loss'], max_tick_unit, y2_col=['explore_var'], trace_kwargs={'showlegend': False, 'line': {'color': color}}, draw=False)
    fig.add_trace(fig_2.data[0], 2, 1)
    fig.add_trace(fig_2.data[1], 3, 1)

    fig.layout['xaxis1'].update(title=max_tick_unit, zerolinewidth=1)
    fig.layout['yaxis1'].update(fig_1.layout['yaxis'])
    fig.layout['yaxis1'].update(domain=[0.55, 1])
    fig.layout['yaxis2'].update(fig_2.layout['yaxis'])
    fig.layout['yaxis2'].update(showgrid=False, domain=[0, 0.45])
    fig.layout['yaxis3'].update(fig_2.layout['yaxis2'])
    fig.layout['yaxis3'].update(overlaying='y2', anchor='x2')
    fig.layout.update(ps.pick(fig_1.layout, ['legend']))
    fig.layout.update(title=f'session graph: {session_spec["name"]} t{meta_spec["trial"]} s{meta_spec["session"]}', width=500, height=600)
    viz.plot(fig)
    viz.save_image(fig, f'{prepath}_{df_mode}_session_graph.png')
    return fig


def plot_trial(trial_spec, trial_metrics):
    '''
    Plot the trial graphs:
    - mean_returns, strengths, sample_efficiencies, training_efficiencies, stabilities (with error bar)
    - consistencies (no error bar)
    uses dual time axes: {frames, opt_steps}
    '''
    meta_spec = trial_spec['meta']
    prepath = meta_spec['prepath']
    title = f'{trial_spec["name"]} trial {meta_spec["trial"]}, {meta_spec["max_session"]} sessions'

    local_trial_metrics = trial_metrics['local']
    name_time_pairs = list(product(('mean_returns', 'strengths', 'stabilities', 'consistencies'), ('frames', 'opt_steps')))
    name_time_pairs += [
        ('sample_efficiencies', 'frames'),
        ('training_efficiencies', 'opt_steps'),
    ]
    for name, time in name_time_pairs:
        if name == 'consistencies':
            fig = viz.plot_sr(
                local_trial_metrics[name],
                local_trial_metrics[time],
                title, name, time)
        else:
            fig = viz.plot_mean_sr(
                local_trial_metrics[name],
                local_trial_metrics[time],
                title, name, time)
        viz.save_image(fig, f'{prepath}_trial_graph_{name}_vs_{time}.png')


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
            fig.add_trace(trace, row_idx + 1, col_idx + 1)
            fig.layout[f'xaxis{col_idx+1}'].update(title='<br>'.join(ps.chunk(x, 20)), zerolinewidth=1, categoryarray=sorted(guard_cat_x.unique()))
        fig.layout[f'yaxis{row_idx+1}'].update(title=y, rangemode='tozero')
    fig.layout.update(title=f'experiment graph: {experiment_spec["name"]}', width=max(600, len(x_cols) * 300), height=700)
    viz.plot(fig)
    return fig


def save_experiment_data(spec, experiment_df, experiment_fig):
    '''Save the experiment data: best_spec, experiment_df, experiment_graph.'''
    prepath = spec['meta']['prepath']
    util.write(experiment_df, f'{prepath}_experiment_df.csv')
    viz.save_image(experiment_fig, f'{prepath}_experiment_graph.png')
    logger.debug(f'Saved experiment data to {prepath}')
    # zip for ease of upload
    predir, _, _, _, _, _ = util.prepath_split(prepath)
    shutil.make_archive(predir, 'zip', predir)
    logger.info(f'All experiment data zipped to {predir}.zip')


# interface analyze methods

def _analyze_session(session, df_mode='eval'):
    '''Helper method for analyze_session to run using eval_df and train_df'''
    prepath = session.spec['meta']['prepath']
    body = session.agent.body
    session_df = getattr(body, f'{df_mode}_df').copy()
    if 'retro_analyze' not in os.environ['PREPATH']:
        util.write(session_df, f'{prepath}_{df_mode}_session_df.csv')

    # calculate metrics
    session_metrics = calc_session_metrics(session_df, body.env.name, prepath)
    if df_mode == 'eval':
        # add session scalar metrics to session
        session.spec['metrics'] = session_metrics['scalar']
        spec_util.save(session.spec, unit='session')

    # plot graph
    session_fig = plot_session(session.spec, session_df, df_mode)
    logger.debug(f'Saved {df_mode} session data and graphs to {prepath}*')
    return session_metrics


def analyze_session(session):
    '''Analyze session and save data, then return metrics'''
    _analyze_session(session, df_mode='train')
    session_metrics = _analyze_session(session, df_mode='eval')
    return session_metrics


def analyze_trial(trial, zip=True):
    '''Analyze trial and save data, then return metrics'''
    prepath = trial.spec['meta']['prepath']
    # calculate metrics
    trial_metrics = calc_trial_metrics(trial.session_metrics_list, prepath)
    # plot graphs
    trial_fig = plot_trial(trial.spec, trial_metrics)
    logger.debug(f'Saved trial data and graphs to {prepath}*')
    # zip files
    if util.get_lab_mode() == 'train' and zip:
        predir, _, _, _, _, _ = util.prepath_split(prepath)
        shutil.make_archive(predir, 'zip', predir)
        logger.info(f'All trial data zipped to {predir}.zip')
    return trial_metrics


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
