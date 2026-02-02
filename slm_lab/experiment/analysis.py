import warnings
from copy import deepcopy
import glob

import numpy as np
import pandas as pd
import pydash as ps
import torch

from slm_lab.lib import logger, util, viz
from slm_lab.spec import random_baseline, spec_util

METRICS_COLS = [
    'frame',
    'total_reward_ma',
    'strength', 'max_strength', 'final_strength',
    'sample_efficiency', 'training_efficiency',
    'stability', 'consistency',
]

logger = logger.get_logger(__name__)


# methods to generate returns (total rewards)

def gen_return(agent, env):
    '''Generate return for an agent and an env in eval mode. eval_env should be a vec env with NUM_EVAL instances'''
    vec_dones = False  # done check for single and vec env
    # swap ref to allow inference based on agent.env
    original_env = agent.env
    agent.env = env
    # start eval loop
    state, info = env.reset()
    while not np.all(vec_dones):
        action = agent.act(state)
        state, reward, term, trunc, info = env.step(action)
        done = np.logical_or(term, trunc)
        vec_dones = np.logical_or(vec_dones, done)  # wait till every vec slot done turns True
    agent.env = original_env  # restore swapped ref
    return np.mean(env.total_reward)


def gen_avg_return(agent, env):
    '''Generate average return for agent and an env'''
    with util.ctx_lab_mode('eval'):  # enter eval context
        agent.algorithm.update()  # set explore_var etc. to end_val under ctx
    with torch.no_grad():
        ret = gen_return(agent, env)
    # exit eval context, restore variables simply by updating
    agent.algorithm.update()
    return ret


# metrics calculation methods

def calc_strength(mean_returns, mean_rand_returns):
    r'''
    Calculate strength for metric
    str &= \frac{1}{N} \sum_{i=0}^N \overline{R}_i - \overline{R}_{rand}
    @param Series:mean_returns A series of mean returns from each checkpoint
    @param float:mean_rand_returns The random baseline
    @returns float:str, Series:local_strs
    '''
    local_strs = mean_returns - mean_rand_returns
    str_ = local_strs.mean()
    return str_, local_strs


def calc_efficiency(local_strs, ts):
    r'''
    Calculate efficiency for metric
    e &= \frac{\sum_{i=0}^N \frac{1}{t_i} str_i}{\sum_{i=0}^N \frac{1}{t_i}}
    @param Series:local_strs A series of local strengths
    @param Series:ts A series of times units (frame or opt_steps)
    @returns float:eff, Series:local_effs
    '''
    # drop inf from when first t is 0
    str_t_ratios = (local_strs / ts).replace([np.inf, -np.inf], np.nan).dropna()
    eff = str_t_ratios.sum() / local_strs.sum()
    local_effs = str_t_ratios.cumsum() / local_strs.cumsum()
    return eff, local_effs


def calc_stability(local_strs):
    r'''
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
    r'''
    Calculate consistency for metric
    con &= 1 - \frac{\sum_{i=0}^N 2 stdev_j(str_{i,j})}{\sum_{i=0}^N avg_j(str_{i,j})}
    @param Series:local_strs_list A list of multiple series of local strengths from different sessions
    @returns float:con, Series:local_cons
    '''
    mean_local_strs, std_local_strs = util.calc_srs_mean_std(local_strs_list)
    local_cons = 1 - 2 * std_local_strs / mean_local_strs
    con = 1 - 2 * std_local_strs.sum() / mean_local_strs.sum()
    return con, local_cons


def to_series(data):
    '''Convert list to Series if needed (for JSON deserialization compatibility)'''
    return pd.Series(data) if isinstance(data, list) else data


def calc_session_metrics(session_df, env_name, info_prepath=None, df_mode=None):
    '''
    Calculate the session metrics: strength, efficiency, stability
    @param DataFrame:session_df Dataframe containing reward, frame, opt_step
    @param str:env_name Name of the environment to get its random baseline
    @param str:info_prepath Optional info_prepath to auto-save the output to
    @param str:df_mode Optional df_mode to save with info_prepath
    @returns dict:metrics Consists of scalar metrics and series local metrics
    '''
    rand_bl = random_baseline.get_random_baseline(env_name)
    if rand_bl is None:
        mean_rand_returns = 0.0
        logger.warning('Random baseline unavailable for environment. Please generate separately.')
    else:
        mean_rand_returns = rand_bl['mean']
    mean_returns = session_df['total_reward']
    frames = session_df['frame']
    opt_steps = session_df['opt_step']

    # Protect against insufficient data points
    if len(mean_returns) == 0:
        logger.warning('Empty session data - using NaN metrics')
        total_reward_ma = np.nan
        str_, local_strs = np.nan, pd.Series(dtype=float)
        max_str, final_str = np.nan, np.nan
    else:
        # Use available data if less than PLOT_MA_WINDOW
        window_size = min(len(mean_returns), viz.PLOT_MA_WINDOW)
        # total_reward_ma: same calculation as real-time total_reward_ma, but computed post-hoc for final analysis
        total_reward_ma = mean_returns[-window_size:].mean()
        str_, local_strs = calc_strength(mean_returns, mean_rand_returns)
        max_str, final_str = local_strs.max(), local_strs.iloc[-1]
    with warnings.catch_warnings():  # mute np.nanmean warning
        warnings.filterwarnings('ignore')
        sample_eff, local_sample_effs = calc_efficiency(local_strs, frames)
        train_eff, local_train_effs = calc_efficiency(local_strs, opt_steps)
        sta, local_stas = calc_stability(local_strs)

    # all the scalar session metrics
    scalar = {
        'total_reward_ma': total_reward_ma,
        'strength': str_,
        'max_strength': max_str,
        'final_strength': final_str,
        'sample_efficiency': sample_eff,
        'training_efficiency': train_eff,
        'stability': sta,
    }
    # all the session local metrics
    local = {
        'mean_returns': mean_returns,
        'strengths': local_strs,
        'sample_efficiencies': local_sample_effs,
        'training_efficiencies': local_train_effs,
        'stabilities': local_stas,
        'frames': frames,
        'opt_steps': opt_steps,
    }
    metrics = {
        'scalar': scalar,
        'local': local,
    }
    if info_prepath is not None:  # auto-save if info_prepath is given
        util.write(metrics, f'{info_prepath}_session_metrics_{df_mode}.json')
        util.write(scalar, f'{info_prepath}_session_metrics_scalar_{df_mode}.json')
    return metrics


def calc_trial_metrics(session_metrics_list, info_prepath=None):
    '''
    Calculate the trial metrics: mean(strength), mean(efficiency), mean(stability), consistency
    @param list:session_metrics_list The metrics collected from each session; format: {session_index: {'scalar': {...}, 'local': {...}}}
    @param str:info_prepath Optional info_prepath to auto-save the output to
    @returns dict:metrics Consists of scalar metrics and series local metrics
    '''
    # calculate mean of session metrics
    scalar_list = [sm['scalar'] for sm in session_metrics_list]
    mean_scalar = pd.DataFrame(scalar_list).mean().to_dict()

    # Convert lists to Series (JSON deserialization artifact)
    mean_returns_list = [to_series(sm['local']['mean_returns']) for sm in session_metrics_list]
    local_strs_list = [to_series(sm['local']['strengths']) for sm in session_metrics_list]
    local_se_list = [to_series(sm['local']['sample_efficiencies']) for sm in session_metrics_list]
    local_te_list = [to_series(sm['local']['training_efficiencies']) for sm in session_metrics_list]
    local_sta_list = [to_series(sm['local']['stabilities']) for sm in session_metrics_list]
    frames = to_series(session_metrics_list[0]['local']['frames'])
    opt_steps = to_series(session_metrics_list[0]['local']['opt_steps'])
    # calculate consistency
    con, local_cons = calc_consistency(local_strs_list)

    # all the scalar trial metrics
    scalar = {
        'frame': frames.iloc[-1] if len(frames) > 0 else 0,
        'total_reward_ma': mean_scalar['total_reward_ma'],
        'strength': mean_scalar['strength'],
        'max_strength': mean_scalar['max_strength'],
        'final_strength': mean_scalar['final_strength'],
        'sample_efficiency': mean_scalar['sample_efficiency'],
        'training_efficiency': mean_scalar['training_efficiency'],
        'stability': mean_scalar['stability'],
        'consistency': con,
    }
    assert set(scalar.keys()) == set(METRICS_COLS)
    # for plotting: gather all local series of sessions
    local = {
        'mean_returns': mean_returns_list,
        'strengths': local_strs_list,
        'sample_efficiencies': local_se_list,
        'training_efficiencies': local_te_list,
        'stabilities': local_sta_list,
        'consistencies': local_cons,  # this is a list
        'frames': frames,
        'opt_steps': opt_steps,
    }
    metrics = {
        'scalar': scalar,
        'local': local,
    }
    if info_prepath is not None:  # auto-save if info_prepath is given
        util.write(metrics, f'{info_prepath}_trial_metrics.json')
        util.write(scalar, f'{info_prepath}_trial_metrics_scalar.json')
        # save important trial metrics in predir for easy access
        util.write(scalar, f'{info_prepath.replace("info/", "")}_trial_metrics_scalar.json')
    return metrics


def calc_experiment_df(trial_data_dict, info_prepath=None):
    '''Collect all trial data (metrics and config) from trials into a dataframe'''
    experiment_df = pd.DataFrame(trial_data_dict).transpose()
    cols = METRICS_COLS
    config_cols = sorted(ps.difference(experiment_df.columns.tolist(), cols))
    sorted_cols = config_cols + cols
    experiment_df = experiment_df.reindex(sorted_cols, axis=1)
    experiment_df.sort_values(by=['strength'], ascending=False, inplace=True)
    # insert trial index
    experiment_df.insert(0, 'trial', experiment_df.index.astype(int))
    if info_prepath is not None:
        util.write(experiment_df, f'{info_prepath}_experiment_df.csv')
        # save important experiment df in predir for easy access
        util.write(experiment_df, f'{info_prepath.replace("info/", "")}_experiment_df.csv')
    return experiment_df


# interface analyze methods

def analyze_session(session_spec, session_df, df_mode, plot=True):
    '''Analyze session and save data, then return metrics. Note there are 2 types of session_df: agent.mt.eval_df and agent.mt.train_df'''
    info_prepath = session_spec['meta']['info_prepath']
    session_df = session_df.copy()  # prevent modification
    assert len(session_df) > 2, 'Need more than 2 datapoint to calculate metrics'  # first datapoint at frame 0 is empty
    util.write(session_df, util.get_session_df_path(session_spec, df_mode))
    # calculate metrics
    session_metrics = calc_session_metrics(session_df, ps.get(session_spec, 'env.name'), info_prepath, df_mode)
    if plot:
        # plot graph
        viz.plot_session(session_spec, session_metrics, session_df, df_mode)
        viz.plot_session(session_spec, session_metrics, session_df, df_mode, ma=True)
    return session_metrics


def analyze_trial(trial_spec, session_metrics_list=None):
    '''Analyze trial and save data, then return metrics. If session_metrics_list not provided, load from saved files.'''
    # Guard: detect if session spec passed instead of trial spec (session >= 0)
    # Restore to trial_spec which has its own meta prepaths without session infix
    if trial_spec['meta']['session'] >= 0:
        trial_spec = deepcopy(trial_spec)
        trial_spec['meta']['trial'] -= 1
        spec_util.tick(trial_spec, 'trial')
    info_prepath = trial_spec['meta']['info_prepath']
    # Load session metrics if not provided
    if session_metrics_list is None:
        # Use smart_path to get absolute path for glob (fixes Ray Tune working directory issues)
        abs_info_prepath = util.smart_path(info_prepath)
        session_files = sorted(glob.glob(f'{abs_info_prepath}_s*_session_metrics_train.json'))
        session_metrics_list = [m for f in session_files if (m := ps.attempt(util.read, f)) and isinstance(m, dict)]
        if not session_metrics_list:
            return None
    # calculate metrics
    trial_metrics = calc_trial_metrics(session_metrics_list, info_prepath)
    # plot graphs
    viz.plot_trial(trial_spec, trial_metrics)
    viz.plot_trial(trial_spec, trial_metrics, ma=True)
    return trial_metrics


def analyze_experiment(spec, trial_data_dict):
    '''Analyze experiment and save data'''
    info_prepath = spec['meta']['info_prepath']
    util.write(trial_data_dict, f'{info_prepath}_trial_data_dict.json')
    # calculate experiment df
    experiment_df = calc_experiment_df(trial_data_dict, info_prepath)
    # plot graph
    viz.plot_experiment(spec, experiment_df, METRICS_COLS)
    viz.plot_experiment_trials(spec, experiment_df, METRICS_COLS)
    return experiment_df
