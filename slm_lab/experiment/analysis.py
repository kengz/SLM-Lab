from slm_lab.lib import logger, util, viz
from slm_lab.spec import random_baseline
import numpy as np
import os
import pandas as pd
import pydash as ps
import shutil
import torch


METRICS_COLS = [
    'final_return_ma',
    'strength', 'max_strength', 'final_strength',
    'sample_efficiency', 'training_efficiency',
    'stability', 'consistency',
]

logger = logger.get_logger(__name__)


# methods to generate returns (total rewards)

def gen_return(agent, env):
    '''Generate return for an agent and an env in eval mode. eval_env should be a vec env with NUM_EVAL instances'''
    # stats variables
    epi_start = True
    ckpt_total_reward = np.nan
    total_reward = 0
    vec_dones = False
    # swap ref to allow inference based on body.env
    main_env = agent.body.env
    agent.body.env = env
    # start eval loop
    state = env.reset()
    done = False
    while not np.all(vec_dones):
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        if hasattr(env.u_env, 'raw_reward'):  # use raw_reward if reward is preprocessed
            reward = env.u_env.raw_reward
        ckpt_total_reward, total_reward, epi_start = util.update_total_reward(ckpt_total_reward, total_reward, epi_start, reward, done)
        vec_dones = np.logical_or(vec_dones, done)  # wait till every vec slot done turns True
    # restore swapped ref
    agent.body.env = main_env
    return np.mean(total_reward)


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
    '''
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
    '''
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
    mean_rand_returns = rand_bl['mean']
    mean_returns = session_df['total_reward']
    frames = session_df['frame']
    opt_steps = session_df['opt_step']

    final_return_ma = mean_returns[-viz.PLOT_MA_WINDOW:].mean()
    str_, local_strs = calc_strength(mean_returns, mean_rand_returns)
    max_str, final_str = local_strs.max(), local_strs.iloc[-1]
    sample_eff, local_sample_effs = calc_efficiency(local_strs, frames)
    train_eff, local_train_effs = calc_efficiency(local_strs, opt_steps)
    sta, local_stas = calc_stability(local_strs)

    # all the scalar session metrics
    scalar = {
        'final_return_ma': final_return_ma,
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
        util.write(metrics, f'{info_prepath}_session_metrics_{df_mode}.pkl')
        util.write(scalar, f'{info_prepath}_session_metrics_scalar_{df_mode}.json')
        # save important metrics in info_prepath directly
        util.write(scalar, f'{info_prepath.replace("info/", "")}_session_metrics_scalar_{df_mode}.json')
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

    mean_returns_list = [sm['local']['mean_returns'] for sm in session_metrics_list]
    local_strs_list = [sm['local']['strengths'] for sm in session_metrics_list]
    local_se_list = [sm['local']['sample_efficiencies'] for sm in session_metrics_list]
    local_te_list = [sm['local']['training_efficiencies'] for sm in session_metrics_list]
    local_sta_list = [sm['local']['stabilities'] for sm in session_metrics_list]
    frames = session_metrics_list[0]['local']['frames']
    opt_steps = session_metrics_list[0]['local']['opt_steps']
    # calculate consistency
    con, local_cons = calc_consistency(local_strs_list)

    # all the scalar trial metrics
    scalar = {
        'final_return_ma': mean_scalar['final_return_ma'],
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
        util.write(metrics, f'{info_prepath}_trial_metrics.pkl')
        util.write(scalar, f'{info_prepath}_trial_metrics_scalar.json')
        # save important metrics in info_prepath directly
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
    experiment_df.insert(0, 'trial', experiment_df.index.astype(np.int))
    if info_prepath is not None:
        util.write(experiment_df, f'{info_prepath}_experiment_df.csv')
        # save important metrics in info_prepath directly
        util.write(experiment_df, f'{info_prepath.replace("info/", "")}_experiment_df.csv')
    return experiment_df


# interface analyze methods

def analyze_session(session_spec, session_df, df_mode, plot=True):
    '''Analyze session and save data, then return metrics. Note there are 2 types of session_df: body.eval_df and body.train_df'''
    info_prepath = session_spec['meta']['info_prepath']
    session_df = session_df.copy()
    assert len(session_df) > 1, f'Need more than 1 datapoint to calculate metrics'
    util.write(session_df, f'{info_prepath}_session_df_{df_mode}.csv')
    # calculate metrics
    session_metrics = calc_session_metrics(session_df, ps.get(session_spec, 'env.0.name'), info_prepath, df_mode)
    if plot:
        # plot graph
        viz.plot_session(session_spec, session_metrics, session_df, df_mode)
        viz.plot_session(session_spec, session_metrics, session_df, df_mode, ma=True)
    return session_metrics


def analyze_trial(trial_spec, session_metrics_list):
    '''Analyze trial and save data, then return metrics'''
    info_prepath = trial_spec['meta']['info_prepath']
    # calculate metrics
    trial_metrics = calc_trial_metrics(session_metrics_list, info_prepath)
    # plot graphs
    viz.plot_trial(trial_spec, trial_metrics)
    viz.plot_trial(trial_spec, trial_metrics, ma=True)
    # zip files
    if util.get_lab_mode() == 'train':
        predir, _, _, _, _, _ = util.prepath_split(info_prepath)
        shutil.make_archive(predir, 'zip', predir)
        logger.info(f'All trial data zipped to {predir}.zip')
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
    # zip files
    predir, _, _, _, _, _ = util.prepath_split(info_prepath)
    shutil.make_archive(predir, 'zip', predir)
    logger.info(f'All experiment data zipped to {predir}.zip')
    return experiment_df
