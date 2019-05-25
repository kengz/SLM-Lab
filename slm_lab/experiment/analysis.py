from slm_lab.lib import logger, util, viz
from slm_lab.spec import random_baseline
import numpy as np
import os
import pandas as pd
import pydash as ps
import shutil

MA_WINDOW = 100
NUM_EVAL = 4
METRICS_COLS = [
    'strength', 'max_strength', 'final_strength',
    'sample_efficiency', 'training_efficiency',
    'stability', 'consistency',
]

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


def get_reward_mas(agent, name='eval_reward_ma'):
    '''Return array of the named reward_ma for all of an agent's bodies.'''
    bodies = getattr(agent, 'nanflat_body_a', [agent.body])
    return np.array([getattr(body, name) for body in bodies], dtype=np.float16)


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
    max_str, final_str = local_strs.max(), local_strs.iloc[-1]
    sample_eff, local_sample_effs = calc_efficiency(local_strs, frames)
    train_eff, local_train_effs = calc_efficiency(local_strs, opt_steps)
    sta, local_stas = calc_stability(local_strs)

    # all the scalar session metrics
    scalar = {
        'strength': str_,
        'max_strength': max_str,
        'final_strength': final_str,
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
    if prepath is not None:  # auto-save if prepath is given
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
    if prepath is not None:  # auto-save if prepath is given
        util.write(metrics, f'{prepath}_trial_metrics.pkl')
        util.write(scalar, f'{prepath}_trial_metrics_scalar.json')
    return metrics


def calc_experiment_df(trial_data_dict, prepath=None):
    '''Collect all trial data (metrics and config) from trials into a dataframe'''
    experiment_df = pd.DataFrame(trial_data_dict).transpose()
    cols = METRICS_COLS
    config_cols = sorted(ps.difference(experiment_df.columns.tolist(), cols))
    sorted_cols = config_cols + cols
    experiment_df = experiment_df.reindex(sorted_cols, axis=1)
    experiment_df.sort_values(by=['strength'], ascending=False, inplace=True)
    if prepath is not None:
        util.write(experiment_df, f'{prepath}_experiment_df.csv')
    return experiment_df


# interface analyze methods

def _analyze_session(session, df_mode='eval'):
    '''Helper method for analyze_session to run using eval_df and train_df'''
    prepath = session.spec['meta']['prepath']
    body = session.agent.body
    session_df = getattr(body, f'{df_mode}_df').copy()
    assert len(session_df) > 1, f'Need more than 2 datapoints to calculate metrics'
    if 'retro_analyze' not in os.environ['PREPATH']:
        util.write(session_df, f'{prepath}_session_df_{df_mode}.csv')
    # calculate metrics
    session_metrics = calc_session_metrics(session_df, body.env.name, prepath)
    body.log_metrics(session_metrics['scalar'])
    # plot graph
    viz.plot_session(session.spec, session_metrics, session_df, df_mode)
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
    viz.plot_trial(trial.spec, trial_metrics)
    logger.debug(f'Saved trial data and graphs to {prepath}*')
    # zip files
    if util.get_lab_mode() == 'train' and zip:
        predir, _, _, _, _, _ = util.prepath_split(prepath)
        shutil.make_archive(predir, 'zip', predir)
        logger.info(f'All trial data zipped to {predir}.zip')
    return trial_metrics


def analyze_experiment(experiment):
    '''Analyze experiment and save data'''
    prepath = experiment.spec['meta']['prepath']
    # calculate experiment df
    experiment_df = calc_experiment_df(experiment.trial_data_dict, prepath)
    # plot graph
    viz.plot_experiment(experiment.spec, experiment_df, METRICS_COLS)
    logger.debug(f'Saved experiment data to {prepath}')
    # zip files
    predir, _, _, _, _, _ = util.prepath_split(prepath)
    shutil.make_archive(predir, 'zip', predir)
    logger.info(f'All experiment data zipped to {predir}.zip')
    return experiment_df
