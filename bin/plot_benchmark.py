# Script to generate latex and markdown graphs and tables
from glob import glob
from slm_lab.lib import logger, util, viz
import numpy as np
import pydash as ps


# declare file patterns
trial_metrics_scalar_path = '*trial_metrics_scalar.json'
trial_metrics_path = '*t0_trial_metrics.pkl'


def get_trial_metrics_scalar(algo, env, data_folder):
    filepaths = glob(f'{data_folder}/*{algo}*{env}*/{trial_metrics_scalar_path}')
    assert len(filepaths) == 1
    filepath = filepaths[0]
    return util.read(filepath)


def get_trial_metrics_path(algo, env, data_folder):
    filepaths = glob(f'{data_folder}/*{algo}*{env}*/info/{trial_metrics_path}')
    assert len(filepaths) == 1
    return filepaths[0]


def get_latex_row(algos, env, data_folder):
    '''
    Get an environment's latex row where each column cell is an algorithm's reward.
    Max value in a row is formatted with textbf
    '''
    env_ret_ma_list = [get_trial_metrics_scalar(algo, env, data_folder)['final_return_ma'] for algo in algos]
    max_val = ps.max_(env_ret_ma_list)
    ret_ma_str_list = []
    for ret_ma in env_ret_ma_list:
        ret_ma_str = str(round(ret_ma, 2))
        if ret_ma == max_val:
            ret_ma_str = f'\\textbf{{{ret_ma_str}}}'
        ret_ma_str_list.append(ret_ma_str)
    latex_row = f'& {env} & {" & ".join(ret_ma_str_list)} \\\\'
    return latex_row


def get_latex_body(algos, envs, data_folder):
    '''Get the benchmark table latex body (without header)'''
    latex_rows = [get_latex_row(algos, env, data_folder) for env in envs]
    latex_body = '\n'.join(latex_rows)
    return latex_body


algos = [
    'a2c_nstep',
    'a2c_gae',
    'ppo',
    'sac',
]

envs = [
    'RoboschoolAnt-v1',
    'RoboschoolAtlasForwardWalk-v1',
    'RoboschoolHalfCheetah-v1',
    'RoboschoolHopper-v1',
    'RoboschoolInvertedDoublePendulum-v1',
    'RoboschoolInvertedPendulum-v1',
    'RoboschoolReacher-v1',
    'RoboschoolWalker2d-v1'
]

data_folder = util.smart_path('../Desktop/benchmark')


latex_body = get_latex_body(algos, envs, data_folder)
print(latex_body)
