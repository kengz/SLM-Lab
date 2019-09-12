# Script to generate latex and markdown graphs and tables
from glob import glob
from slm_lab.lib import logger, util, viz
import numpy as np
import pydash as ps


# declare file patterns
trial_metrics_scalar_path = '*trial_metrics_scalar.json'
trial_metrics_path = '*t0_trial_metrics.pkl'


def get_trial_metrics_scalar(algo, env, data_folder):
    try:
        filepaths = glob(f'{data_folder}/{algo}*{env}*/{trial_metrics_scalar_path}')
        assert len(filepaths) == 1, f'{algo}, {env}, {filepaths}'
        filepath = filepaths[0]
        return util.read(filepath)
    except Exception as e:
        # blank fill
        return {'final_return_ma': ''}


def get_latex_row(algos, env, data_folder):
    '''
    Get an environment's latex row where each column cell is an algorithm's reward.
    Max value in a row is formatted with textbf
    '''
    env_ret_ma_list = [get_trial_metrics_scalar(algo, env, data_folder)['final_return_ma'] for algo in algos]
    try:
        max_val = ps.max_([k for k in env_ret_ma_list if isinstance(k, (int, float))])
    except Exception as e:
        print(env, env_ret_ma_list)
        raise
    ret_ma_str_list = []
    for ret_ma in env_ret_ma_list:
        if isinstance(ret_ma, str):
            ret_ma_str = str(ret_ma)
        else:
            if abs(ret_ma) < 100:
                ret_ma_str = str(round(ret_ma, 2))
            else:
                ret_ma_str = str(round(ret_ma))
        if ret_ma and ret_ma == max_val:
            ret_ma_str = f'\\textbf{{{ret_ma_str}}}'
        ret_ma_str_list.append(ret_ma_str)
    env = env.split('-')[0]
    latex_row = f'& {env} & {" & ".join(ret_ma_str_list)} \\\\'
    return latex_row


def get_latex_body(algos, envs, data_folder):
    '''Get the benchmark table latex body (without header)'''
    latex_rows = [get_latex_row(algos, env, data_folder) for env in envs]
    latex_body = '\n'.join(latex_rows)
    return latex_body


def get_latex_im_body(envs):
    latex_ims = []
    for env in envs:
        latex_im = f'\subfloat{{\includegraphics[width=1.22in]{{images/{env}_multi_trial_graph_mean_returns_ma_vs_frames.png}}}}'
        latex_ims.append(latex_im)

    im_matrix = ps.chunk(latex_ims, 4)
    latex_im_body = '\\\\\n'.join([' & \n'.join(row) for row in im_matrix])
    return latex_im_body


def get_trial_metrics_path(algo, env, data_folder):
    filepaths = glob(f'{data_folder}/{algo}*{env}*/info/{trial_metrics_path}')
    assert len(filepaths) == 1
    return filepaths[0]


def plot_env(algos, env, data_folder, legend_list=None):
    trial_metrics_path_list = [get_trial_metrics_path(algo, env, data_folder) for algo in algos]
    if env == 'lunar':
        env = 'LunarLander'
    title = env
    if legend_list is None:
        graph_prepath = f'{data_folder}/{env}'
    else:
        graph_prepath = f'{data_folder}/{env}-legend'
    viz.plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath, ma=True, name_time_pairs=[('mean_returns', 'frames')])


def plot_envs(algos, envs, data_folder, legend_list):
    for idx, env in enumerate(envs):
        plot_env(algos, env, data_folder)
        if idx == len(envs) - 1:
            # plot extra to crop legend out
            plot_env(algos, env, data_folder, legend_list)


# Continuous
# Roboschool + Unity

algos = [
    'a2c_gae',
    'a2c_nstep',
    'ppo',
    'sac',
]
legend_list = [
    'A2C (GAE)',
    'A2C (n-step)',
    'PPO',
    'SAC',
]

envs = [
    'RoboschoolAnt-v1',
    'RoboschoolAtlasForwardWalk-v1',
    'RoboschoolHalfCheetah-v1',
    'RoboschoolHopper-v1',
    'RoboschoolInvertedDoublePendulum-v1',
    'RoboschoolInvertedPendulum-v1',
    'RoboschoolReacher-v1',
    'RoboschoolWalker2d-v1',
    'RoboschoolHumanoid-v1',
    'RoboschoolHumanoidFlagrun-v1',
    'RoboschoolHumanoidFlagrunHarder-v1',
    'Unity3DBall-v0',
    'Unity3DBallHard-v0',
    'UnityCrawlerDynamic-v0',
    'UnityCrawlerStatic-v0',
    'UnityReacher-v0',
    'UnityWalker-v0',
]

data_folder = util.smart_path('../Desktop/benchmark/cont')
latex_body = get_latex_body(algos, envs, data_folder)
print(latex_body)
plot_envs(algos, envs, data_folder, legend_list)


# Discrete
# LunarLander + Small Atari + Unity

algos = [
    'dqn',
    'ddqn_per',
    'a2c_gae',
    'a2c_nstep',
    'ppo',
    'sac',
]
legend_list = [
    'DQN',
    'DDQN+PER',
    'A2C (GAE)',
    'A2C (n-step)',
    'PPO',
    'SAC',
]

envs = [
    'lunar',
    'BeamRider',
    'Breakout',
    'MsPacman',
    'Pong',
    'Seaquest',
    'SpaceInvaders',
    'Qbert',
    'UnityHallway',
    'UnityPushBlock',
    'UnityPyramids',
]

data_folder = util.smart_path('../Desktop/benchmark/discrete')
latex_body = get_latex_body(algos, envs, data_folder)
print(latex_body)
plot_envs(algos, envs, data_folder, legend_list)


# Atari full

algos = [
    'dqn_atari',
    'ddqn_per',
    'a2c_gae',
    'a2c_nstep',
    'ppo',
]
legend_list = [
    'DQN',
    'DDQN+PER',
    'A2C (GAE)',
    'A2C (n-step)',
    'PPO',
]

envs = [
    "Adventure", "AirRaid", "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis", "BankHeist", "BattleZone", "BeamRider", "Berzerk", "Bowling", "Boxing", "Breakout", "Carnival", "Centipede", "ChopperCommand", "CrazyClimber", "Defender", "DemonAttack", "DoubleDunk", "ElevatorAction", "FishingDerby", "Freeway", "Frostbite", "Gopher", "Gravitar", "Hero", "IceHockey", "Jamesbond", "JourneyEscape", "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge", "MsPacman", "NameThisGame", "Phoenix", "Pitfall", "Pong", "Pooyan", "PrivateEye", "Qbert", "Riverraid", "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris", "SpaceInvaders", "StarGunner", "Tennis", "TimePilot", "Tutankham", "UpNDown", "Venture", "VideoPinball", "WizardOfWor", "YarsRevenge", "Zaxxon"
]

data_folder = util.smart_path('../Desktop/benchmark/atari')
latex_body = get_latex_body(algos, envs, data_folder)
print(latex_body)
latex_im_body = get_latex_im_body(envs)
print(latex_im_body)
plot_envs(algos, envs, data_folder, legend_list)
