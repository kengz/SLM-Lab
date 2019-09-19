# Script to generate latex and markdown graphs and tables
# NOTE: add this to viz.create_layout method:
# font=dict(size=18),
from glob import glob
from slm_lab.lib import logger, util, viz
import numpy as np
import pydash as ps


# declare file patterns
trial_metrics_scalar_path = '*trial_metrics_scalar.json'
trial_metrics_path = '*t0_trial_metrics.pkl'
env_name_map = {
    'lunar': 'LunarLander',
    'ong': 'Pong',
    'bert': 'Qbert',
    'humanoid': 'RoboschoolHumanoid',
    'humanoidflagrun': 'RoboschoolHumanoidFlagrun',
    'humanoidflagrunharder': 'RoboschoolHumanoidFlagrunHarder',
}
master_legend_list = [
    'DQN',
    'DDQN+PER',
    'A2C (GAE)',
    'A2C (n-step)',
    'PPO',
    'SAC',
]
master_palette_dict = dict(zip(master_legend_list, viz.get_palette(len(master_legend_list))))
master_palette_dict['Async SAC'] = master_palette_dict['SAC']


def guard_env_name(env):
    env = env.strip('_').strip('-')
    if env in env_name_map:
        return env_name_map[env]
    else:
        return env


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
    env = guard_env_name(env)
    latex_row = f'{env} & {" & ".join(ret_ma_str_list)} \\\\'
    return latex_row


def get_latex_body(algos, envs, data_folder):
    '''Get the benchmark table latex body (without header)'''
    latex_rows = [get_latex_row(algos, env, data_folder) for env in envs]
    latex_body = '\n'.join(latex_rows)
    return latex_body


def get_latex_im_body(envs):
    latex_ims = []
    for env in envs:
        env = guard_env_name(env)
        latex_im = f'\subfloat{{\includegraphics[width=1.22in]{{images/{env}_multi_trial_graph_mean_returns_ma_vs_frames.png}}}}'
        latex_ims.append(latex_im)

    im_matrix = ps.chunk(latex_ims, 4)
    latex_im_body = '\\\\\n'.join([' & \n'.join(row) for row in im_matrix])
    return latex_im_body


def get_trial_metrics_path(algo, env, data_folder):
    filepaths = glob(f'{data_folder}/{algo}*{env}*/info/{trial_metrics_path}')
    assert len(filepaths) == 1, f'{algo}, {env}, {filepaths}'
    return filepaths[0]


def plot_env(algos, env, data_folder, legend_list=None, frame_scales=None):
    trial_metrics_path_list = []
    for idx, algo in enumerate(algos):
        try:
            trial_metrics_path_list.append(get_trial_metrics_path(algo, env, data_folder))
        except Exception as e:
            if legend_list is not None:  # remove from legend
                del legend_list[idx]
            logger.warning(f'Nothing to plot for algo: {algo}, env: {env}')
    env = guard_env_name(env)
    title = env
    if legend_list is None:
        graph_prepath = f'{data_folder}/{env}'
    else:
        graph_prepath = f'{data_folder}/{env}-legend'
    if legend_list is not None:
        palette = [master_palette_dict[k] for k in legend_list]
    else:
        palette = None
    viz.plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath, ma=True, name_time_pairs=[('mean_returns', 'frames')], frame_scales=frame_scales, palette=palette)


def plot_envs(algos, envs, data_folder, legend_list, frame_scales=None):
    for idx, env in enumerate(envs):
        try:
            plot_env(algos, env, data_folder, frame_scales=frame_scales)
            if idx == len(envs) - 1:
                # plot extra to crop legend out
                plot_env(algos, env, data_folder, legend_list=legend_list, frame_scales=frame_scales)
        except Exception as e:
            logger.warning(f'Cant plot for env: {env}. Error: {e}')


# Discrete
# LunarLander + Small Atari + Unity
data_folder = util.smart_path('../Desktop/benchmark/discrete')

algos = [
    'dqn',
    'ddqn_per',
    'a2c_gae',
    'a2c_nstep',
    'ppo',
    '*sac',
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
    'BeamRider',
    'Breakout',
    'MsPacman',
    'ong',
    'Seaquest',
    'SpaceInvaders',
    'bert',
    'lunar',
    'UnityHallway',
    'UnityPushBlock',
    'UnityPyramids',
]

latex_body = get_latex_body(algos, envs, data_folder)
print(latex_body)
latex_im_body = get_latex_im_body(envs)
print(latex_im_body)

# plot normal
envs = [
    'BeamRider',
    'Breakout',
    'MsPacman',
    'Seaquest',
    'SpaceInvaders',
    'lunar',
    'UnityHallway',
    'UnityPushBlock',
    'UnityPyramids',
]
plot_envs(algos, envs, data_folder, legend_list)

# Replot Pong and Qbert for Async SAC
envs = [
    'ong',
    'bert',
]
plot_envs(algos, envs, data_folder, legend_list, frame_scales=[(-1, 6)])



# Continuous
# Roboschool + Unity
data_folder = util.smart_path('../Desktop/benchmark/cont')

algos = [
    'a2c_gae',
    'a2c_nstep',
    'ppo',
    '*sac',
]
legend_list = [
    'A2C (GAE)',
    'A2C (n-step)',
    'PPO',
    'SAC',
]
envs = [
    'RoboschoolAnt',
    'RoboschoolAtlasForwardWalk',
    'RoboschoolHalfCheetah',
    'RoboschoolHopper',
    'RoboschoolInvertedDoublePendulum',
    'RoboschoolInvertedPendulum',
    'RoboschoolReacher',
    'RoboschoolWalker2d',
    'humanoid_',
    'humanoidflagrun_',
    'humanoidflagrunharder',
    'Unity3DBall-',
    'Unity3DBallHard',
    # 'UnityCrawlerDynamic',
    'UnityCrawlerStatic',
    'UnityReacher',
    # 'UnityWalker',
]

latex_body = get_latex_body(algos, envs, data_folder)
print(latex_body)
latex_im_body = get_latex_im_body(envs)
print(latex_im_body)


# plot simple
envs = [
    'RoboschoolAnt',
    'RoboschoolAtlasForwardWalk',
    'RoboschoolHalfCheetah',
    'RoboschoolHopper',
    'RoboschoolInvertedDoublePendulum',
    'RoboschoolInvertedPendulum',
    'RoboschoolReacher',
    'RoboschoolWalker2d',
    'Unity3DBall-',
    'Unity3DBallHard',
    # 'UnityCrawlerDynamic',
    'UnityCrawlerStatic',
    'UnityReacher',
    # 'UnityWalker',
]
plot_envs(algos, envs, data_folder, legend_list)


# plot humanoids with async sac
envs = [
    'humanoid_',
]
plot_envs(algos, envs, data_folder, legend_list, frame_scales=[(-1, 16)])

envs = [
    'humanoidflagrun_',
]
plot_envs(algos, envs, data_folder, legend_list, frame_scales=[(-1, 32)])


# exclude n-step since it's out of scale
algos = [
    'a2c_gae',
    'ppo',
    'async_sac',
]
legend_list = [
    'A2C (GAE)',
    'PPO',
    'Async SAC',
]
# plot harder humanoids with more workers
envs = [
    'humanoidflagrunharder',
]
plot_envs(algos, envs, data_folder, legend_list, frame_scales=[(-1, 32)])





# Atari full
data_folder = util.smart_path('../Desktop/benchmark/atari')

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

latex_body = get_latex_body(algos, envs, data_folder)
print(latex_body)
latex_im_body = get_latex_im_body(envs)
print(latex_im_body)
plot_envs(algos, envs, data_folder, legend_list)
