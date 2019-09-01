# Script to plot graphs from data/
from slm_lab.lib import logger, util, viz
import numpy as np

# Atari
trial_metrics_path_list = [
    'data/dqn_atari_PongNoFrameskip-v4_2019_07_28_142154/info/dqn_atari_PongNoFrameskip-v4_t0_trial_metrics.pkl',  # DQN
    'data/ddqn_per_atari_PongNoFrameskip-v4_2019_07_30_000958/info/ddqn_per_atari_PongNoFrameskip-v4_t0_trial_metrics.pkl',  # DDQN PER
    'data/a2c_nstep_atari_PongNoFrameskip-v4_2019_07_28_020953/info/a2c_nstep_atari_PongNoFrameskip-v4_t0_trial_metrics.pkl',  # A2C Nstep
    'data/a2c_gae_atari_PongNoFrameskip-v4_2019_07_28_084758/info/a2c_gae_atari_PongNoFrameskip-v4_t0_trial_metrics.pkl',  # A2C GAE
    'data/ppo_atari_PongNoFrameskip-v4_2019_07_29_042926/info/ppo_atari_PongNoFrameskip-v4_t0_trial_metrics.pkl',  # PPO
]
legend_list = [
    'DQN',
    'DoubleDQN+PER',
    'A2C (n-step)',
    'A2C (GAE)',
    'PPO',
]
title = f'multi trial graph: Pong'
graph_prepath = 'data/benchmark_pong'
viz.plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath)
viz.plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath, ma=True)


# Roboschool
env_list = [
    'RoboschoolAnt',
    'RoboschoolAtlasForwardWalk',
    'RoboschoolHalfCheetah',
    'RoboschoolHopper',
    'RoboschoolInvertedDoublePendulum',
    'RoboschoolInvertedPendulum',
    'RoboschoolInvertedPendulumSwingup',
    'RoboschoolReacher',
    'RoboschoolWalker2d',
]

for env in env_list:
    trial_metrics_path_list = [
        f'data/a2c_gae_roboschool_{env}-v1_2019_08_27_135211/info/a2c_gae_roboschool_{env}-v1_t0_trial_metrics.pkl',
        f'data/a2c_nstep_roboschool_{env}-v1_2019_08_27_075653/info/a2c_nstep_roboschool_{env}-v1_t0_trial_metrics.pkl',
        f'data/ppo_roboschool_{env}-v1_2019_08_27_182010/info/ppo_roboschool_{env}-v1_t0_trial_metrics.pkl',
        f'data/sac_roboschool_{env}-v1_2019_08_24_201811/info/sac_roboschool_{env}-v1_t0_trial_metrics.pkl',
    ]
    legend_list = [
        'A2C (GAE)',
        'A2C (n-step)',
        'PPO',
        'SAC',
    ]
    title = f'multi trial graph: {env}'
    graph_prepath = f'data/almanac_{env}'
    viz.plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath)
    viz.plot_multi_trial(trial_metrics_path_list, legend_list, title, graph_prepath, ma=True)
