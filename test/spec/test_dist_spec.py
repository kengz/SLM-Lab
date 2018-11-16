from slm_lab.agent.net import net_util
from slm_lab.experiment import analysis
from slm_lab.experiment.control import Trial
from slm_lab.experiment.monitor import InfoSpace
from slm_lab.lib import util
from slm_lab.spec import spec_util
import os
import pandas as pd
import pytest


# helper method to run all tests in test_spec
def run_trial_test_dist(spec_file, spec_name=False):
    spec = spec_util.get(spec_file, spec_name)
    spec = util.override_test_spec(spec)
    info_space = InfoSpace()
    info_space.tick('trial')
    spec['meta']['distributed'] = True
    spec['meta']['max_session'] = 2

    trial = Trial(spec, info_space)
    # manually run the logic to obtain global nets for testing to ensure global net gets updated
    global_nets = trial.init_global_nets()
    net = list(global_nets.values())[0]
    assert_trained = net_util.gen_assert_trained(net)
    session_datas = trial.parallelize_sessions(global_nets)
    assert_trained(net, loss=1.0)
    trial.session_data_dict = {data.index[0]: data for data in session_datas}
    trial_data = analysis.analyze_trial(trial)
    trial.close()
    assert isinstance(trial_data, pd.DataFrame)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('reinforce.json', 'reinforce_mlp_cartpole'),
    ('reinforce.json', 'reinforce_rnn_cartpole'),
    # ('reinforce.json', 'reinforce_conv_breakout'),
])
def test_reinforce_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('reinforce.json', 'reinforce_mlp_pendulum'),
    ('reinforce.json', 'reinforce_rnn_pendulum'),
])
def test_reinforce_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('a3c.json', 'a3c_gae_mlp_shared_cartpole'),
    ('a3c.json', 'a3c_gae_mlp_separate_cartpole'),
    ('a3c.json', 'a3c_gae_rnn_shared_cartpole'),
    ('a3c.json', 'a3c_gae_rnn_separate_cartpole'),
    # ('a3c.json', 'a3c_gae_conv_shared_breakout'),
    # ('a3c.json', 'a3c_gae_conv_separate_breakout'),
])
def test_a3c_gae_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('a3c.json', 'a3c_gae_mlp_shared_pendulum'),
    ('a3c.json', 'a3c_gae_mlp_separate_pendulum'),
    ('a3c.json', 'a3c_gae_rnn_shared_pendulum'),
    ('a3c.json', 'a3c_gae_rnn_separate_pendulum'),
])
def test_a3c_gae_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('dppo.json', 'dppo_mlp_shared_cartpole'),
    ('dppo.json', 'dppo_mlp_separate_cartpole'),
    ('dppo.json', 'dppo_rnn_shared_cartpole'),
    ('dppo.json', 'dppo_rnn_separate_cartpole'),
    # ('dppo.json', 'dppo_conv_shared_breakout'),
    # ('dppo.json', 'dppo_conv_separate_breakout'),
])
def test_dppo_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ppo.json', 'ppo_mlp_shared_pendulum'),
    ('ppo.json', 'ppo_mlp_separate_pendulum'),
    ('ppo.json', 'ppo_rnn_shared_pendulum'),
    ('ppo.json', 'ppo_rnn_separate_pendulum'),
])
def test_ppo_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ppo_sil.json', 'ppo_sil_mlp_shared_cartpole'),
    ('ppo_sil.json', 'ppo_sil_mlp_separate_cartpole'),
    ('ppo_sil.json', 'ppo_sil_rnn_shared_cartpole'),
    ('ppo_sil.json', 'ppo_sil_rnn_separate_cartpole'),
])
def test_ppo_sil_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ppo_sil.json', 'ppo_sil_mlp_shared_pendulum'),
    ('ppo_sil.json', 'ppo_sil_mlp_separate_pendulum'),
    ('ppo_sil.json', 'ppo_sil_rnn_shared_pendulum'),
    ('ppo_sil.json', 'ppo_sil_rnn_separate_pendulum'),
])
def test_ppo_sil_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('sil.json', 'sil_mlp_shared_cartpole'),
    ('sil.json', 'sil_mlp_separate_cartpole'),
    ('sil.json', 'sil_rnn_shared_cartpole'),
    ('sil.json', 'sil_rnn_separate_cartpole'),
    # ('sil.json', 'sil_conv_shared_breakout'),
    # ('sil.json', 'sil_conv_separate_breakout'),
])
def test_sil_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('sil.json', 'sil_mlp_shared_pendulum'),
    ('sil.json', 'sil_mlp_separate_pendulum'),
    ('sil.json', 'sil_rnn_shared_pendulum'),
    ('sil.json', 'sil_rnn_separate_pendulum'),
])
def test_sil_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('sarsa.json', 'sarsa_mlp_boltzmann_cartpole'),
    ('sarsa.json', 'sarsa_mlp_epsilon_greedy_cartpole'),
    ('sarsa.json', 'sarsa_rnn_boltzmann_cartpole'),
    ('sarsa.json', 'sarsa_rnn_epsilon_greedy_cartpole'),
    # ('sarsa.json', 'sarsa_conv_boltzmann_breakout'),
    # ('sarsa.json', 'sarsa_conv_epsilon_greedy_breakout'),
])
def test_sarsa_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('dqn.json', 'vanilla_dqn_cartpole'),
    ('dqn.json', 'dqn_boltzmann_cartpole'),
    ('dqn.json', 'dqn_epsilon_greedy_cartpole'),
    ('dqn.json', 'drqn_boltzmann_cartpole'),
    ('dqn.json', 'drqn_epsilon_greedy_cartpole'),
    # ('dqn.json', 'dqn_boltzmann_breakout'),
    # ('dqn.json', 'dqn_epsilon_greedy_breakout'),
    ('dqn.json', 'dqn_stack_epsilon_greedy_lunar'),
])
def test_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ddqn.json', 'ddqn_boltzmann_cartpole'),
    ('ddqn.json', 'ddqn_epsilon_greedy_cartpole'),
    ('ddqn.json', 'ddrqn_boltzmann_cartpole'),
    ('ddqn.json', 'ddrqn_epsilon_greedy_cartpole'),
    # ('ddqn.json', 'ddqn_boltzmann_breakout'),
    # ('ddqn.json', 'ddqn_epsilon_greedy_breakout'),
])
def test_ddqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('dueling_dqn.json', 'dueling_dqn_boltzmann_cartpole'),
    ('dueling_dqn.json', 'dueling_dqn_epsilon_greedy_cartpole'),
    # ('dueling_dqn.json', 'dueling_dqn_boltzmann_breakout'),
    # ('dueling_dqn.json', 'dueling_dqn_epsilon_greedy_breakout'),
])
def test_dueling_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('multitask_dqn.json', 'multitask_dqn_boltzmann_cartpole'),
    ('multitask_dqn.json', 'multitask_dqn_epsilon_greedy_cartpole'),
])
def test_multitask_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('hydra_dqn.json', 'hydra_dqn_boltzmann_cartpole'),
    ('hydra_dqn.json', 'hydra_dqn_epsilon_greedy_cartpole'),
])
def test_multitask_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)
