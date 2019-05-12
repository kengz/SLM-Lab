from flaky import flaky
from slm_lab.agent.net import net_util
from slm_lab.experiment import analysis
from slm_lab.experiment.control import Trial
from slm_lab.lib import util
from slm_lab.spec import spec_util
import os
import pandas as pd
import pydash as ps
import pytest


# helper method to run all tests in test_spec
def run_trial_test_dist(spec_file, spec_name=False):
    spec = spec_util.get(spec_file, spec_name)
    spec = spec_util.override_test_spec(spec)
    spec_util.tick(spec, 'trial')
    spec['meta']['distributed'] = True
    spec['meta']['max_session'] = 2

    trial = Trial(spec)
    # manually run the logic to obtain global nets for testing to ensure global net gets updated
    global_nets = trial.init_global_nets()
    # only test first network
    if ps.is_list(global_nets):  # multiagent only test first
        net = list(global_nets[0].values())[0]
    else:
        net = list(global_nets.values())[0]
    session_datas = trial.parallelize_sessions(global_nets)
    trial.session_data_dict = {data.index[0]: data for data in session_datas}
    trial_data = analysis.analyze_trial(trial)
    trial.close()
    assert isinstance(trial_data, pd.DataFrame)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce.json', 'reinforce_mlp_cartpole'),
    ('experimental/reinforce.json', 'reinforce_rnn_cartpole'),
    # ('experimental/reinforce.json', 'reinforce_conv_breakout'),
])
def test_reinforce_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce.json', 'reinforce_mlp_pendulum'),
    ('experimental/reinforce.json', 'reinforce_rnn_pendulum'),
])
def test_reinforce_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/a3c.json', 'a3c_gae_mlp_shared_cartpole'),
    ('experimental/a3c.json', 'a3c_gae_mlp_separate_cartpole'),
    ('experimental/a3c.json', 'a3c_gae_rnn_shared_cartpole'),
    ('experimental/a3c.json', 'a3c_gae_rnn_separate_cartpole'),
    # ('experimental/a3c.json', 'a3c_gae_conv_shared_breakout'),
    # ('experimental/a3c.json', 'a3c_gae_conv_separate_breakout'),
])
def test_a3c_gae_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/a3c.json', 'a3c_gae_mlp_shared_pendulum'),
    ('experimental/a3c.json', 'a3c_gae_mlp_separate_pendulum'),
    ('experimental/a3c.json', 'a3c_gae_rnn_shared_pendulum'),
    ('experimental/a3c.json', 'a3c_gae_rnn_separate_pendulum'),
])
def test_a3c_gae_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dppo.json', 'dppo_mlp_shared_cartpole'),
    ('experimental/dppo.json', 'dppo_mlp_separate_cartpole'),
    ('experimental/dppo.json', 'dppo_rnn_shared_cartpole'),
    ('experimental/dppo.json', 'dppo_rnn_separate_cartpole'),
    # ('experimental/dppo.json', 'dppo_conv_shared_breakout'),
    # ('experimental/dppo.json', 'dppo_conv_separate_breakout'),
])
def test_dppo_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo.json', 'ppo_mlp_shared_pendulum'),
    ('experimental/ppo.json', 'ppo_mlp_separate_pendulum'),
    ('experimental/ppo.json', 'ppo_rnn_shared_pendulum'),
    ('experimental/ppo.json', 'ppo_rnn_separate_pendulum'),
])
def test_ppo_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo_sil.json', 'ppo_sil_mlp_shared_cartpole'),
    ('experimental/ppo_sil.json', 'ppo_sil_mlp_separate_cartpole'),
    ('experimental/ppo_sil.json', 'ppo_sil_rnn_shared_cartpole'),
    ('experimental/ppo_sil.json', 'ppo_sil_rnn_separate_cartpole'),
])
def test_ppo_sil_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo_sil.json', 'ppo_sil_mlp_shared_pendulum'),
    ('experimental/ppo_sil.json', 'ppo_sil_mlp_separate_pendulum'),
    ('experimental/ppo_sil.json', 'ppo_sil_rnn_shared_pendulum'),
    ('experimental/ppo_sil.json', 'ppo_sil_rnn_separate_pendulum'),
])
def test_ppo_sil_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sil.json', 'sil_mlp_shared_cartpole'),
    ('experimental/sil.json', 'sil_mlp_separate_cartpole'),
    ('experimental/sil.json', 'sil_rnn_shared_cartpole'),
    ('experimental/sil.json', 'sil_rnn_separate_cartpole'),
    # ('experimental/sil.json', 'sil_conv_shared_breakout'),
    # ('experimental/sil.json', 'sil_conv_separate_breakout'),
])
def test_sil_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sil.json', 'sil_mlp_shared_pendulum'),
    ('experimental/sil.json', 'sil_mlp_separate_pendulum'),
    ('experimental/sil.json', 'sil_rnn_shared_pendulum'),
    ('experimental/sil.json', 'sil_rnn_separate_pendulum'),
])
def test_sil_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sarsa.json', 'sarsa_mlp_boltzmann_cartpole'),
    ('experimental/sarsa.json', 'sarsa_mlp_epsilon_greedy_cartpole'),
    ('experimental/sarsa.json', 'sarsa_rnn_boltzmann_cartpole'),
    ('experimental/sarsa.json', 'sarsa_rnn_epsilon_greedy_cartpole'),
    # ('experimental/sarsa.json', 'sarsa_conv_boltzmann_breakout'),
    # ('experimental/sarsa.json', 'sarsa_conv_epsilon_greedy_breakout'),
])
def test_sarsa_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dqn.json', 'vanilla_dqn_cartpole'),
    ('experimental/dqn.json', 'dqn_boltzmann_cartpole'),
    ('experimental/dqn.json', 'dqn_epsilon_greedy_cartpole'),
    ('experimental/dqn.json', 'drqn_boltzmann_cartpole'),
    ('experimental/dqn.json', 'drqn_epsilon_greedy_cartpole'),
    # ('experimental/dqn.json', 'dqn_boltzmann_breakout'),
    # ('experimental/dqn.json', 'dqn_epsilon_greedy_breakout'),
    ('experimental/dqn.json', 'dqn_stack_epsilon_greedy_lunar'),
])
def test_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ddqn.json', 'ddqn_boltzmann_cartpole'),
    ('experimental/ddqn.json', 'ddqn_epsilon_greedy_cartpole'),
    ('experimental/ddqn.json', 'ddrqn_boltzmann_cartpole'),
    ('experimental/ddqn.json', 'ddrqn_epsilon_greedy_cartpole'),
    # ('experimental/ddqn.json', 'ddqn_boltzmann_breakout'),
    # ('experimental/ddqn.json', 'ddqn_epsilon_greedy_breakout'),
])
def test_ddqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dueling_dqn.json', 'dueling_dqn_boltzmann_cartpole'),
    ('experimental/dueling_dqn.json', 'dueling_dqn_epsilon_greedy_cartpole'),
    # ('experimental/dueling_dqn.json', 'dueling_dqn_boltzmann_breakout'),
    # ('experimental/dueling_dqn.json', 'dueling_dqn_epsilon_greedy_breakout'),
])
def test_dueling_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/hydra_dqn.json', 'hydra_dqn_boltzmann_cartpole'),
    ('experimental/hydra_dqn.json', 'hydra_dqn_epsilon_greedy_cartpole'),
])
def test_hydra_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)
