from flaky import flaky
from slm_lab.experiment.control import Trial
from slm_lab.experiment.monitor import InfoSpace
from slm_lab.lib import util
from slm_lab.spec import spec_util
import os
import pandas as pd
import pytest
import sys


# helper method to run all tests in test_spec
def run_trial_test(spec_file, spec_name=False):
    spec = spec_util.get(spec_file, spec_name)
    spec = spec_util.override_test_spec(spec)
    info_space = InfoSpace()
    info_space.tick('trial')
    trial = Trial(spec, info_space)
    trial_data = trial.run()
    assert isinstance(trial_data, pd.DataFrame)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce.json', 'reinforce_mlp_cartpole'),
    ('experimental/reinforce.json', 'reinforce_rnn_cartpole'),
    # ('experimental/reinforce.json', 'reinforce_conv_breakout'),
])
def test_reinforce(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce.json', 'reinforce_mlp_pendulum'),
    ('experimental/reinforce.json', 'reinforce_rnn_pendulum'),
])
def test_reinforce_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/a2c.json', 'a2c_mlp_shared_cartpole'),
    ('experimental/a2c.json', 'a2c_mlp_separate_cartpole'),
    ('experimental/a2c.json', 'a2c_rnn_shared_cartpole'),
    ('experimental/a2c.json', 'a2c_rnn_separate_cartpole'),
    # ('experimental/a2c.json', 'a2c_conv_shared_breakout'),
    # ('experimental/a2c.json', 'a2c_conv_separate_breakout'),
    ('experimental/a2c.json', 'a2c_mlp_concat_cartpole'),
])
def test_a2c(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/a2c.json', 'a2c_mlp_shared_pendulum'),
    ('experimental/a2c.json', 'a2c_mlp_separate_pendulum'),
    ('experimental/a2c.json', 'a2c_rnn_shared_pendulum'),
    ('experimental/a2c.json', 'a2c_rnn_separate_pendulum'),
])
def test_a2c_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo.json', 'ppo_mlp_shared_cartpole'),
    ('experimental/ppo.json', 'ppo_mlp_separate_cartpole'),
    ('experimental/ppo.json', 'ppo_rnn_shared_cartpole'),
    ('experimental/ppo.json', 'ppo_rnn_separate_cartpole'),
    # ('experimental/ppo.json', 'ppo_conv_shared_breakout'),
    # ('experimental/ppo.json', 'ppo_conv_separate_breakout'),
])
def test_ppo(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo.json', 'ppo_mlp_shared_pendulum'),
    ('experimental/ppo.json', 'ppo_mlp_separate_pendulum'),
    ('experimental/ppo.json', 'ppo_rnn_shared_pendulum'),
    ('experimental/ppo.json', 'ppo_rnn_separate_pendulum'),
    # ('experimental/ppo_halfcheetah.json', 'ppo_halfcheetah'),
    # ('experimental/ppo_invertedpendulum.json', 'ppo_invertedpendulum'),
])
def test_ppo_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo_sil.json', 'ppo_sil_mlp_shared_cartpole'),
    ('experimental/ppo_sil.json', 'ppo_sil_mlp_separate_cartpole'),
    ('experimental/ppo_sil.json', 'ppo_sil_rnn_shared_cartpole'),
    ('experimental/ppo_sil.json', 'ppo_sil_rnn_separate_cartpole'),
])
def test_ppo_sil(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo_sil.json', 'ppo_sil_mlp_shared_pendulum'),
    ('experimental/ppo_sil.json', 'ppo_sil_mlp_separate_pendulum'),
    ('experimental/ppo_sil.json', 'ppo_sil_rnn_shared_pendulum'),
    ('experimental/ppo_sil.json', 'ppo_sil_rnn_separate_pendulum'),
])
def test_ppo_sil_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sil.json', 'sil_mlp_shared_cartpole'),
    ('experimental/sil.json', 'sil_mlp_separate_cartpole'),
    ('experimental/sil.json', 'sil_rnn_shared_cartpole'),
    ('experimental/sil.json', 'sil_rnn_separate_cartpole'),
    # ('experimental/sil.json', 'sil_conv_shared_breakout'),
    # ('experimental/sil.json', 'sil_conv_separate_breakout'),
])
def test_sil(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sil.json', 'sil_mlp_shared_pendulum'),
    ('experimental/sil.json', 'sil_mlp_separate_pendulum'),
    ('experimental/sil.json', 'sil_rnn_shared_pendulum'),
    ('experimental/sil.json', 'sil_rnn_separate_pendulum'),
])
def test_sil_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sarsa.json', 'sarsa_mlp_boltzmann_cartpole'),
    ('experimental/sarsa.json', 'sarsa_mlp_epsilon_greedy_cartpole'),
    ('experimental/sarsa.json', 'sarsa_rnn_boltzmann_cartpole'),
    ('experimental/sarsa.json', 'sarsa_rnn_epsilon_greedy_cartpole'),
    # ('experimental/sarsa.json', 'sarsa_conv_boltzmann_breakout'),
    # ('experimental/sarsa.json', 'sarsa_conv_epsilon_greedy_breakout'),
])
def test_sarsa(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


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
def test_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ddqn.json', 'ddqn_boltzmann_cartpole'),
    ('experimental/ddqn.json', 'ddqn_epsilon_greedy_cartpole'),
    ('experimental/ddqn.json', 'ddrqn_boltzmann_cartpole'),
    ('experimental/ddqn.json', 'ddrqn_epsilon_greedy_cartpole'),
    # ('experimental/ddqn.json', 'ddqn_boltzmann_breakout'),
    # ('experimental/ddqn.json', 'ddqn_epsilon_greedy_breakout'),
])
def test_ddqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dueling_dqn.json', 'dueling_dqn_boltzmann_cartpole'),
    ('experimental/dueling_dqn.json', 'dueling_dqn_epsilon_greedy_cartpole'),
    # ('experimental/dueling_dqn.json', 'dueling_dqn_boltzmann_breakout'),
    # ('experimental/dueling_dqn.json', 'dueling_dqn_epsilon_greedy_breakout'),
])
def test_dueling_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/hydra_dqn.json', 'hydra_dqn_boltzmann_cartpole'),
    ('experimental/hydra_dqn.json', 'hydra_dqn_epsilon_greedy_cartpole'),
    # ('experimental/hydra_dqn.json', 'hydra_dqn_epsilon_greedy_cartpole_2dball'),
])
def test_hydra_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dqn.json', 'dqn_pong'),
    # ('experimental/a2c.json', 'a2c_pong'),
])
def test_atari(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce.json', 'reinforce_conv_vizdoom'),
])
def test_reinforce_vizdoom(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('base.json', 'base_case_unity'),
    ('base.json', 'base_case_openai'),
    ('random.json', 'random_cartpole'),
    ('random.json', 'random_pendulum'),
    # ('base.json', 'multi_agent'),
    # ('base.json', 'multi_agent_multi_env'),
])
def test_base(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('base.json', 'multi_body'),
    ('base.json', 'multi_env'),
])
def test_base_multi(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)
