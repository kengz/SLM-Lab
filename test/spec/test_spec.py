from slm_lab.experiment.control import Trial, Session
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import pandas as pd
import pytest


# helper method to run all tests below, split for parallelization
def run_trial_test(spec_file, spec_name):
    spec = spec_util.get(spec_file, spec_name)
    spec = util.override_test_spec(spec)
    trial = Trial(spec)
    trial_data = trial.run()
    assert isinstance(trial_data, pd.DataFrame)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('base.json', 'base_case'),
    ('base.json', 'base_case_openai'),
    ('random.json', 'random_cartpole'),
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


# NOTE mute conv tests for onpolicy memory because there is no preprocessor yet and images will be very large
@pytest.mark.parametrize('spec_file,spec_name', [
    ('reinforce.json', 'reinforce_mlp_cartpole'),
    ('reinforce.json', 'reinforce_rnn_cartpole'),
    # ('reinforce.json', 'reinforce_conv_breakout'),
])
def test_reinforce(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ac.json', 'ac_mlp_shared_cartpole'),
    ('ac.json', 'ac_mlp_separate_cartpole'),
    ('ac.json', 'ac_rnn_shared_cartpole'),
    ('ac.json', 'ac_rnn_separate_cartpole'),
    # ('ac.json', 'ac_conv_shared_breakout'),
    # ('ac.json', 'ac_conv_separate_breakout'),
])
def test_ac(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('a2c.json', 'a2c_mlp_shared_cartpole'),
    ('a2c.json', 'a2c_mlp_separate_cartpole'),
    ('a2c.json', 'a2c_rnn_shared_cartpole'),
    ('a2c.json', 'a2c_rnn_separate_cartpole'),
    # ('a2c.json', 'a2c_conv_shared_breakout'),
    # ('a2c.json', 'a2c_conv_separate_breakout'),
])
def test_a2c(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ppo.json', 'ppo_mlp_shared_cartpole'),
    ('ppo.json', 'ppo_mlp_separate_cartpole'),
    ('ppo.json', 'ppo_rnn_shared_cartpole'),
    ('ppo.json', 'ppo_rnn_separate_cartpole'),
    # ('ppo.json', 'ppo_conv_shared_breakout'),
    # ('ppo.json', 'ppo_conv_separate_breakout'),
])
def test_ppo(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('sarsa.json', 'sarsa_mlp_boltzmann_cartpole'),
    ('sarsa.json', 'sarsa_mlp_epsilon_greedy_cartpole'),
    ('sarsa.json', 'sarsa_rnn_boltzmann_cartpole'),
    ('sarsa.json', 'sarsa_rnn_epsilon_greedy_cartpole'),
    # ('sarsa.json', 'sarsa_conv_boltzmann_breakout'),
    # ('sarsa.json', 'sarsa_conv_epsilon_greedy_breakout'),
])
def test_sarsa(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('dqn.json', 'dqn_boltzmann_cartpole'),
    ('dqn.json', 'dqn_epsilon_greedy_cartpole'),
    ('dqn.json', 'drqn_boltzmann_cartpole'),
    ('dqn.json', 'drqn_epsilon_greedy_cartpole'),
    ('dqn.json', 'dqn_boltzmann_breakout'),
    ('dqn.json', 'dqn_epsilon_greedy_breakout'),
])
def test_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ddqn.json', 'ddqn_boltzmann_cartpole'),
    ('ddqn.json', 'ddqn_epsilon_greedy_cartpole'),
    ('ddqn.json', 'ddrqn_boltzmann_cartpole'),
    ('ddqn.json', 'ddrqn_epsilon_greedy_cartpole'),
    ('ddqn.json', 'ddqn_boltzmann_breakout'),
    ('ddqn.json', 'ddqn_epsilon_greedy_breakout'),
])
def test_ddqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('multitask_dqn.json', 'multitask_dqn_boltzmann_cartpole'),
    ('multitask_dqn.json', 'multitask_dqn_epsilon_greedy_cartpole'),
])
def test_multitask_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('hydra_dqn.json', 'hydra_dqn_boltzmann_cartpole'),
    ('hydra_dqn.json', 'hydra_dqn_epsilon_greedy_cartpole'),
])
def test_multitask_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)
