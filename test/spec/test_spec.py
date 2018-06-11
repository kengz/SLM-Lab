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
    ('benchmark_cartpole.json', 'reinforce_mlp_cartpole'),
    ('benchmark_cartpole.json', 'reinforce_recurrent_cartpole'),
    ('benchmark_cartpole.json', 'ac_mlp_shared_cartpole'),
    ('benchmark_cartpole.json', 'ac_mlp_separate_cartpole'),
    ('benchmark_cartpole.json', 'ac_rnn_shared_cartpole'),
    ('benchmark_cartpole.json', 'ac_rnn_separate_cartpole'),
    ('benchmark_cartpole.json', 'a2c_mlp_shared_cartpole'),
    ('benchmark_cartpole.json', 'a2c_mlp_separate_cartpole'),
    ('benchmark_cartpole.json', 'a2c_rnn_shared_cartpole'),
    ('benchmark_cartpole.json', 'a2c_rnn_separate_cartpole'),
    ('benchmark_cartpole.json', 'ppo_mlp_shared_cartpole'),
    ('benchmark_cartpole.json', 'ppo_mlp_separate_cartpole'),
    ('benchmark_cartpole.json', 'ppo_rnn_shared_cartpole'),
    ('benchmark_cartpole.json', 'ppo_rnn_separate_cartpole'),
    ('benchmark_cartpole.json', 'sarsa_mlp_boltzmann_cartpole'),
    ('benchmark_cartpole.json', 'sarsa_mlp_epsilon_greedy_cartpole'),
    ('benchmark_cartpole.json', 'sarsa_rnn_boltzmann_cartpole'),
    ('benchmark_cartpole.json', 'sarsa_rnn_epsilon_greedy_cartpole'),
    ('benchmark_cartpole.json', 'dqn_boltzmann_cartpole'),
    ('benchmark_cartpole.json', 'dqn_epsilon_greedy_cartpole'),
    ('benchmark_cartpole.json', 'ddqn_boltzmann_cartpole'),
    ('benchmark_cartpole.json', 'ddqn_epsilon_greedy_cartpole'),
    ('benchmark_cartpole.json', 'drqn_boltzmann_cartpole'),
    ('benchmark_cartpole.json', 'drqn_epsilon_greedy_cartpole'),
    ('benchmark_cartpole.json', 'ddrqn_boltzmann_cartpole'),
    ('benchmark_cartpole.json', 'ddrqn_epsilon_greedy_cartpole'),
    ('benchmark_cartpole.json', 'multitask_dqn_boltzmann_cartpole'),
    ('benchmark_cartpole.json', 'multitask_dqn_epsilon_greedy_cartpole'),
    ('benchmark_cartpole.json', 'hydra_dqn_boltzmann_cartpole'),
    ('benchmark_cartpole.json', 'hydra_dqn_epsilon_greedy_cartpole'),
])
def test_all(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


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


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ac.json', 'ac_cartpole'),
    ('ac.json', 'ac_shared_cartpole'),
    ('ac.json', 'ac_batch_cartpole'),
    ('ac.json', 'ac_recurrent_cartpole'),
    # ('ac.json', 'ac_conv_breakout'),
])
def test_actor_critic(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('a2c.json', 'a2c_cartpole'),
    ('a2c.json', 'a2c_shared_cartpole'),
    ('a2c.json', 'a2c_batch_cartpole'),
    ('a2c.json', 'a2c_recurrent_cartpole'),
    # ('a2c.json', 'a2c_conv_breakout'),
])
def test_a2c(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('dqn.json', 'dqn_cartpole'),
    ('dqn.json', 'double_dqn_cartpole_replace'),
    ('dqn.json', 'multitask_dqn_cartpole'),
    ('dqn.json', 'hydra_dqn_cartpole'),
    # ('dqn_atari.json', 'dqn_breakout'),
])
def test_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ppo.json', 'ppo_cartpole'),
    # ('ppo.json', 'ppo_shared_cartpole'),
    # ('ppo.json', 'ppo_batch_cartpole'),
    # ('ppo.json', 'ppo_recurrent_cartpole'),
])
def test_ppo(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('reinforce.json', 'reinforce_cartpole'),
    ('reinforce.json', 'reinforce_cartpole_recurrent'),
    # ('reinforce.json', 'reinforce_conv_breakout'),
])
def test_reinforce(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('sarsa.json', 'sarsa_cartpole'),
    ('sarsa.json', 'sarsa_cartpole_recurrent'),
    ('sarsa.json', 'sarsa_cartpole_episodic'),
])
def test_sarsa(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)
