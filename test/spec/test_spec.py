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


@pytest.mark.parametrize('spec_file,spec_name', [
    ('ac.json', 'ac_cartpole'),
    ('ac.json', 'ac_shared_cartpole'),
    ('ac.json', 'ac_recurrent_cartpole'),
    # ('ac.json', 'ac_conv_breakout'),
])
def test_actor_critic(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('a2c.json', 'a2c_cartpole'),
    ('a2c.json', 'a2c_recurrent_cartpole'),
    ('a2c.json', 'a2c_shared_cartpole'),
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


# @pytest.mark.parametrize('spec_file,spec_name', [
#     ('ppo.json', 'ppo_cartpole'),
# ])
# def test_ppo(spec_file, spec_name):
#     run_trial_test(spec_file, spec_name)


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
