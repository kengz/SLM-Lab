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
])
def test_multi_body(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('base.json', 'multi_env'),
])
def test_multi_env(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


# @pytest.mark.skip(reason='TODO broken by pytorch in CI https://circleci.com/gh/kengz/SLM-Lab/997')
@pytest.mark.parametrize('spec_file,spec_name', [
    ('reinforce.json', 'reinforce_cartpole'),
])
def test_pg_algo(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


# @pytest.mark.skip(reason='TODO broken by pytorch in CI https://circleci.com/gh/kengz/SLM-Lab/997')
@pytest.mark.parametrize('spec_file,spec_name', [
    ('dqn.json', 'dqn_cartpole'),
    ('sarsa.json', 'sarsa_cartpole'),
])
def test_value_algo(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


# @pytest.mark.skip(reason='TODO broken by pytorch in CI https://circleci.com/gh/kengz/SLM-Lab/997')
@pytest.mark.parametrize('spec_file,spec_name', [
    ('actor_critic.json', 'actor_critic_cartpole'),
    ('ppo.json', 'ppo_cartpole'),
])
def test_pg_value_algo(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)
