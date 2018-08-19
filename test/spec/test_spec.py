from flaky import flaky
from slm_lab.experiment.control import Trial, Session
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import os
import pandas as pd
import pytest


# helper method to run all tests below, split for parallelization
def run_trial_test(spec_file, spec_name, distributed=False):
    spec = spec_util.get(spec_file, spec_name)
    spec = util.override_test_spec(spec)
    if distributed:
        spec['meta']['distributed'] = True
        if os.environ.get('CI') != 'true':  # CI has not enough CPU
            spec['meta']['max_session'] = 2
    trial = Trial(spec)
    trial_data = trial.run()
    assert isinstance(trial_data, pd.DataFrame)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('reinforce.json', 'reinforce_mlp_cartpole', False),
    ('reinforce.json', 'reinforce_mlp_cartpole', False),
    ('reinforce.json', 'reinforce_rnn_cartpole', False),
    # ('reinforce.json', 'reinforce_conv_breakout', False),
    ('reinforce.json', 'reinforce_mlp_cartpole', True),
    ('reinforce.json', 'reinforce_mlp_cartpole', True),
    ('reinforce.json', 'reinforce_rnn_cartpole', True),
    # ('reinforce.json', 'reinforce_conv_breakout', True),
])
def test_reinforce(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('reinforce.json', 'reinforce_mlp_pendulum', False),
    ('reinforce.json', 'reinforce_rnn_pendulum', False),
    ('reinforce.json', 'reinforce_mlp_pendulum', True),
    ('reinforce.json', 'reinforce_rnn_pendulum', True),
])
def test_reinforce_cont(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('ac.json', 'ac_mlp_shared_cartpole', False),
    ('ac.json', 'ac_mlp_separate_cartpole', False),
    ('ac.json', 'ac_rnn_shared_cartpole', False),
    ('ac.json', 'ac_rnn_separate_cartpole', False),
    # ('ac.json', 'ac_conv_shared_breakout', False),
    # ('ac.json', 'ac_conv_separate_breakout', False),
    ('ac.json', 'ac_mlp_shared_cartpole', True),
    ('ac.json', 'ac_mlp_separate_cartpole', True),
    ('ac.json', 'ac_rnn_shared_cartpole', True),
    ('ac.json', 'ac_rnn_separate_cartpole', True),
    # ('ac.json', 'ac_conv_shared_breakout', True),
    # ('ac.json', 'ac_conv_separate_breakout', True),
])
def test_ac(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('ac.json', 'ac_mlp_shared_pendulum', False),
    ('ac.json', 'ac_mlp_separate_pendulum', False),
    ('ac.json', 'ac_rnn_shared_pendulum', False),
    ('ac.json', 'ac_rnn_separate_pendulum', False),
    ('ac.json', 'ac_mlp_shared_pendulum', True),
    ('ac.json', 'ac_mlp_separate_pendulum', True),
    ('ac.json', 'ac_rnn_shared_pendulum', True),
    ('ac.json', 'ac_rnn_separate_pendulum', True),
])
def test_ac_cont(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('a2c.json', 'a2c_mlp_shared_cartpole', False),
    ('a2c.json', 'a2c_mlp_separate_cartpole', False),
    ('a2c.json', 'a2c_rnn_shared_cartpole', False),
    ('a2c.json', 'a2c_rnn_separate_cartpole', False),
    # ('a2c.json', 'a2c_conv_shared_breakout', False),
    # ('a2c.json', 'a2c_conv_separate_breakout', False),
    ('a2c.json', 'a2c_mlp_shared_cartpole', True),
    ('a2c.json', 'a2c_mlp_separate_cartpole', True),
    ('a2c.json', 'a2c_rnn_shared_cartpole', True),
    ('a2c.json', 'a2c_rnn_separate_cartpole', True),
    # ('a2c.json', 'a2c_conv_shared_breakout', True),
    # ('a2c.json', 'a2c_conv_separate_breakout', True),
])
def test_a2c(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('a2c.json', 'a2c_mlp_shared_pendulum', False),
    ('a2c.json', 'a2c_mlp_separate_pendulum', False),
    ('a2c.json', 'a2c_rnn_shared_pendulum', False),
    ('a2c.json', 'a2c_rnn_separate_pendulum', False),
    ('a2c.json', 'a2c_mlp_shared_pendulum', True),
    ('a2c.json', 'a2c_mlp_separate_pendulum', True),
    ('a2c.json', 'a2c_rnn_shared_pendulum', True),
    ('a2c.json', 'a2c_rnn_separate_pendulum', True),
])
def test_a2c_cont(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('ppo.json', 'ppo_mlp_shared_cartpole', False),
    ('ppo.json', 'ppo_mlp_separate_cartpole', False),
    ('ppo.json', 'ppo_rnn_shared_cartpole', False),
    ('ppo.json', 'ppo_rnn_separate_cartpole', False),
    # ('ppo.json', 'ppo_conv_shared_breakout', False),
    # ('ppo.json', 'ppo_conv_separate_breakout', False),
    ('ppo.json', 'ppo_mlp_shared_cartpole', True),
    ('ppo.json', 'ppo_mlp_separate_cartpole', True),
    ('ppo.json', 'ppo_rnn_shared_cartpole', True),
    ('ppo.json', 'ppo_rnn_separate_cartpole', True),
    # ('ppo.json', 'ppo_conv_shared_breakout', True),
    # ('ppo.json', 'ppo_conv_separate_breakout', True),
])
def test_ppo(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('ppo.json', 'ppo_mlp_shared_pendulum', False),
    ('ppo.json', 'ppo_mlp_separate_pendulum', False),
    ('ppo.json', 'ppo_rnn_shared_pendulum', False),
    ('ppo.json', 'ppo_rnn_separate_pendulum', False),
    ('ppo.json', 'ppo_mlp_shared_pendulum', True),
    ('ppo.json', 'ppo_mlp_separate_pendulum', True),
    ('ppo.json', 'ppo_rnn_shared_pendulum', True),
    ('ppo.json', 'ppo_rnn_separate_pendulum', True),
])
def test_ppo_cont(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('ppo_sil.json', 'ppo_sil_mlp_shared_cartpole', False),
    ('ppo_sil.json', 'ppo_sil_mlp_separate_cartpole', False),
    ('ppo_sil.json', 'ppo_sil_rnn_shared_cartpole', False),
    ('ppo_sil.json', 'ppo_sil_rnn_separate_cartpole', False),
    ('ppo_sil.json', 'ppo_sil_mlp_shared_cartpole', True),
    ('ppo_sil.json', 'ppo_sil_mlp_separate_cartpole', True),
    ('ppo_sil.json', 'ppo_sil_rnn_shared_cartpole', True),
    ('ppo_sil.json', 'ppo_sil_rnn_separate_cartpole', True),
])
def test_ppo_sil(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('ppo_sil.json', 'ppo_sil_mlp_shared_pendulum', False),
    ('ppo_sil.json', 'ppo_sil_mlp_separate_pendulum', False),
    ('ppo_sil.json', 'ppo_sil_rnn_shared_pendulum', False),
    ('ppo_sil.json', 'ppo_sil_rnn_separate_pendulum', False),
    ('ppo_sil.json', 'ppo_sil_mlp_shared_pendulum', True),
    ('ppo_sil.json', 'ppo_sil_mlp_separate_pendulum', True),
    ('ppo_sil.json', 'ppo_sil_rnn_shared_pendulum', True),
    ('ppo_sil.json', 'ppo_sil_rnn_separate_pendulum', True),
])
def test_ppo_sil_cont(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('sil.json', 'sil_mlp_shared_cartpole', False),
    ('sil.json', 'sil_mlp_separate_cartpole', False),
    ('sil.json', 'sil_rnn_shared_cartpole', False),
    ('sil.json', 'sil_rnn_separate_cartpole', False),
    # ('sil.json', 'sil_conv_shared_breakout', False),
    # ('sil.json', 'sil_conv_separate_breakout', False),
    ('sil.json', 'sil_mlp_shared_cartpole', True),
    ('sil.json', 'sil_mlp_separate_cartpole', True),
    ('sil.json', 'sil_rnn_shared_cartpole', True),
    ('sil.json', 'sil_rnn_separate_cartpole', True),
    # ('sil.json', 'sil_conv_shared_breakout', True),
    # ('sil.json', 'sil_conv_separate_breakout', True),
])
def test_sil(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('sil.json', 'sil_mlp_shared_pendulum', False),
    ('sil.json', 'sil_mlp_separate_pendulum', False),
    ('sil.json', 'sil_rnn_shared_pendulum', False),
    ('sil.json', 'sil_rnn_separate_pendulum', False),
    ('sil.json', 'sil_mlp_shared_pendulum', True),
    ('sil.json', 'sil_mlp_separate_pendulum', True),
    ('sil.json', 'sil_rnn_shared_pendulum', True),
    ('sil.json', 'sil_rnn_separate_pendulum', True),
])
def test_sil_cont(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('sarsa.json', 'sarsa_mlp_boltzmann_cartpole', False),
    ('sarsa.json', 'sarsa_mlp_epsilon_greedy_cartpole', False),
    ('sarsa.json', 'sarsa_rnn_boltzmann_cartpole', False),
    ('sarsa.json', 'sarsa_rnn_epsilon_greedy_cartpole', False),
    # ('sarsa.json', 'sarsa_conv_boltzmann_breakout', False),
    # ('sarsa.json', 'sarsa_conv_epsilon_greedy_breakout', False),
    ('sarsa.json', 'sarsa_mlp_boltzmann_cartpole', True),
    ('sarsa.json', 'sarsa_mlp_epsilon_greedy_cartpole', True),
    ('sarsa.json', 'sarsa_rnn_boltzmann_cartpole', True),
    ('sarsa.json', 'sarsa_rnn_epsilon_greedy_cartpole', True),
    # ('sarsa.json', 'sarsa_conv_boltzmann_breakout', True),
    # ('sarsa.json', 'sarsa_conv_epsilon_greedy_breakout', True),
])
def test_sarsa(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@flaky
@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('dqn.json', 'vanilla_dqn_cartpole', False),
    ('dqn.json', 'dqn_boltzmann_cartpole', False),
    ('dqn.json', 'dqn_epsilon_greedy_cartpole', False),
    ('dqn.json', 'drqn_boltzmann_cartpole', False),
    ('dqn.json', 'drqn_epsilon_greedy_cartpole', False),
    # ('dqn.json', 'dqn_boltzmann_breakout', False),
    # ('dqn.json', 'dqn_epsilon_greedy_breakout', False),
    ('dqn.json', 'dqn_stack_epsilon_greedy_lunar', False),
    ('dqn.json', 'vanilla_dqn_cartpole', True),
    ('dqn.json', 'dqn_boltzmann_cartpole', True),
    ('dqn.json', 'dqn_epsilon_greedy_cartpole', True),
    ('dqn.json', 'drqn_boltzmann_cartpole', True),
    ('dqn.json', 'drqn_epsilon_greedy_cartpole', True),
    # ('dqn.json', 'dqn_boltzmann_breakout', True),
    # ('dqn.json', 'dqn_epsilon_greedy_breakout', True),
    ('dqn.json', 'dqn_stack_epsilon_greedy_lunar', True),
])
def test_dqn(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('ddqn.json', 'ddqn_boltzmann_cartpole', False),
    ('ddqn.json', 'ddqn_epsilon_greedy_cartpole', False),
    ('ddqn.json', 'ddrqn_boltzmann_cartpole', False),
    ('ddqn.json', 'ddrqn_epsilon_greedy_cartpole', False),
    # ('ddqn.json', 'ddqn_boltzmann_breakout', False),
    # ('ddqn.json', 'ddqn_epsilon_greedy_breakout', False),
    ('ddqn.json', 'ddqn_boltzmann_cartpole', True),
    ('ddqn.json', 'ddqn_epsilon_greedy_cartpole', True),
    ('ddqn.json', 'ddrqn_boltzmann_cartpole', True),
    ('ddqn.json', 'ddrqn_epsilon_greedy_cartpole', True),
    # ('ddqn.json', 'ddqn_boltzmann_breakout', True),
    # ('ddqn.json', 'ddqn_epsilon_greedy_breakout', True),
])
def test_ddqn(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('dueling_dqn.json', 'dueling_dqn_boltzmann_cartpole', False),
    ('dueling_dqn.json', 'dueling_dqn_epsilon_greedy_cartpole', False),
    # ('dueling_dqn.json', 'dueling_dqn_boltzmann_breakout', False),
    # ('dueling_dqn.json', 'dueling_dqn_epsilon_greedy_breakout', False),
    ('dueling_dqn.json', 'dueling_dqn_boltzmann_cartpole', True),
    ('dueling_dqn.json', 'dueling_dqn_epsilon_greedy_cartpole', True),
    # ('dueling_dqn.json', 'dueling_dqn_boltzmann_breakout', True),
    # ('dueling_dqn.json', 'dueling_dqn_epsilon_greedy_breakout', True),
])
def test_dueling_dqn(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('multitask_dqn.json', 'multitask_dqn_boltzmann_cartpole', False),
    ('multitask_dqn.json', 'multitask_dqn_epsilon_greedy_cartpole', False),
    ('multitask_dqn.json', 'multitask_dqn_boltzmann_cartpole', True),
    ('multitask_dqn.json', 'multitask_dqn_epsilon_greedy_cartpole', True),
])
def test_multitask_dqn(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.parametrize('spec_file,spec_name,distributed', [
    ('hydra_dqn.json', 'hydra_dqn_boltzmann_cartpole', False),
    ('hydra_dqn.json', 'hydra_dqn_epsilon_greedy_cartpole', False),
    ('hydra_dqn.json', 'hydra_dqn_boltzmann_cartpole', True),
    ('hydra_dqn.json', 'hydra_dqn_epsilon_greedy_cartpole', True),
])
def test_multitask_dqn(spec_file, spec_name, distributed):
    run_trial_test(spec_file, spec_name, distributed)


@pytest.mark.skipif(os.environ.get('CI') == 'true', reason="CI has not enough RAM")
@pytest.mark.parametrize('spec_file,spec_name', [
    ('dqn.json', 'dqn_boltzmann_breakout'),
    ('dqn.json', 'dqn_epsilon_greedy_breakout'),
    ('ddqn.json', 'ddqn_boltzmann_breakout'),
    ('ddqn.json', 'ddqn_epsilon_greedy_breakout'),
])
def test_dqn_breakout(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('base.json', 'base_case_unity'),
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
