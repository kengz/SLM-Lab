from flaky import flaky
from slm_lab.agent.net import net_util
from slm_lab.experiment import analysis
from slm_lab.experiment.control import Trial
from slm_lab.lib import util
from slm_lab.spec import spec_util
import os
import pydash as ps
import pytest


# helper method to run all tests in test_spec
def run_trial_test_dist(spec_file, spec_name=False):
    spec = spec_util.get(spec_file, spec_name)
    spec = spec_util.override_test_spec(spec)
    spec_util.tick(spec, 'trial')
    spec['meta']['distributed'] = 'synced'
    spec['meta']['max_session'] = 2

    trial = Trial(spec)
    # manually run the logic to obtain global nets for testing to ensure global net gets updated
    global_nets = trial.init_global_nets()
    # only test first network
    if ps.is_list(global_nets):  # multiagent only test first
        net = list(global_nets[0].values())[0]
    else:
        net = list(global_nets.values())[0]
    session_metrics_list = trial.parallelize_sessions(global_nets)
    trial_metrics = analysis.analyze_trial(spec, session_metrics_list)
    trial.close()
    assert isinstance(trial_metrics, dict)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_cartpole.json', 'reinforce_cartpole'),
])
def test_reinforce_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_pendulum.json', 'reinforce_pendulum'),
])
def test_reinforce_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/a3c/a3c_cartpole.json', 'a3c_gae_shared_cartpole'),
    ('experimental/a3c/a3c_cartpole.json', 'a3c_gae_separate_cartpole'),
    ('experimental/a3c/a3c_cartpole.json', 'a3c_gae_rnn_shared_cartpole'),
    ('experimental/a3c/a3c_cartpole.json', 'a3c_gae_rnn_separate_cartpole'),
    # ('experimental/a3c.json', 'a3c_gae_conv_shared_breakout'),
    # ('experimental/a3c.json', 'a3c_gae_conv_separate_breakout'),
])
def test_a3c_gae_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


# @pytest.mark.parametrize('spec_file,spec_name', [
#     ('experimental/a3c.json', 'a3c_gae_mlp_shared_pendulum'),
#     ('experimental/a3c.json', 'a3c_gae_mlp_separate_pendulum'),
#     ('experimental/a3c.json', 'a3c_gae_rnn_shared_pendulum'),
#     ('experimental/a3c.json', 'a3c_gae_rnn_separate_pendulum'),
# ])
# def test_a3c_gae_cont_dist(spec_file, spec_name):
#     run_trial_test_dist(spec_file, spec_name)


# @pytest.mark.parametrize('spec_file,spec_name', [
#     ('experimental/dppo/ddqn_cartpole.json', 'dppo_mlp_shared_cartpole'),
#     ('experimental/dppo/ddqn_cartpole.json', 'dppo_mlp_separate_cartpole'),
#     ('experimental/dppo/ddqn_cartpole.json', 'dppo_rnn_shared_cartpole'),
#     ('experimental/dppo/ddqn_cartpole.json', 'dppo_rnn_separate_cartpole'),
#     # ('experimental/dppo.json', 'dppo_conv_shared_breakout'),
#     # ('experimental/dppo.json', 'dppo_conv_separate_breakout'),
# ])
# def test_dppo_dist(spec_file, spec_name):
#     run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo/ppo_pendulum.json', 'ppo_shared_pendulum'),
    ('experimental/ppo/ppo_pendulum.json', 'ppo_separate_pendulum'),
    ('experimental/ppo/ppo_pendulum.json', 'ppo_rnn_shared_pendulum'),
    ('experimental/ppo/ppo_pendulum.json', 'ppo_rnn_separate_pendulum'),
])
def test_ppo_cont_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sil/ppo_sil_cartpole.json', 'ppo_sil_shared_cartpole'),
    ('experimental/sil/ppo_sil_cartpole.json', 'ppo_sil_separate_cartpole'),
    ('experimental/sil/ppo_sil_cartpole.json', 'ppo_sil_rnn_shared_cartpole'),
    ('experimental/sil/ppo_sil_cartpole.json', 'ppo_sil_rnn_separate_cartpole'),
])
def test_ppo_sil_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


# @flaky
# @pytest.mark.parametrize('spec_file,spec_name', [
#     ('experimental/ppo_sil.json', 'ppo_sil_mlp_shared_pendulum'),
#     ('experimental/ppo_sil.json', 'ppo_sil_mlp_separate_pendulum'),
#     ('experimental/ppo_sil.json', 'ppo_sil_rnn_shared_pendulum'),
#     ('experimental/ppo_sil.json', 'ppo_sil_rnn_separate_pendulum'),
# ])
# def test_ppo_sil_cont_dist(spec_file, spec_name):
#     run_trial_test_dist(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sil/sil_cartpole.json', 'sil_shared_cartpole'),
    ('experimental/sil/sil_cartpole.json', 'sil_separate_cartpole'),
    ('experimental/sil/sil_cartpole.json', 'sil_rnn_shared_cartpole'),
    ('experimental/sil/sil_cartpole.json', 'sil_rnn_separate_cartpole'),
    # ('experimental/sil.json', 'sil_conv_shared_breakout'),
    # ('experimental/sil.json', 'sil_conv_separate_breakout'),
])
def test_sil_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


# @flaky
# @pytest.mark.parametrize('spec_file,spec_name', [
#     ('experimental/sil.json', 'sil_mlp_shared_pendulum'),
#     ('experimental/sil.json', 'sil_mlp_separate_pendulum'),
#     ('experimental/sil.json', 'sil_rnn_shared_pendulum'),
#     ('experimental/sil.json', 'sil_rnn_separate_pendulum'),
# ])
# def test_sil_cont_dist(spec_file, spec_name):
#     run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sarsa/sarsa_cartpole.json', 'sarsa_epsilon_greedy_cartpole'),
    ('experimental/sarsa/sarsa_cartpole.json', 'sarsa_boltzmann_cartpole'),
])
def test_sarsa_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dqn/dqn_cartpole.json', 'vanilla_dqn_cartpole'),
    ('experimental/dqn/dqn_cartpole.json', 'dqn_boltzmann_cartpole'),
    ('experimental/dqn/dqn_cartpole.json', 'dqn_epsilon_greedy_cartpole'),
    ('experimental/dqn/dqn_cartpole.json', 'drqn_boltzmann_cartpole'),
    ('experimental/dqn/dqn_cartpole.json', 'drqn_epsilon_greedy_cartpole'),
    # ('experimental/dqn.json', 'dqn_boltzmann_breakout'),
    # ('experimental/dqn.json', 'dqn_epsilon_greedy_breakout'),
    ('experimental/dqn/dqn_lunar_search.json', 'vanilla_dqn_concat_lunar'),
    ('experimental/dqn/dqn_lunar_search.json', 'dqn_concat_replace_lunar'),
    ('experimental/dqn/dqn_lunar_search.json', 'dqn_concat_polyak_lunar'),
])
def test_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dqn/ddqn_cartpole.json', 'ddqn_boltzmann_cartpole'),
    ('experimental/dqn/ddqn_cartpole.json', 'ddqn_epsilon_greedy_cartpole'),
    ('experimental/dqn/ddqn_cartpole.json', 'ddrqn_boltzmann_cartpole'),
    ('experimental/dqn/ddqn_cartpole.json', 'ddrqn_epsilon_greedy_cartpole'),
    # ('experimental/ddqn.json', 'ddqn_boltzmann_breakout'),
    # ('experimental/ddqn.json', 'ddqn_epsilon_greedy_breakout'),
])
def test_ddqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dqn/dueling_dqn_cartpole.json', 'dueling_dqn_boltzmann_cartpole'),
    ('experimental/dqn/dueling_dqn_cartpole.json', 'dueling_dqn_epsilon_greedy_cartpole'),
    # ('experimental/dueling_dqn.json', 'dueling_dqn_boltzmann_breakout'),
    # ('experimental/dueling_dqn.json', 'dueling_dqn_epsilon_greedy_breakout'),
])
def test_dueling_dqn_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)




@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo/coin_ppo.json', 'ppo_coin_game'),
    ('experimental/ppo/coin_ppo.json', 'ppo_coin_game_utilitarian'),
])
def test_ppo_coin_game_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)

@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo/ppo_iterated_prisonner_dillema.json', 'ppo_ipd'),
    ('experimental/ppo/ppo_iterated_prisonner_dillema.json', 'ppo_ipd_utilitarian'),
])
def test_ppo_ipd_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)




@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_coin_game.json', 'reinforce_icoingame'),
])
def test_reinforce_coin_game_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)

@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_guessing_game.json', 'reinforce_guessing_game'),
])
def test_reinforce_guessing_game_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)

@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_iterated_prisonner_dillema.json', 'reinforce_ipd'),
])
def test_reinforce_ipd_dist(spec_file, spec_name):
    run_trial_test_dist(spec_file, spec_name)
