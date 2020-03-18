from flaky import flaky
from slm_lab.experiment.control import Trial
from slm_lab.spec import spec_util
import pytest


# helper method to run all tests in test_spec
def run_trial_test(spec_file, spec_name=False):
    spec = spec_util.get(spec_file, spec_name)
    spec = spec_util.override_test_spec(spec)
    spec_util.tick(spec, 'trial')
    trial = Trial(spec)
    trial_metrics = trial.run()
    assert isinstance(trial_metrics, dict)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_cartpole.json', 'reinforce_cartpole'),
])
def test_reinforce(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_pendulum.json', 'reinforce_pendulum'),
])
def test_reinforce_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sarsa/sarsa_cartpole.json', 'sarsa_epsilon_greedy_cartpole'),
    ('experimental/sarsa/sarsa_cartpole.json', 'sarsa_boltzmann_cartpole'),
])
def test_sarsa(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/a2c/a2c_cartpole.json', 'a2c_shared_cartpole'),
    ('experimental/a2c/a2c_cartpole.json', 'a2c_separate_cartpole'),
    ('experimental/a2c/a2c_cartpole.json', 'a2c_concat_cartpole'),
    ('experimental/a2c/a2c_cartpole.json', 'a2c_rnn_shared_cartpole'),
    ('experimental/a2c/a2c_cartpole.json', 'a2c_rnn_separate_cartpole'),
])
def test_a2c(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/a2c/a2c_pendulum.json', 'a2c_shared_pendulum'),
    ('experimental/a2c/a2c_pendulum.json', 'a2c_separate_pendulum'),
    ('experimental/a2c/a2c_pendulum.json', 'a2c_concat_pendulum'),
    ('experimental/a2c/a2c_pendulum.json', 'a2c_rnn_shared_pendulum'),
    ('experimental/a2c/a2c_pendulum.json', 'a2c_rnn_separate_pendulum'),
])
def test_a2c_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo/ppo_cartpole.json', 'ppo_shared_cartpole'),
    ('experimental/ppo/ppo_cartpole.json', 'ppo_separate_cartpole'),
    ('experimental/ppo/ppo_cartpole.json', 'ppo_rnn_shared_cartpole'),
    ('experimental/ppo/ppo_cartpole.json', 'ppo_rnn_separate_cartpole'),
])
def test_ppo(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo/ppo_pendulum.json', 'ppo_shared_pendulum'),
    ('experimental/ppo/ppo_pendulum.json', 'ppo_separate_pendulum'),
    ('experimental/ppo/ppo_pendulum.json', 'ppo_rnn_shared_pendulum'),
    ('experimental/ppo/ppo_pendulum.json', 'ppo_rnn_separate_pendulum'),
])
def test_ppo_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sil/sil_cartpole.json', 'sil_shared_cartpole'),
    ('experimental/sil/sil_cartpole.json', 'sil_separate_cartpole'),
    ('experimental/sil/sil_cartpole.json', 'sil_rnn_shared_cartpole'),
    ('experimental/sil/sil_cartpole.json', 'sil_rnn_separate_cartpole'),
])
def test_sil(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sil/ppo_sil_cartpole.json', 'ppo_sil_shared_cartpole'),
    ('experimental/sil/ppo_sil_cartpole.json', 'ppo_sil_separate_cartpole'),
    ('experimental/sil/ppo_sil_cartpole.json', 'ppo_sil_rnn_shared_cartpole'),
    ('experimental/sil/ppo_sil_cartpole.json', 'ppo_sil_rnn_separate_cartpole'),
])
def test_ppo_sil(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/sac/sac_cartpole.json', 'sac_cartpole'),
    # ('benchmark/sac/sac_halfcheetah.json', 'sac_halfcheetah'),
])
def test_sac_cont(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dqn/dqn_cartpole.json', 'vanilla_dqn_cartpole'),
    ('experimental/dqn/dqn_cartpole.json', 'dqn_boltzmann_cartpole'),
    ('experimental/dqn/dqn_cartpole.json', 'dqn_epsilon_greedy_cartpole'),
    ('experimental/dqn/dqn_cartpole.json', 'drqn_boltzmann_cartpole'),
    ('experimental/dqn/dqn_cartpole.json', 'drqn_epsilon_greedy_cartpole'),
    ('benchmark/dqn/dqn_lunar.json', 'dqn_concat_lunar'),
])
def test_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dqn/ddqn_cartpole.json', 'ddqn_boltzmann_cartpole'),
    ('experimental/dqn/ddqn_cartpole.json', 'ddqn_epsilon_greedy_cartpole'),
    ('experimental/dqn/ddqn_cartpole.json', 'ddrqn_boltzmann_cartpole'),
    ('experimental/dqn/ddqn_cartpole.json', 'ddrqn_epsilon_greedy_cartpole'),
    ('experimental/dqn/ddqn_lunar.json', 'ddqn_concat_lunar'),
])
def test_ddqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/dqn/dueling_dqn_cartpole.json', 'dueling_dqn_boltzmann_cartpole'),
    ('experimental/dqn/dueling_dqn_cartpole.json', 'dueling_dqn_epsilon_greedy_cartpole'),
])
def test_dueling_dqn(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@flaky
@pytest.mark.parametrize('spec_file,spec_name', [
    ('benchmark/dqn/dqn_pong.json', 'dqn_pong'),
    ('benchmark/a2c/a2c_gae_pong.json', 'a2c_gae_pong'),
])
def test_atari(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)


@pytest.mark.parametrize('spec_file,spec_name', [
    # ('experimental/misc/base.json', 'base_case_unity'),
    ('experimental/misc/base.json', 'base_case_openai'),
    ('experimental/misc/random.json', 'random_cartpole'),
    # ('experimental/misc/random.json', 'random_pendulum'),  # mp EOF error
])
def test_base(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)



@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo/ppo_coin_game.json', 'ppo_coin_game'),
    ('experimental/ppo/ppo_coin_game.json', 'ppo_coin_game_utilitarian'),
])
def test_ppo_coin_game(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)

@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/ppo/ppo_iterated_prisonner_dillema.json', 'ppo_ipd'),
    ('experimental/ppo/ppo_iterated_prisonner_dillema.json', 'ppo_ipd_utilitarian'),
])
def test_ppo_ipd(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)




@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_coin_game.json', 'reinforce_icoingame'),
])
def test_reinforce_coin_game(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)

@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_guessing_game.json', 'reinforce_guessing_game'),
])
def test_reinforce_guessing_game(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)

@pytest.mark.parametrize('spec_file,spec_name', [
    ('experimental/reinforce/reinforce_iterated_prisonner_dillema.json', 'reinforce_ipd'),
])
def test_reinforce_ipd(spec_file, spec_name):
    run_trial_test(spec_file, spec_name)
