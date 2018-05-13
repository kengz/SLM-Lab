from slm_lab.experiment.monitor import InfoSpace
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import util
from slm_lab.spec import spec_util
from flaky import flaky
import pytest
import os
import shutil
# TODO Fix pytest seg faults on Ubuntu


def generic_algorithm_test(spec, algorithm_name):
    '''Need new InfoSpace() per trial otherwise session id doesn't tick correctly'''
    trial = Trial(spec, info_space=InfoSpace())
    trial_data = trial.run()
    folders = [x for x in os.listdir('data/') if x.startswith(algorithm_name)]
    assert len(folders) == 1
    path = 'data/' + folders[0]
    sess_data = util.read(path + '/' + algorithm_name + '_t0_s0_session_df.csv')
    rewards = sess_data['0.2'].replace("reward", -1).astype(float)
    print(f'rewards: {rewards}')
    maxr = rewards.max()
    '''Delete test data folder and trial'''
    shutil.rmtree(path)
    del trial
    return maxr


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=5)
def test_sarsa():
    algorithm_name = 'unit_test_sarsa'
    spec = spec_util.get('test.json', 'unit_test_sarsa')
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=5)
def test_sarsa_episodic():
    algorithm_name = 'unit_test_sarsa_episodic'
    spec = spec_util.get('test.json', 'unit_test_sarsa')
    spec['name'] = algorithm_name
    spec['agent'][0]['memory']['name'] = "OnPolicyReplay"
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=5)
def test_sarsa_recurrent():
    algorithm_name = 'unit_test_sarsa_recurrent'
    spec = spec_util.get('test.json', 'unit_test_sarsa')
    spec['name'] = algorithm_name
    spec['agent'][0]['memory']['name'] = "OnPolicyNStepBatchReplay"
    spec['agent'][0]['net']['seq_len'] = 4
    spec['agent'][0]['net']['type'] = "RecurrentNet"
    spec['agent'][0]['net']['hid_layers'] = [64]
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_dqn():
    algorithm_name = 'unit_test_dqn'
    spec = spec_util.get('test.json', 'unit_test_dqn')
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_vanilla_dqn():
    algorithm_name = 'unit_test_vanilla_dqn'
    spec = spec_util.get('test.json', 'unit_test_dqn')
    spec['name'] = algorithm_name
    spec['agent'][0]['algorithm']['name'] = "VanillaDQN"
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_double_dqn():
    algorithm_name = 'unit_test_double_dqn'
    spec = spec_util.get('test.json', 'unit_test_dqn')
    spec['name'] = algorithm_name
    spec['agent'][0]['algorithm']['name'] = "DoubleDQN"
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_reinforce():
    algorithm_name = 'unit_test_reinforce'
    spec = spec_util.get('test.json', 'unit_test_reinforce')
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_reinforce_with_entropy():
    algorithm_name = 'unit_test_reinforce_with_entropy'
    spec = spec_util.get('test.json', 'unit_test_reinforce')
    spec['name'] = algorithm_name
    spec['agent'][0]['algorithm']['add_entropy'] = True
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_reinforce_multi_epi():
    algorithm_name = 'unit_test_reinforce_multi_epi'
    spec = spec_util.get('test.json', 'unit_test_reinforce')
    spec['name'] = algorithm_name
    spec['agent'][0]['algorithm']['training_frequency'] = 3
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_reinforce_recurrent():
    algorithm_name = 'unit_test_reinforce_recurrent'
    spec = spec_util.get('test.json', 'unit_test_reinforce')
    spec['name'] = algorithm_name
    spec['agent'][0]['memory']['name'] = "OnPolicyNStepReplay"
    spec['agent'][0]['net']['seq_len'] = 4
    spec['agent'][0]['net']['type'] = "RecurrentNet"
    spec['agent'][0]['net']['hid_layers'] = [16]
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic():
    algorithm_name = 'unit_test_actor_critic'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic_entropy():
    algorithm_name = 'unit_test_actor_critic_entropy'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    spec['name'] = algorithm_name
    spec['agent'][0]['algorithm']['add_entropy'] = True
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic_noGAE():
    algorithm_name = 'unit_test_actor_critic_noGAE'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    spec['name'] = algorithm_name
    spec['agent'][0]['algorithm']['add_GAE'] = False
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic_shared():
    algorithm_name = 'unit_test_actor_critic_shared'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    spec['name'] = algorithm_name
    spec['agent'][0]['net']['type'] = "MLPshared"
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic_multi_epi():
    algorithm_name = 'unit_test_actor_critic_multi_epi'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    spec['name'] = algorithm_name
    spec['agent'][0]['algorithm']['training_frequency'] = 3
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic_batch():
    algorithm_name = 'unit_test_actor_critic_batch'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    spec['name'] = algorithm_name
    spec['agent'][0]['memory']['name'] = "OnPolicyBatchReplay"
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic_recurrent_episodic():
    algorithm_name = 'unit_test_actor_critic_recurrent_episodic'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    spec['name'] = algorithm_name
    spec['agent'][0]['memory']['name'] = "OnPolicyNStepReplay"
    spec['agent'][0]['net']['seq_len'] = 4
    spec['agent'][0]['net']['type'] = "Recurrentseparate"
    spec['agent'][0]['net']['hid_layers'] = [16]
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic_recurrent_batch():
    algorithm_name = 'unit_test_actor_critic_recurrent_batch'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    spec['name'] = algorithm_name
    spec['agent'][0]['memory']['name'] = "OnPolicyNStepBatchReplay"
    spec['agent'][0]['net']['seq_len'] = 4
    spec['agent'][0]['net']['type'] = "Recurrentseparate"
    spec['agent'][0]['net']['hid_layers'] = [16]
    assert generic_algorithm_test(spec, algorithm_name) > 100


@pytest.mark.skip(reason="Crashes on CI")
@flaky(max_runs=3)
def test_actor_critic_recurrent_episodic_shared():
    algorithm_name = 'unit_test_actor_critic_recurrent_episodic_shared'
    spec = spec_util.get('test.json', 'unit_test_actor_critic')
    spec['name'] = algorithm_name
    spec['agent'][0]['memory']['name'] = "OnPolicyNStepReplay"
    spec['agent'][0]['net']['seq_len'] = 4
    spec['agent'][0]['net']['type'] = "Recurrentshared"
    spec['agent'][0]['net']['hid_layers'] = [16]
    assert generic_algorithm_test(spec, algorithm_name) > 100
