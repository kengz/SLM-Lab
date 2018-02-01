from slm_lab.experiment.monitor import InfoSpace
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import util, logger
from slm_lab.spec import spec_util
from flaky import flaky
import pytest
import os
import shutil


def generic_algo_test(spec, algo_name):
    '''Need new InfoSpace() per trial otherwise session id doesn't tick correctly'''
    trial = Trial(spec, info_space=InfoSpace())
    trial_data = trial.run()
    folders = [x for x in os.listdir('data/') if x.startswith(algo_name)]
    assert len(folders) == 1
    path = 'data/' + folders[0]
    sess_data = util.read(path + '/' + algo_name + '_t0_s0_session_df.csv')
    rewards = sess_data['0.2'].replace("reward", -1).astype(float)
    print(f'rewards: {rewards}')
    maxr = rewards.max()
    '''Delete test data folder and trial'''
    shutil.rmtree(path)
    del trial, spec
    return maxr
#
#
# @flaky(max_runs=3)
# def test_dqn():
#     algo_name = 'unit_test_dqn'
#     spec = spec_util.get('test.json', 'unit_test_dqn')
#     assert generic_algo_test(spec, algo_name) > 100
#
#
# @flaky(max_runs=3)
# def test_vanilla_dqn():
#     algo_name = 'unit_test_vanilla_dqn'
#     spec = spec_util.get('test.json', 'unit_test_dqn')
#     spec['name'] = algo_name
#     spec['agent'][0]['algorithm']['name'] = "VanillaDQN"
#     assert generic_algo_test(spec, algo_name) > 100
#
#
# @flaky(max_runs=3)
# def test_double_dqn():
#     algo_name = 'unit_test_double_dqn'
#     spec = spec_util.get('test.json', 'unit_test_dqn')
#     spec['name'] = algo_name
#     spec['agent'][0]['algorithm']['name'] = "DoubleDQN"
#     assert generic_algo_test(spec, algo_name) > 100


@flaky(max_runs=3)
def test_reinforce():
    algo_name = 'unit_test_reinforce'
    spec = spec_util.get('test.json', 'unit_test_reinforce')
    assert generic_algo_test(spec, algo_name) > 100


@flaky(max_runs=3)
def unit_test_reinforce_with_entropy():
    algo_name = 'unit_test_reinforce_with_entropy'
    spec = spec_util.get('test.json', 'unit_test_reinforce')
    spec['name'] = algo_name
    spec['agent'][0]['algorithm']['add_entropy'] = True
    assert generic_algo_test(spec, algo_name) > 100


@flaky(max_runs=3)
def test_reinforce_multi_epi():
    algo_name = 'unit_test_reinforce_multi_epi'
    spec = spec_util.get('test.json', 'unit_test_reinforce')
    spec['name'] = algo_name
    spec['agent'][0]['algorithm']['num_epis_to_collect'] = 3
    assert generic_algo_test(spec, algo_name) > 100
