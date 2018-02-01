from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import util
from slm_lab.spec import spec_util
from flaky import flaky
import pytest
import os
import shutil


def generic_algo_test(spec, algo_name):
    trial_data = Trial(spec).run()
    folders = [x for x in os.listdir('data/') if x.startswith(algo_name)]
    assert len(folders) == 1
    path = 'data/' + folders[0]
    sess_data = util.read(path + '/' + algo_name + '_t0_s0_session_df.csv')
    rewards = sess_data['0.2'].replace("reward", -1).astype(float)
    print(f'rewards: {rewards}')
    maxr = rewards.max()
    '''Delete test data folder'''
    shutil.rmtree(path)
    return maxr


def get_dqn_spec():
    return spec_util.get('test.json', 'unit_test_dqn')


# @flaky(max_runs=3)
# def test_dqn():
#     algo_name = 'unit_test_dqn'
#     spec = get_dqn_spec()
#     assert generic_algo_test(spec, algo_name) > 100


# @flaky(max_runs=3)
def test_vanilla_dqn():
    algo_name = 'unit_test_vanilla_dqn'
    spec = get_dqn_spec()
    spec['name'] = algo_name
    spec['agent'][0]['algorithm']['name'] = "VanillaDQN"
    print(spec)
    assert True is False
    assert generic_algo_test(spec, algo_name) > 100


# @flaky(max_runs=3)
# def test_double_dqn():
#     algo_name = 'unit_test_double_dqn'
#     spec = get_dqn_spec()
#     spec['name'] = algo_name
#     spec['agent'][0]['algorithm']['name'] = "DoubleDQN"
#     assert generic_algo_test(spec, algo_name) > 100
