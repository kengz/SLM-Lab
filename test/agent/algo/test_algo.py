from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import util
from slm_lab.spec import spec_util
from flaky import flaky
import pytest
import os
import shutil


@flaky(max_runs=5)
def test_algo(test_algorithms):
    algo_name = test_algorithms
    spec = spec_util.get('test.json', algo_name)
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
    assert maxr > 100
