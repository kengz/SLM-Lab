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
    # Delete test data folder and trial
    shutil.rmtree(path)
    del trial
    return maxr


# NOTE score-based tests should be done on benchmarking instead of CI. Example is kept below for reference
# @pytest.mark.skip(reason="Crashes on CI")
# @flaky(max_runs=5)
# def test_sarsa():
#     algorithm_name = 'unit_test_sarsa'
#     spec = spec_util.get('test.json', 'unit_test_sarsa')
#     assert generic_algorithm_test(spec, algorithm_name) > 100
