from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import util
from slm_lab.spec import spec_util
import pandas as pd
import pytest


def test_session(test_spec, test_info_space):
    test_info_space.tick('session')
    session = Session(test_spec, test_info_space)
    session_data = session.run()
    assert isinstance(session_data, pd.DataFrame)


def test_trial(test_spec, test_info_space):
    test_info_space.tick('trial')
    trial = Trial(test_spec, test_info_space)
    trial_data = trial.run()
    assert isinstance(trial_data, pd.DataFrame)


def test_trial_demo(test_info_space):
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    spec = util.override_test_spec(spec)
    spec['env'][0]['save_frequency'] = 1
    test_info_space.tick('trial')
    trial_data = Trial(spec, test_info_space).run()
    assert isinstance(trial_data, pd.DataFrame)


def test_experiment(test_info_space):
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    spec = util.override_test_spec(spec)
    test_info_space.tick('experiment')
    experiment_data = Experiment(spec, test_info_space).run()
    assert isinstance(experiment_data, pd.DataFrame)
