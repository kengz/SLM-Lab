from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import util
from slm_lab.spec import spec_util
import pandas as pd
import pytest


def test_session(test_spec):
    session = Session(test_spec)
    session_data = session.run()
    assert isinstance(session_data, pd.DataFrame)


def test_trial(test_spec):
    trial = Trial(test_spec)
    trial_data = trial.run()
    assert isinstance(trial_data, pd.DataFrame)


def test_trial_demo():
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    spec = util.override_test_spec(spec)
    trial_data = Trial(spec).run()
    assert isinstance(trial_data, pd.DataFrame)


def test_experiment(test_spec):
    experiment = Experiment(test_spec)
    experiment_data = experiment.run()
    assert isinstance(experiment_data, pd.DataFrame)
