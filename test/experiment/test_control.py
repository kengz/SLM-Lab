from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import util
from slm_lab.spec import spec_util
import pandas as pd
import pytest


def test_session(test_spec):
    session = Session(test_spec)
    session_data = session.run()
    # TODO session data checker method
    assert isinstance(session_data, pd.DataFrame)


@pytest.mark.skip(reason='TODO broken by pytest cov https://circleci.com/gh/kengz/SLM-Lab/997')
def test_trial(test_spec):
    trial = Trial(test_spec)
    trial_data = trial.run()
    # TODO trial data checker method
    assert isinstance(trial_data, pd.DataFrame)


@pytest.mark.skip(reason='TODO broken by pytest cov https://circleci.com/gh/kengz/SLM-Lab/997')
def test_trial_demo():
    spec = spec_util.get('reinforce.json', 'reinforce_cartpole')
    util.override_test_spec(spec)
    trial_data = Trial(spec).run()
    assert isinstance(trial_data, pd.DataFrame)


@pytest.mark.skip(reason='TODO tmp search removal')
def test_experiment(test_spec):
    experiment = Experiment(test_spec)
    experiment_data = experiment.run()
    # TODO experiment data checker method
    assert isinstance(experiment_data, pd.DataFrame)
