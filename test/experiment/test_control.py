import pandas as pd
import pytest
from slm_lab import spec
from slm_lab.experiment.control import Monitor, Session, Trial


# TODO test control steps in detail when complete

def test_session(test_exp_spec):
    monitor = Monitor(test_exp_spec)
    session = Session(test_exp_spec, monitor)
    session_data = session.run()
    # TODO session data checker method
    assert isinstance(session_data, pd.DataFrame)


def test_trial(test_exp_spec):
    monitor = Monitor(test_exp_spec)
    trial = Trial(test_exp_spec, monitor)
    trial_data = trial.run()
    # TODO trial data checker method
    assert isinstance(trial_data, pd.DataFrame)


def test_experiment():
    return
