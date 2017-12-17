from slm_lab.experiment.control import Session, Trial
import os
import pandas as pd
import pytest


# TODO test control steps in detail when complete

@pytest.mark.skip(reason='TODO in dev')
def test_session(test_spec):
    session = Session(test_spec)
    session_data = session.run()
    # TODO session data checker method
    assert isinstance(session_data, pd.DataFrame)


@pytest.mark.skip(reason='TODO in dev')
def test_trial(test_spec):
    trial = Trial(test_spec)
    trial_data = trial.run()
    # TODO trial data checker method
    assert isinstance(trial_data, pd.DataFrame)


def test_experiment():
    return
