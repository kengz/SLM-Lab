from copy import deepcopy
from flaky import flaky
from slm_lab.experiment import analysis
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.spec import spec_util
import pandas as pd
import pytest


def test_session(test_spec):
    spec_util.tick(test_spec, 'trial')
    spec_util.tick(test_spec, 'session')
    analysis.save_spec(test_spec, unit='trial')
    session = Session(test_spec)
    session_data = session.run()
    assert isinstance(session_data, pd.DataFrame)


def test_session_total_t(test_spec):
    spec_util.tick(test_spec, 'trial')
    spec_util.tick(test_spec, 'session')
    analysis.save_spec(test_spec, unit='trial')
    spec = deepcopy(test_spec)
    env_spec = spec['env'][0]
    env_spec['max_tick'] = 30
    spec['meta']['max_tick_unit'] = 'total_t'
    session = Session(spec)
    assert session.env.clock.max_tick_unit == 'total_t'
    session_data = session.run()
    assert isinstance(session_data, pd.DataFrame)


def test_trial(test_spec):
    spec_util.tick(test_spec, 'trial')
    analysis.save_spec(test_spec, unit='trial')
    trial = Trial(test_spec)
    trial_data = trial.run()
    assert isinstance(trial_data, pd.DataFrame)


def test_trial_demo():
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    analysis.save_spec(spec, unit='experiment')
    spec = spec_util.override_test_spec(spec)
    spec_util.tick(spec, 'trial')
    trial_data = Trial(spec).run()
    assert isinstance(trial_data, pd.DataFrame)


@pytest.mark.skip(reason="Unstable")
@flaky
def test_demo_performance():
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    analysis.save_spec(spec, unit='experiment')
    for env_spec in spec['env']:
        env_spec['max_tick'] = 2000
    spec_util.tick(spec, 'trial')
    trial = Trial(spec)
    spec_util.tick(spec, 'session')
    session = Session(spec)
    session.run()
    last_reward = session.agent.body.train_df.iloc[-1]['reward']
    assert last_reward > 50, f'last_reward is too low: {last_reward}'


def test_experiment():
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    analysis.save_spec(spec, unit='experiment')
    spec = spec_util.override_test_spec(spec)
    spec_util.tick(spec, 'experiment')
    experiment_data = Experiment(spec).run()
    assert isinstance(experiment_data, pd.DataFrame)
