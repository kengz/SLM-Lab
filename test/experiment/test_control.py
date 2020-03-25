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
    spec_util.save(test_spec, unit='trial')
    session = Session(test_spec)
    session_metrics = session.run()
    assert isinstance(session_metrics, dict)


def test_trial(test_spec):
    spec_util.tick(test_spec, 'trial')
    spec_util.save(test_spec, unit='trial')
    trial = Trial(test_spec)
    trial_metrics = trial.run()
    assert isinstance(trial_metrics, dict)


def test_trial_demo():
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    spec_util.save(spec, unit='experiment')
    spec = spec_util.override_test_spec(spec)
    spec_util.tick(spec, 'trial')
    trial_metrics = Trial(spec).run()
    assert isinstance(trial_metrics, dict)


@pytest.mark.skip(reason="Unstable")
@flaky
def test_demo_performance():
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    spec_util.save(spec, unit='experiment')
    for env_spec in spec['env']:
        env_spec['max_frame'] = 2000
    spec_util.tick(spec, 'trial')
    trial = Trial(spec)
    spec_util.tick(spec, 'session')
    session = Session(spec)
    session.run()
    last_reward = session.world.bodies.train_df.iloc[-1]['total_reward']
    assert last_reward > 50, f'last_reward is too low: {last_reward}'


@pytest.mark.skip(reason="Cant run on CI")
def test_experiment():
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    spec_util.save(spec, unit='experiment')
    spec = spec_util.override_test_spec(spec)
    spec_util.tick(spec, 'experiment')
    experiment_df = Experiment(spec).run()
    # TODO test this manually and implement a better support of experiment analysis with multi agents
    # print("HEAD")
    # print(type(experiment_df))
    # print(len(experiment_df))
    # print(experiment_df.columns)
    # print(experiment_df.head())
    # for c in experiment_df.columns:
    #     print(c, experiment_df[c].tolist())
    # assert 0
    assert isinstance(experiment_df, pd.DataFrame)
