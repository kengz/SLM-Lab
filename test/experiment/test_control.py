# from slm_lab.experiment.control import Session, Trial, Experiment
# from slm_lab.spec import spec_util
# import pandas as pd
# import pytest
#
#
# def test_session(test_spec):
#     session = Session(test_spec)
#     session_data = session.run()
#     # TODO session data checker method
#     assert isinstance(session_data, pd.DataFrame)
#
#
# def test_trial(test_spec):
#     trial = Trial(test_spec)
#     trial_data = trial.run()
#     # TODO trial data checker method
#     assert isinstance(trial_data, pd.DataFrame)
#
#
# @pytest.mark.skip(reason='TODO broken by pytorch in session parallelization')
# def test_trial_demo():
#     spec = spec_util.get('reinforce.json', 'reinforce_cartpole')
#     spec['meta']['train_mode'] = True
#     spec['meta']['max_episode'] = 20
#     Trial(spec).run()
#
#
# @pytest.mark.skip(reason='TODO tmp search removal')
# def test_experiment(test_spec):
#     experiment = Experiment(test_spec)
#     experiment_data = experiment.run()
#     # TODO experiment data checker method
#     assert isinstance(experiment_data, pd.DataFrame)
