# from slm_lab.spec import spec_util
# from slm_lab.experiment.control import Trial, Session
# import pytest
#
#
# @pytest.mark.parametrize('spec_file,spec_name', [
#     ('base.json', 'base_case'),
#     ('base.json', 'base_case_openai'),
#     ('base.json', 'multi_body'),
#     ('base.json', 'multi_env'),
#     # ('base.json', 'multi_agent'),
#     # ('base.json', 'multi_agent_multi_env'),
# ])
# def test_base(spec_file, spec_name):
#     spec = spec_util.get(spec_file, spec_name)
#     spec['meta']['train_mode'] = True
#     spec['meta']['max_episode'] = 20
#     trial = Trial(spec)
#     trial_data = trial.run()
#
#
# @pytest.mark.skip(reason='TODO broken by pytorch in session parallelization')
# @pytest.mark.parametrize('spec_file,spec_name', [
#     ('dqn.json', 'dqn_spec_template'),
#     ('dqn.json', 'dqn_test_case'),
#     ('reinforce.json', 'reinforce_cartpole'),
#     ('actor_critic.json', 'actor_critic_cartpole'),
# ])
# def test_algo(spec_file, spec_name):
#     spec = spec_util.get(spec_file, spec_name)
#     spec['meta']['train_mode'] = True
#     spec['meta']['max_episode'] = 20
#     trial = Trial(spec)
#     trial_data = trial.run()
