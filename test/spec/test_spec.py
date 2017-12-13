from slm_lab.spec import spec_util
from slm_lab.experiment.control import Trial, Session
import pytest


@pytest.mark.parametrize('spec_file,spec_name', [
    ('base.json', 'base_case'),
    ('base.json', 'base_case_openai'),
    ('base.json', 'multi_body'),
    ('base.json', 'multi_env'),
    ('dqn.json', 'dqn_spec_template'),
    ('dqn.json', 'dqn_test_case'),
])
def test_base(spec_file, spec_name):
    spec = spec_util.get(spec_file, spec_name)
    spec['meta']['train_mode'] = True
    trial = Trial(spec)
    trial_data = trial.run()


def test_build_session(test_session):
    assert isinstance(test_session, Session)
