from slm_lab import spec


def test_check():
    exp_spec = spec.get('default.json', 'base_case')
    assert spec.check(exp_spec, spec_name='base_case')


def test_check_all():
    assert spec.check_all()


def test_get():
    exp_spec = spec.get('default.json', 'base_case')
    assert exp_spec is not None
