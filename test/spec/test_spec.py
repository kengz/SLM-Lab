from slm_lab import spec


def test_check():
    exp_spec = spec.get('base.json', 'base_case')
    assert spec.check(exp_spec, spec_name='base_case')


def test_check_all():
    assert spec.check_all()


def test_get():
    exp_spec = spec.get('base.json', 'base_case')
    assert exp_spec is not None


def test_resolve_AEB():
    inner_exp_spec = spec.get('base.json', 'general_inner')
    inner_AEB_space = spec.resolve_AEB(inner_exp_spec)
    assert inner_AEB_space == [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]

    outer_exp_spec = spec.get('base.json', 'general_outer')
    outer_AEB_space = spec.resolve_AEB(outer_exp_spec)
    assert outer_AEB_space == [(0, 0, 0),
                               (0, 0, 1),
                               (0, 1, 0),
                               (0, 1, 1),
                               (1, 0, 0),
                               (1, 0, 1),
                               (1, 1, 0),
                               (1, 1, 1)]

    custom_exp_spec = spec.get('base.json', 'general_custom')
    custom_AEB_space = spec.resolve_AEB(custom_exp_spec)
    assert custom_AEB_space == [(0, 0, 0),
                                (0, 1, 0),
                                (0, 1, 1),
                                (0, 1, 2),
                                (0, 1, 3),
                                (0, 1, 4),
                                (0, 1, 5),
                                (0, 1, 6),
                                (0, 1, 7),
                                (0, 1, 8),
                                (0, 1, 9),
                                (0, 1, 10),
                                (0, 1, 11)]
