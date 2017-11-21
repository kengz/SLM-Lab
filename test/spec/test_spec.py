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
    inner_coor_list = spec.resolve_AEB(inner_exp_spec)
    assert inner_coor_list == [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]

    outer_exp_spec = spec.get('base.json', 'general_outer')
    outer_coor_list = spec.resolve_AEB(outer_exp_spec)
    assert outer_coor_list == [(0, 0, 0),
                               (0, 0, 1),
                               (0, 1, 0),
                               (0, 1, 1),
                               (1, 0, 0),
                               (1, 0, 1),
                               (1, 1, 0),
                               (1, 1, 1)]
