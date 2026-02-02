from slm_lab.spec import spec_util


def test_check():
    spec = spec_util.get('experimental/misc/base.json', 'base_case_gymnasium')
    assert spec_util.check(spec)


def test_check_all():
    assert spec_util.check_all()


def test_get():
    spec = spec_util.get('experimental/misc/base.json', 'base_case_gymnasium')
    assert spec is not None
