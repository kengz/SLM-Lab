import numpy as np
import pytest
from slm_lab.spec import spec_util


def test_check():
    spec = spec_util.get('base.json', 'base_case')
    assert spec_util.check(spec, spec_name='base_case')


def test_check_all():
    assert spec_util.check_all()


def test_get():
    spec = spec_util.get('base.json', 'base_case')
    assert spec is not None


@pytest.mark.parametrize('spec_name,aeb_coor_list', [
    ('general_inner', [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]),
    ('general_outer', [(0, 0, 0),
                       (0, 0, 1),
                       (0, 1, 0),
                       (0, 1, 1),
                       (1, 0, 0),
                       (1, 0, 1),
                       (1, 1, 0),
                       (1, 1, 1)]),
    ('general_custom', [(0, 0, 0),
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
                        (0, 1, 11)]),
])
def test_resolve_aeb(spec_name, aeb_coor_list):
    spec = spec_util.get('base.json', spec_name)
    resolved_aeb_coor_list = spec_util.resolve_aeb(spec)
    assert resolved_aeb_coor_list == aeb_coor_list
