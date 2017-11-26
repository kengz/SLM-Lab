import numpy as np
from slm_lab.spec import spec_util


def test_check():
    spec = spec_util.get('base.json', 'base_case')
    assert spec_util.check(spec, spec_name='base_case')


def test_check_all():
    assert spec_util.check_all()


def test_get():
    spec = spec_util.get('base.json', 'base_case')
    assert spec is not None


def test_resolve_aeb():
    inner_spec = spec_util.get('base.json', 'general_inner')
    inner_aeb_coor_list = spec_util.resolve_aeb(inner_spec)
    assert inner_aeb_coor_list == [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]

    outer_spec = spec_util.get('base.json', 'general_outer')
    outer_aeb_coor_list = spec_util.resolve_aeb(outer_spec)
    assert outer_aeb_coor_list == [(0, 0, 0),
                                   (0, 0, 1),
                                   (0, 1, 0),
                                   (0, 1, 1),
                                   (1, 0, 0),
                                   (1, 0, 1),
                                   (1, 1, 0),
                                   (1, 1, 1)]

    custom_spec = spec_util.get('base.json', 'general_custom')
    custom_aeb_coor_list = spec_util.resolve_aeb(custom_spec)
    assert custom_aeb_coor_list == [(0, 0, 0),
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
