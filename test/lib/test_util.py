from slm_lab.agent import Agent
from slm_lab.env import Clock
from slm_lab.lib import util
import numpy as np
import os
import pandas as pd
import pydash as ps
import pytest


def test_calc_ts_diff():
    ts1 = '2017_10_17_084739'
    ts2 = '2017_10_17_084740'
    ts_diff = util.calc_ts_diff(ts2, ts1)
    assert ts_diff == '0:00:01'


def test_cast_df(test_df, test_list):
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(util.cast_df(test_df), pd.DataFrame)

    assert not isinstance(test_list, pd.DataFrame)
    assert isinstance(util.cast_df(test_list), pd.DataFrame)


def test_cast_list(test_list, test_str):
    assert ps.is_list(test_list)
    assert ps.is_list(util.cast_list(test_list))

    assert not ps.is_list(test_str)
    assert ps.is_list(util.cast_list(test_str))


@pytest.mark.parametrize('d,res_d', [
    ({'a': 0, 'b': 1}, {'a': 0, 'b': 1}),
    ({'a': None, 'b': 1}, {'b': 1}),
    ({'a': np.nan, 'b': 1}, {'b': 1}),
])
def test_compact_dict(d, res_d):
    assert util.compact_dict(d) == res_d


@pytest.mark.parametrize('arr,arr_len', [
    ([0, 1, 2], 3),
    ([0, 1, 2, None], 3),
    ([0, 1, 2, np.nan], 3),
    ([0, 1, 2, np.nan, np.nan], 3),
    ([0, 1, Clock()], 3),
    ([0, 1, Clock(), np.nan], 3),
])
def test_count_nonan(arr, arr_len):
    assert util.count_nonan(np.array(arr)) == arr_len


def test_dedent(test_multiline_str):
    dedented_string = util.dedent(test_multiline_str)
    assert dedented_string == 'lorem ipsum dolor\nsit amet\n\nconsectetur adipiscing elit'


@pytest.mark.parametrize('d,flat_d', [
    ({'a': 1}, {'a': 1}),
    ({'a': {'b': 1}}, {'a.b': 1}),
    ({
        'level1': {
            'level2': {
                'level3': 0,
                'level3b': 1
            },
            'level2b': {
                'level3': [2, 3]
            }
        }
    }, {'level1.level2.level3': 0,
        'level1.level2.level3b': 1,
        'level1.level2b.level3': [2, 3]}),
    ({
        'level1': {
            'level2': [
                  {'level3': 0},
                  {'level3b': 1}
            ],
            'level2b': {
                'level3': [2, 3]
            }
        }
    }, {'level1.level2.0.level3': 0,
        'level1.level2.1.level3b': 1,
        'level1.level2b.level3': [2, 3]}),
])
def test_flatten_dict(d, flat_d):
    assert util.flatten_dict(d) == flat_d


@pytest.mark.parametrize('arr', [
    ([0, 1, 2]),
    ([0, 1, 2, None]),
    ([0, 1, 2, np.nan]),
    ([0, 1, 2, np.nan, np.nan]),
    ([0, 1, Clock()]),
    ([0, 1, Clock(), np.nan]),
])
def test_filter_nonan(arr):
    arr = np.array(arr)
    assert np.array_equal(util.filter_nonan(arr), arr[:3])


@pytest.mark.parametrize('arr,res', [
    ([0, np.nan], [0]),
    ([[0, np.nan], [1, 2]], [0, 1, 2]),
    ([[[0], [np.nan]], [[1], [2]]], [0, 1, 2]),
])
def test_nanflatten(arr, res):
    arr = np.array(arr)
    res = np.array(res)
    assert np.array_equal(util.nanflatten(arr), res)


@pytest.mark.parametrize('v,isnan', [
    (0, False),
    (1, False),
    (Clock(), False),
    (None, True),
    (np.nan, True),
])
def test_gen_isnan(v, isnan):
    assert util.gen_isnan(v) == isnan


def test_get_env_path():
    assert 'node_modules/slm-env-3dball/build/3dball' in util.get_env_path(
        '3dball')


def test_get_fn_list():
    fn_list = util.get_fn_list(Agent)
    assert 'reset' in fn_list
    assert 'act' in fn_list
    assert 'update' in fn_list


def test_get_ts():
    ts = util.get_ts()
    assert ps.is_string(ts)
    assert util.RE_FILE_TS.match(ts)


def test_is_jupyter():
    assert not util.is_jupyter()


@pytest.mark.parametrize('vec,res', [
    ([1, 1, 1], [False, False, False]),
    ([1, 1, 2], [False, False, True]),
    ([[1, 1], [1, 1], [1, 2]], [False, False, True]),
])
def test_is_outlier(vec, res):
    assert np.array_equal(util.is_outlier(vec), res)


def test_is_sub_dict():
    sub_dict = {'a': 1, 'b': 2}
    super_dict = {'a': 0, 'b': 0, 'c': 0}
    assert util.is_sub_dict(sub_dict, super_dict)
    assert not util.is_sub_dict(super_dict, sub_dict)

    nested_sub_dict = {'a': {'b': 1}, 'c': 2}
    nested_super_dict = {'a': {'b': 0}, 'c': 0, 'd': 0}
    assert util.is_sub_dict(nested_sub_dict, nested_super_dict)
    assert not util.is_sub_dict(nested_super_dict, nested_sub_dict)
    incon_nested_super_dict = {'a': {'b': 0}, 'c': {'d': 0}}
    assert not util.is_sub_dict(nested_sub_dict, incon_nested_super_dict)
    assert not util.is_sub_dict(incon_nested_super_dict, nested_sub_dict)


def test_ndenumerate_nonan():
    arr = np.full((2, 3), np.nan, dtype=object)
    np.fill_diagonal(arr, 1)
    for (a, b), body in util.ndenumerate_nonan(arr):
        assert a == b
        assert body == 1


@pytest.mark.parametrize('v,isall', [
    ([1, 1], True),
    ([True, True], True),
    ([np.nan, 1], True),
    ([0, 1], False),
    ([False, True], False),
    ([np.nan, np.nan], False),
])
def test_nonan_all(v, isall):
    assert util.nonan_all(v) == isall


def test_s_get(test_agent):
    spec = util.s_get(test_agent, 'aeb_space.spec')
    assert ps.is_dict(spec)
    spec = util.s_get(test_agent, 'aeb_space').spec
    assert ps.is_dict(spec)


def test_set_attr():
    class Foo:
        bar = 0
    foo = Foo()
    util.set_attr(foo, {'bar': 1, 'baz': 2})
    assert foo.bar == 1
    assert foo.baz == 2


def test_smart_path():
    rel_path = 'test/lib/test_util.py'
    fake_rel_path = 'test/lib/test_util.py_fake'
    abs_path = os.path.abspath(__file__)
    assert util.smart_path(rel_path) == abs_path
    assert util.smart_path(fake_rel_path) == abs_path + '_fake'
    assert util.smart_path(abs_path) == abs_path
    assert util.smart_path(abs_path, as_dir=True) == os.path.dirname(abs_path)


@pytest.mark.parametrize('filename,dtype', [
    ('test_df.csv', pd.DataFrame),
    ('test_df.xls', pd.DataFrame),
])
def test_write_read_as_df(test_df, filename, dtype):
    data_path = f'test/fixture/lib/util/{filename}'
    util.write(test_df, util.smart_path(data_path))
    assert os.path.exists(data_path)
    data_df = util.read(util.smart_path(data_path))
    assert isinstance(data_df, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_dict.json', dict),
    ('test_dict.yml', dict),
])
def test_write_read_as_plain_dict(test_dict, filename, dtype):
    data_path = f'test/fixture/lib/util/{filename}'
    util.write(test_dict, util.smart_path(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.smart_path(data_path))
    assert isinstance(data_dict, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_list.json', list),
])
def test_write_read_as_plain_list(test_list, filename, dtype):
    data_path = f'test/fixture/lib/util/{filename}'
    util.write(test_list, util.smart_path(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.smart_path(data_path))
    assert isinstance(data_dict, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_str.txt', str),
])
def test_write_read_as_plain_list(test_str, filename, dtype):
    data_path = f'test/fixture/lib/util/{filename}'
    util.write(test_str, util.smart_path(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.smart_path(data_path))
    assert isinstance(data_dict, dtype)


def test_read_file_not_found():
    fake_rel_path = 'test/lib/test_util.py_fake'
    with pytest.raises(FileNotFoundError) as excinfo:
        util.read(fake_rel_path)
