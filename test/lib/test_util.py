from slm_lab.agent import Agent
from slm_lab.lib import util
import os
import pandas as pd
import pytest


def test_calc_timestamp_diff():
    ts1 = '2017_10_17_084739'
    ts2 = '2017_10_17_084740'
    ts_diff = util.calc_timestamp_diff(ts2, ts1)
    assert ts_diff == '0:00:01'


def test_cast_df(test_df, test_list):
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(util.cast_df(test_df), pd.DataFrame)

    assert not isinstance(test_list, pd.DataFrame)
    assert isinstance(util.cast_df(test_list), pd.DataFrame)


def test_cast_list(test_list, test_str):
    assert isinstance(test_list, list)
    assert isinstance(util.cast_list(test_list), list)

    assert not isinstance(test_str, list)
    assert isinstance(util.cast_list(test_str), list)


def test_dedent(test_multiline_str):
    dedented_string = util.dedent(test_multiline_str)
    assert dedented_string == 'lorem ipsum dolor\nsit amet\n\nconsectetur adipiscing elit'


def test_flatten_dict(test_dict):
    assert util.flatten_dict(test_dict) == test_dict
    assert util.flatten_dict({'a': {'b': 1}}) == {'a.b': 1}
    assert util.flatten_dict({'a': {'b': 1}}) == {'a.b': 1}
    assert util.flatten_dict({'a': {'b': 1}}, sep='_') == {'a_b': 1}


def test_get_env_path():
    assert 'node_modules/slm-env-3dball/build/3dball' in util.get_env_path(
        '3dball')


def test_get_fn_list():
    fn_list = util.get_fn_list(Agent)
    assert 'reset' in fn_list
    assert 'act' in fn_list
    assert 'update' in fn_list


def test_get_timestamp():
    timestamp = util.get_timestamp()
    assert isinstance(timestamp, str)
    assert util.RE_FILE_TS.match(timestamp)


def test_is_jupyter():
    assert not util.is_jupyter()


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
    ('test_df.xlsx', pd.DataFrame),
])
def test_write_read_as_df(test_df, filename, dtype):
    data_path = f'test/fixture/common/util/{filename}'
    util.write(test_df, util.smart_path(data_path))
    assert os.path.exists(data_path)
    data_df = util.read(util.smart_path(data_path))
    assert isinstance(data_df, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_dict.json', dict),
    ('test_dict.yml', dict),
])
def test_write_read_as_plain_dict(test_dict, filename, dtype):
    data_path = f'test/fixture/common/util/{filename}'
    util.write(test_dict, util.smart_path(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.smart_path(data_path))
    assert isinstance(data_dict, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_list.json', list),
])
def test_write_read_as_plain_list(test_list, filename, dtype):
    data_path = f'test/fixture/common/util/{filename}'
    util.write(test_list, util.smart_path(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.smart_path(data_path))
    assert isinstance(data_dict, dtype)


@pytest.mark.parametrize('filename,dtype', [
    ('test_str.txt', str),
])
def test_write_read_as_plain_list(test_str, filename, dtype):
    data_path = f'test/fixture/common/util/{filename}'
    util.write(test_str, util.smart_path(data_path))
    assert os.path.exists(data_path)
    data_dict = util.read(util.smart_path(data_path))
    assert isinstance(data_dict, dtype)


def test_read_file_not_found():
    fake_rel_path = 'test/lib/test_util.py_fake'
    with pytest.raises(FileNotFoundError) as excinfo:
        util.read(fake_rel_path)
