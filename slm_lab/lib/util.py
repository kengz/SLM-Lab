from datetime import datetime
from slm_lab import ROOT_DIR
import collections
import json
import numpy as np
import os
import pandas as pd
import pydash as _
import regex as re
import ujson
import yaml

DF_FILE_EXT = ['.csv', '.xlsx', '.xls']
FILE_TS_FORMAT = '%Y_%m_%d_%H%M%S'
RE_FILE_TS = re.compile(r'(\d{4}_\d{2}_\d{2}_\d{6})')
RE_INDENT = re.compile('(^\n)|(?!\n)\s{2,}|(\n\s+)$')
SPACE_PATH = ['agent', 'agent_space', 'aeb_space', 'env_space', 'env']


class LabJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)


def calc_timestamp_diff(ts2, ts1):
    '''
    Calculate the time from timestamps ts1 to ts2
    @param {str} ts2 Later timestamp in the FILE_TS_FORMAT
    @param {str} ts1 Earlier timestamp in the FILE_TS_FORMAT
    @returns {str} delta_t in %H:%M:%S format
    @example

    ts1 = '2017_10_17_084739'
    ts2 = '2017_10_17_084740'
    ts_diff = util.calc_timestamp_diff(ts2, ts1)
    # => '0:00:01'
    '''
    delta_t = datetime.strptime(
        ts2, FILE_TS_FORMAT) - datetime.strptime(
        ts1, FILE_TS_FORMAT)
    return str(delta_t)


def cast_df(val):
    '''missing pydash method to cast value as DataFrame'''
    if isinstance(val, pd.DataFrame):
        return val
    return pd.DataFrame(val)


def cast_list(val):
    '''missing pydash method to cast value as list'''
    if _.is_list(val):
        return val
    else:
        return [val]


def count_nonnan(arr):
    try:
        return np.count_nonzero(~np.isnan(arr))
    except Exception:
        return len(flatten_nonan(arr))


def dedent(string):
    '''Method to dedent the broken python multiline string'''
    return RE_INDENT.sub('', string)


def flatten_dict(d, parent_key='', sep='.'):
    '''Missing pydash method to flatten dict'''
    items = []
    for k, v in d.items():
        if parent_key:
            new_key = parent_key + sep + k
        else:
            new_key = k

        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_nonan(arr):
    flat_arr = arr.flatten()
    try:
        return flat_arr[~np.isnan(flat_arr)]
    except Exception:
        return np.array([v for v in flat_arr if v is not np.nan])


def get_aeb_shape(aeb_list):
    return np.amax(aeb_list, axis=0) + 1


def get_class_name(obj, lower=False):
    '''Get the class name of an object'''
    class_name = obj.__class__.__name__
    if lower:
        class_name = class_name.lower()
    return class_name


def get_class_attr(obj):
    '''Get the class attr of an object as dict'''
    attr_dict = {}
    for k, v in obj.__dict__.items():
        if hasattr(v, '__dict__') or _.is_tuple(v):
            val = str(v)
        else:
            val = v
        attr_dict[k] = val
    return attr_dict


def get_env_path(env_name):
    '''Get the path to Unity env binaries distributed via npm'''
    env_path = smart_path(f'node_modules/slm-env-{env_name}/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(
        env_dir), f'Missing {env_path}. See README to install from yarn.'
    return env_path


def get_file_ext(data_path):
    return os.path.splitext(data_path)[-1]


def get_fn_list(Cls):
    '''
    Get the callable, non-private functions of a class
    @returns {[*str]} A list of strings of fn names
    '''
    fn_list = _.filter_(
        dir(Cls),
        lambda fn: not fn.endswith('__') and callable(getattr(Cls, fn)))
    return fn_list


def get_timestamp(pattern=FILE_TS_FORMAT):
    '''
    Get current timestamp, defaults to format used for filename
    @param {str} pattern To format the timestamp
    @returns {str} timestamp
    @example

    util.get_timestamp()
    # => '2017_10_17_084739'
    '''
    timestamp_obj = datetime.now()
    timestamp = timestamp_obj.strftime(pattern)
    assert RE_FILE_TS.search(timestamp)
    return timestamp


def is_jupyter():
    '''Check if process is in Jupyter kernel'''
    try:
        get_ipython().config
        return True
    except NameError:
        return False
    return False


def is_sub_dict(sub_dict, super_dict):
    '''
    Check if sub_dict is a congruent subset of super_dict
    @param {dict} sub_dict The sub dictionary
    @param {dict} super_dict The super dictionary
    @returns {bool}
    @example

    sub_dict = {'a': 1, 'b': 2}
    super_dict = {'a': 0, 'b': 0, 'c': 0}
    util.is_sub_dict(sub_dict, super_dict)
    # => True

    nested_sub_dict = {'a': {'b': 1}, 'c': 2}
    nested_super_dict = {'a': {'b': 0}, 'c': 0, 'd': 0}
    util.is_sub_dict(nested_sub_dict, nested_super_dict)
    # => True

    incon_nested_super_dict = {'a': {'b': 0}, 'c': {'d': 0}}
    util.is_sub_dict(nested_sub_dict, incon_nested_super_dict)
    # => False
    '''
    for sub_k, sub_v in sub_dict.items():
        if sub_k not in super_dict:
            return False
        super_v = super_dict[sub_k]
        if type(sub_v) != type(super_v):
            return False
        if _.is_dict(sub_v):
            if not is_sub_dict(sub_v, super_v):
                return False
        else:
            if sub_k not in super_dict:
                return False
    return True


def read(data_path):
    '''
    Universal data reading method with smart data parsing
    - {.csv, .xlsx, .xls} to DataFrame
    - {.json} to dict, list
    - {.yml} to dict
    - {*} to str
    - TODO {.h5} to model weights
    - TODO {db-query} to dict, DataFrame
    @param {str} data_path The data path to read from
    @returns {data} The read data in sensible format
    @example

    data_df = util.read('test/fixture/common/util/test_df.csv')
    data_df = util.read('test/fixture/common/util/test_df.xls')
    data_df = util.read('test/fixture/common/util/test_df.xlsx')
    # => <DataFrame>

    data_dict = util.read('test/fixture/common/util/test_dict.json')
    data_dict = util.read('test/fixture/common/util/test_dict.yml')
    # => <dict>

    data_list = util.read('test/fixture/common/util/test_list.json')
    # => <list>

    data_str = util.read('test/fixture/common/util/test_str.txt')
    # => <str>
    '''
    data_path = smart_path(data_path)
    try:
        assert os.path.isfile(data_path)
    except AssertionError:
        raise FileNotFoundError(data_path)
    ext = get_file_ext(data_path)
    if ext in DF_FILE_EXT:
        data = read_as_df(data_path)
    else:
        data = read_as_plain(data_path)
    return data


def read_as_df(data_path):
    '''Submethod to read data as DataFrame'''
    ext = get_file_ext(data_path)
    if ext in ['.xlsx', 'xls']:
        data = pd.read_excel(data_path)
    else:  # .csv
        data = pd.read_csv(data_path)
    return data


def read_as_plain(data_path):
    '''Submethod to read data as plain type'''
    open_file = open(data_path, 'r')
    ext = get_file_ext(data_path)
    if ext == '.json':
        data = ujson.load(open_file)
    elif ext == '.yml':
        data = yaml.load(open_file)
    else:
        data = open_file.read()
    open_file.close()
    return data


def s_get(cls, attr_path):
    '''
    Method to get attribute across space via inferring agent <-> env paths.
    @example
    self.agent.agent_space.aeb_space.clock
    # equivalently
    util.s_get(self, 'aeb_space.clock')
    '''
    from_class_name = get_class_name(cls, lower=True)
    from_idx = _.find_index(
        SPACE_PATH, lambda s: from_class_name in (s, s.replace('_', '')))
    from_idx = max(from_idx, 0)
    attr_path = attr_path.split('.')
    to_idx = SPACE_PATH.index(attr_path[0])
    assert -1 not in (from_idx, to_idx)
    if from_idx < to_idx:
        path_link = SPACE_PATH[from_idx: to_idx]
    else:
        path_link = _.reverse(SPACE_PATH[to_idx: from_idx])

    res = cls
    for attr in path_link + attr_path:
        if not (get_class_name(res, lower=True) in (attr, attr.replace('_', ''))):
            res = getattr(res, attr)
    return res


def self_desc(cls):
    '''Method to get self description, used at init.'''
    desc_list = [f' - {k} = {v}' for k, v in get_class_attr(cls).items()]
    desc_list.insert(0, f'{get_class_name(cls)}:')
    desc = '\n'.join(desc_list)
    return desc


def set_attr(obj, attr_dict):
    '''Set attribute of an object from a dict'''
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
    return obj


def smart_path(data_path, as_dir=False):
    '''
    Resolve data_path into abspath with fallback to join from ROOT_DIR
    @param {str} data_path The input data path to resolve
    @param {bool} as_dir Whether to return as dirname
    @returns {str} The normalized absolute data_path
    @example

    util.smart_path('slm_lab/lib')
    # => '/Users/ANON/Documents/slm_lab/slm_lab/lib'

    util.smart_path('/tmp')
    # => '/tmp'
    '''
    if not os.path.isabs(data_path):
        abs_path = os.path.abspath(data_path)
        if os.path.exists(abs_path):
            data_path = abs_path
        else:
            data_path = os.path.join(ROOT_DIR, data_path)
    if as_dir:
        data_path = os.path.dirname(data_path)
    return os.path.normpath(data_path)


def to_json(d, indent=2):
    '''Shorthand method for stringify JSON with indent'''
    return json.dumps(d, indent=indent, cls=LabJsonEncoder)


def to_tuple_list(l):
    '''Returns a copy of the list with its elements as tuples'''
    return [tuple(row) for row in l]


def write(data, data_path):
    '''
    Universal data writing method with smart data parsing
    - {.csv, .xlsx, .xls} from DataFrame
    - {.json} from dict, list
    - {.yml} from dict
    - {*} from str(*)
    - TODO {.h5} from model weights
    - TODO {db-query} from dict, DataFrame
    @param {*} data The data to write
    @param {str} data_path The data path to write to
    @returns {data_path} The data path written to
    @example

    data_path = util.write(data_df, 'test/fixture/common/util/test_df.csv')
    data_path = util.write(data_df, 'test/fixture/common/util/test_df.xls')
    data_path = util.write(data_df, 'test/fixture/common/util/test_df.xlsx')

    data_path = util.write(data_dict, 'test/fixture/common/util/test_dict.json')
    data_path = util.write(data_dict, 'test/fixture/common/util/test_dict.yml')

    data_path = util.write(data_list, 'test/fixture/common/util/test_list.json')

    data_path = util.write(data_str, 'test/fixture/common/util/test_str.txt')
    '''
    data_path = smart_path(data_path)
    data_dir = os.path.dirname(data_path)
    os.makedirs(data_dir, exist_ok=True)
    ext = get_file_ext(data_path)
    if ext in DF_FILE_EXT:
        write_as_df(data, data_path)
    else:
        write_as_plain(data, data_path)
    return data_path


def write_as_df(data, data_path):
    '''Submethod to write data as DataFrame'''
    df = cast_df(data)
    ext = get_file_ext(data_path)
    if ext in ['.xlsx', 'xls']:
        writer = pd.ExcelWriter(data_path)
        df.to_excel(writer)
        writer.save()
        writer.close()
    else:  # .csv
        df.to_csv(data_path, index=False)
    return data_path


def write_as_plain(data, data_path):
    '''Submethod to write data as plain type'''
    open_file = open(data_path, 'w')
    ext = get_file_ext(data_path)
    if ext == '.json':
        json.dump(data, open_file)
    elif ext == '.yml':
        yaml.dump(data, open_file)
    else:
        open_file.write(str(data))
    open_file.close()
    return data_path
