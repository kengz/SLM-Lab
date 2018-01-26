from datetime import datetime
from slm_lab import ROOT_DIR
from torch.autograd import Variable
import collections
import colorlover as cl
import json
import math
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pydash as _
import regex as re
import torch
import ujson
import yaml

CPU_NUM = mp.cpu_count()
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


def calc_ts_diff(ts2, ts1):
    '''
    Calculate the time from tss ts1 to ts2
    @param {str} ts2 Later ts in the FILE_TS_FORMAT
    @param {str} ts1 Earlier ts in the FILE_TS_FORMAT
    @returns {str} delta_t in %H:%M:%S format
    @example

    ts1 = '2017_10_17_084739'
    ts2 = '2017_10_17_084740'
    ts_diff = util.calc_ts_diff(ts2, ts1)
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


def compact_dict(d):
    '''Return dict without None or np.nan values'''
    return {k: v for k, v in d.items() if not gen_isnan(v)}


def concat_dict(d_list):
    '''Concatenate all the dicts by their array values'''
    cat_dict = {}
    for k in d_list[0]:
        arr = np.concatenate([d[k] for d in d_list])
        cat_dict[k] = arr
    return cat_dict


def count_nonan(arr):
    try:
        return np.count_nonzero(~np.isnan(arr))
    except Exception:
        return len(filter_nonan(arr))


def dedent(string):
    '''Method to dedent the broken python multiline string'''
    return RE_INDENT.sub('', string)


def flatten_dict(obj, delim='.'):
    '''Missing pydash method to flatten dict'''
    nobj = {}
    for key, val in obj.items():
        if _.is_dict(val) and not _.is_empty(val):
            strip = flatten_dict(val, delim)
            for k, v in strip.items():
                nobj[key + delim + k] = v
        elif _.is_list(val) and not _.is_empty(val) and _.is_dict(val[0]):
            for idx, v in enumerate(val):
                nobj[key + delim + str(idx)] = v
                if _.is_object(v):
                    nobj = flatten_dict(nobj, delim)
        else:
            nobj[key] = val
    return nobj


def filter_nonan(arr):
    '''Filter to np array with no nan'''
    try:
        return arr[~np.isnan(arr)]
    except Exception:
        mixed_type = []
        for v in arr:
            if not gen_isnan(v):
                mixed_type.append(v)
        return np.array(mixed_type, dtype=arr.dtype)


def nanflatten(arr):
    '''Flatten np array while ignoring nan, like np.nansum etc.'''
    flat_arr = arr.reshape(-1)
    return filter_nonan(flat_arr)


def flatten_once(arr):
    '''Flatten np array only once instead if all the way by flatten()'''
    return arr.reshape(-1, *arr.shape[2:])


def gen_isnan(v):
    '''Check isnan for general type (np.isnan is only operable on np type)'''
    try:
        return np.isnan(v).all()
    except Exception:
        return v is None


def get_df_aeb_list(session_df):
    '''Get the aeb list for session_df for iterating.'''
    aeb_list = sorted(_.uniq(
        [(a, e, b) for a, e, b, col in session_df.columns.tolist()]))
    return aeb_list


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


def get_fn_list(a_cls):
    '''
    Get the callable, non-private functions of a class
    @returns {[*str]} A list of strings of fn names
    '''
    fn_list = _.filter_(
        dir(a_cls),
        lambda fn: not fn.endswith('__') and callable(getattr(a_cls, fn)))
    return fn_list


def get_ts(pattern=FILE_TS_FORMAT):
    '''
    Get current ts, defaults to format used for filename
    @param {str} pattern To format the ts
    @returns {str} ts
    @example

    util.get_ts()
    # => '2017_10_17_084739'
    '''
    ts_obj = datetime.now()
    ts = ts_obj.strftime(pattern)
    assert RE_FILE_TS.search(ts)
    return ts


def guard_data_a(cls, data_a, data_name):
    '''Guard data_a in case if it scalar, create a data_a and fill.'''
    if np.isscalar(data_a):
        new_data_a, = s_get(cls, 'aeb_space').init_data_s([data_name], a=cls.a)
        for (e, b), body in ndenumerate_nonan(cls.body_a):
            new_data_a[(e, b)] = data_a
        data_a = new_data_a
    return data_a


def interp(scl, r):
    '''
    Replacement for colorlover.interp
    Interpolate a color scale "scl" to a new one with length "r"
        Fun usage in IPython notebook:
        HTML( to_html( to_hsl( interp( cl.scales['11']['qual']['Paired'], 5000 ) ) ) )
    '''
    c = []
    SCL_FI = len(scl) - 1  # final index of color scale
    # garyfeng:
    # the following line is buggy.
    # r = [x * 0.1 for x in range(r)] if isinstance( r, int ) else r
    r = [x * 1.0 * SCL_FI / r for x in range(r)] if isinstance(r, int) else r
    # end garyfeng

    scl = cl.to_numeric(scl)

    def interp3(fraction, start, end):
        ''' Interpolate between values of 2, 3-member tuples '''
        def intp(f, s, e):
            return s + (e - s) * f
        return tuple([intp(fraction, start[i], end[i]) for i in range(3)])

    def rgb_to_hsl(rgb):
        ''' Adapted from M Bostock's RGB to HSL converter in d3.js
            https://github.com/mbostock/d3/blob/master/src/color/rgb.js '''
        r, g, b = float(rgb[0]) / 255.0,\
            float(rgb[1]) / 255.0,\
            float(rgb[2]) / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        h = s = l = (mx + mn) / 2
        if mx == mn:  # achromatic
            h = 0
            s = 0 if l > 0 and l < 1 else h
        else:
            d = mx - mn
            s = d / (mx + mn) if l < 0.5 else d / (2 - mx - mn)
            if mx == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif mx == g:
                h = (b - r) / d + 2
            else:
                h = r - g / d + 4

        return (int(round(h * 60, 4)), int(round(s * 100, 4)), int(round(l * 100, 4)))

    for i in r:
        # garyfeng: c_i could be rounded up so scl[c_i+1] will go off range
        # c_i = int(i*math.floor(SCL_FI)/round(r[-1])) # start color index
        # c_i = int(math.floor(i*math.floor(SCL_FI)/round(r[-1]))) # start color index
        # c_i = if c_i < len(scl)-1 else hsl_o

        c_i = int(math.floor(i))
        section_min = math.floor(i)
        section_max = math.ceil(i)
        fraction = (i - section_min)  # /(section_max-section_min)

        hsl_o = rgb_to_hsl(scl[c_i])  # convert rgb to hls
        hsl_f = rgb_to_hsl(scl[c_i + 1])
        # section_min = c_i*r[-1]/SCL_FI
        # section_max = (c_i+1)*(r[-1]/SCL_FI)
        # fraction = (i-section_min)/(section_max-section_min)
        hsl = interp3(fraction, hsl_o, hsl_f)
        c.append('hsl' + str(hsl))

    return cl.to_hsl(c)


def is_jupyter():
    '''Check if process is in Jupyter kernel'''
    try:
        get_ipython().config
        return True
    except NameError:
        return False
    return False


def is_outlier(points, thres=3.5):
    '''
    Detects outliers using MAD modified_z_score method, generalized to work on points.
    From https://stackoverflow.com/a/22357811/3865298
    @example

    is_outlier([1, 1, 1])
    # => array([False, False, False], dtype=bool)
    is_outlier([1, 1, 2])
    # => array([False, False,  True], dtype=bool)
    is_outlier([[1, 1], [1, 1], [1, 2]])
    # => array([False, False,  True], dtype=bool)
    '''
    points = np.array(points)
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    with np.errstate(divide='ignore', invalid='ignore'):
        modified_z_score = 0.6745 * diff / med_abs_deviation
        return modified_z_score > thres


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


def monkey_patch(base_cls, extend_cls):
    '''Monkey patch a base class with methods from extend_cls'''
    ext_fn_list = get_fn_list(extend_cls)
    for fn in ext_fn_list:
        setattr(base_cls, fn, getattr(extend_cls, fn))


def ndenumerate_nonan(arr):
    '''Generic ndenumerate for np.ndenumerate with only not gen_isnan values'''
    return (idx_v for idx_v in np.ndenumerate(arr) if not gen_isnan(idx_v[1]))


def nonan_all(v):
    '''Generic np.all that also returns false if array is all np.nan'''
    return bool(np.all(v) and ~np.all(np.isnan(v)))


def parallelize_fn(fn, args):
    '''
    Parallelize a method fn, args and return results with order preserved per args.
    fn should take only a single arg.
    @returns {list} results Order preserved output from fn.
    '''
    def pool_init():
        # you can never be too safe in multiprocessing gc
        import gc
        gc.collect()
    pool = mp.Pool(CPU_NUM,
                   initializer=pool_init, maxtasksperchild=1)
    results = pool.map(fn, args)
    pool.close()
    pool.join()
    return results


def read(data_path, **kwargs):
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
        data = read_as_df(data_path, **kwargs)
    else:
        data = read_as_plain(data_path, **kwargs)
    return data


def read_as_df(data_path, **kwargs):
    '''Submethod to read data as DataFrame'''
    ext = get_file_ext(data_path)
    if ext in ['.xlsx', 'xls']:
        data = pd.read_excel(data_path, **kwargs)
    else:  # .csv
        data = pd.read_csv(data_path, **kwargs)
    return data


def read_as_plain(data_path, **kwargs):
    '''Submethod to read data as plain type'''
    open_file = open(data_path, 'r')
    ext = get_file_ext(data_path)
    if ext == '.json':
        data = ujson.load(open_file, **kwargs)
    elif ext == '.yml':
        data = yaml.load(open_file, **kwargs)
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
    desc_list = [f'{get_class_name(cls)}:']
    for k, v in get_class_attr(cls).items():
        if k == 'spec':
            continue
        if _.is_dict(v) or _.is_dict(_.head(v)):
            desc_v = to_json(v)
        else:
            desc_v = v
        desc_list.append(f'- {k} = {desc_v}')
    desc = '\n'.join(desc_list)
    return desc


def session_df_to_data(session_df):
    '''
    Convert a multiindex session_df (df) with column levels (a,e,b,col) to session_data[aeb] = aeb_df
    @example

    session_df = util.read(filepath, header=[0, 1, 2, 3])
    session_data = util.session_df_to_data(session_df)
    '''
    session_data = {}
    aeb_list = get_df_aeb_list(session_df)
    for aeb in aeb_list:
        aeb_df = session_df.loc[:, aeb]
        session_data[aeb] = aeb_df
    return session_data


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
        json.dump(data, open_file, indent=2)
    elif ext == '.yml':
        yaml.dump(data, open_file)
    else:
        open_file.write(str(data))
    open_file.close()
    return data_path


def to_torch_batch(batch):
    '''Mutate a batch (dict) to make its values from numpy into PyTorch Variable'''
    float_data_names = ['states', 'actions', 'rewards', 'dones', 'next_states']
    for k in float_data_names:
        batch[k] = Variable(torch.from_numpy(batch[k]).float())
    return batch


def to_torch_nested_batch(batch):
    '''Mutate a nested batch (dict of lists) to make its values from numpy into PyTorch Variable.'''
    float_data_names = ['states', 'actions', 'rewards', 'dones', 'next_states']
    for k in float_data_names:
        batch[k] = [Variable(torch.from_numpy(x).float()) for x in batch[k]]
    return batch


def to_torch_nested_batch_ex_rewards(batch):
    '''Mutate a nested batch (dict of lists) to make its values from numpy into PyTorch Variable.'''
    float_data_names = ['states', 'actions', 'dones', 'next_states']
    for k in float_data_names:
        batch[k] = [Variable(torch.from_numpy(x).float()) for x in batch[k]]
    return batch
