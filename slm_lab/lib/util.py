from datetime import datetime
from importlib import reload
from scipy import misc
from slm_lab import ROOT_DIR
import cv2
import json
import math
import numpy as np
import os
import pandas as pd
import pydash as ps
import regex as re
import scipy as sp
import subprocess
import sys
import torch
import torch.multiprocessing as mp
import ujson
import yaml

NUM_CPUS = mp.cpu_count()
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
        elif isinstance(obj, (np.ndarray, pd.Series)):
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
    delta_t = datetime.strptime(ts2, FILE_TS_FORMAT) - datetime.strptime(ts1, FILE_TS_FORMAT)
    return str(delta_t)


def cast_df(val):
    '''missing pydash method to cast value as DataFrame'''
    if isinstance(val, pd.DataFrame):
        return val
    return pd.DataFrame(val)


def cast_list(val):
    '''missing pydash method to cast value as list'''
    if ps.is_list(val):
        return val
    else:
        return [val]


def compact_dict(d):
    '''Return dict without None or np.nan values'''
    return {k: v for k, v in d.items() if not gen_isnan(v)}


def concat_batches(batches):
    '''
    Concat batch objects from body.memory.sample() into one batch, when all bodies experience similar envs
    Also concat any nested epi sub-batches into flat batch
    {k: arr1} + {k: arr2} = {k: arr1 + arr2}
    '''
    # if is nested, then is episodic
    is_episodic = isinstance(batches[0]['dones'][0], (list, np.ndarray))
    concat_batch = {}
    for k in batches[0]:
        datas = []
        for batch in batches:
            data = batch[k]
            if is_episodic:  # make into plain batch instead of nested
                data = np.concatenate(data)
            datas.append(data)
        concat_batch[k] = np.concatenate(datas)
    return concat_batch


def count_nonan(arr):
    try:
        return np.count_nonzero(~np.isnan(arr))
    except Exception:
        return len(filter_nonan(arr))


def dedent(string):
    '''Method to dedent the broken python multiline string'''
    return RE_INDENT.sub('', string)


def downcast_float32(df):
    '''Downcast any float64 col to float32 to allow safer pandas comparison'''
    for col in df.columns:
        if df[col].dtype == 'float':
            df[col] = df[col].astype('float32')
    return df


def fast_uniform_sample(mem_size, batch_size):
    '''Fast uniform sampling for large memory size (indices) by binning the number line and sampling from each bin'''
    if mem_size <= batch_size:
        return np.random.randint(mem_size, size=batch_size)
    num_base = math.floor(mem_size / batch_size)
    bin_start = np.arange(batch_size, dtype=np.int) * num_base
    bin_idx = np.random.randint(num_base, size=batch_size)
    bin_idx += bin_start
    return bin_idx


def flatten_dict(obj, delim='.'):
    '''Missing pydash method to flatten dict'''
    nobj = {}
    for key, val in obj.items():
        if ps.is_dict(val) and not ps.is_empty(val):
            strip = flatten_dict(val, delim)
            for k, v in strip.items():
                nobj[key + delim + k] = v
        elif ps.is_list(val) and not ps.is_empty(val) and ps.is_dict(val[0]):
            for idx, v in enumerate(val):
                nobj[key + delim + str(idx)] = v
                if ps.is_object(v):
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


def fix_multi_index_dtype(df):
    '''Restore aeb multi_index dtype from string to int, when read from file'''
    df.columns = pd.MultiIndex.from_tuples([(int(x[0]), int(x[1]), int(x[2]), x[3]) for x in df.columns])
    return df


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
    aeb_list = sorted(ps.uniq([(a, e, b) for a, e, b, col in session_df.columns.tolist()]))
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
        if hasattr(v, '__dict__') or ps.is_tuple(v):
            val = str(v)
        else:
            val = v
        attr_dict[k] = val
    return attr_dict


def get_env_path(env_name):
    '''Get the path to Unity env binaries distributed via npm'''
    env_path = smart_path(f'node_modules/slm-env-{env_name}/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(env_dir), f'Missing {env_path}. See README to install from yarn.'
    return env_path


def get_file_ext(data_path):
    return os.path.splitext(data_path)[-1]


def get_fn_list(a_cls):
    '''
    Get the callable, non-private functions of a class
    @returns {[*str]} A list of strings of fn names
    '''
    fn_list = ps.filter_(dir(a_cls), lambda fn: not fn.endswith('__') and callable(getattr(a_cls, fn)))
    return fn_list


def get_git_sha():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], close_fds=True, cwd=ROOT_DIR).decode().strip()


def get_lab_mode():
    return os.environ.get('lab_mode')


def get_prepath(spec, info_space, unit='experiment'):
    spec_name = spec['name']
    predir = f'data/{spec_name}_{info_space.experiment_ts}'
    prename = f'{spec_name}'
    trial_index = info_space.get('trial')
    session_index = info_space.get('session')
    if unit == 'trial':
        prename += f'_t{trial_index}'
    elif unit == 'session':
        prename += f'_t{trial_index}_s{session_index}'
    prepath = f'{predir}/{prename}'
    return prepath


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
        for eb, body in ndenumerate_nonan(cls.body_a):
            new_data_a[eb] = data_a
        data_a = new_data_a
    return data_a


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


def is_singleton(spec):
    '''Check if spec uses a singleton Session'''
    return len(spec['agent']) == 1 and len(spec['env']) == 1 and spec['body']['num'] == 1


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


def override_dev_spec(spec):
    spec['meta']['max_session'] = 1
    spec['meta']['max_trial'] = 2
    return spec


def override_test_spec(spec):
    for env_spec in spec['env']:
        env_spec['max_episode'] = 3
        env_spec['max_timestep'] = 100
    spec['meta']['max_session'] = 1
    spec['meta']['max_trial'] = 2
    return spec


def parallelize_fn(fn, args, num_cpus=NUM_CPUS):
    '''
    Parallelize a method fn, args and return results with order preserved per args.
    fn should take only a single arg.
    @returns {list} results Order preserved output from fn.
    '''
    def pool_init():
        # you can never be too safe in multiprocessing gc
        import gc
        gc.collect()
    pool = mp.Pool(num_cpus, initializer=pool_init, maxtasksperchild=1)
    results = pool.map(fn, args)
    pool.close()
    pool.join()
    return results


def prepath_split(prepath):
    '''Split prepath into prefolder and prename'''
    prepath = prepath.strip('_')
    tail = prepath.split('data/')[-1]
    prefolder, prename = tail.split('/')
    return prefolder, prename


def prepath_to_experiment_ts(prepath):
    predir = prepath_to_predir(prepath)
    experiment_ts = RE_FILE_TS.findall(predir)[0]
    return experiment_ts


def prepath_to_info_space(prepath):
    '''Create info_space from prepath such that it returns the same prepath with spec'''
    from slm_lab.experiment.monitor import InfoSpace
    experiment_ts = prepath_to_experiment_ts(prepath)
    trial_index, session_index = prepath_to_idxs(prepath)
    # create info_space for prepath
    info_space = InfoSpace()
    info_space.experiment_ts = experiment_ts
    info_space.set('experiment', 0)
    info_space.set('trial', trial_index)
    info_space.set('session', session_index)
    return info_space


def prepath_to_idxs(prepath):
    '''Extract trial index and session index from prepath if available'''
    spec_name = prepath_to_spec_name(prepath)
    _prefolder, prename = prepath_split(prepath)
    idxs_tail = prename.replace(spec_name, '').strip('_')
    idxs_strs = idxs_tail.split('_')[:2]
    assert len(idxs_strs) > 0, 'No trial/session indices found in prepath'
    tidx = idxs_strs[0]
    assert tidx.startswith('t')
    trial_index = int(tidx.strip('t'))
    if len(idxs_strs) == 1:  # has session
        session_index = None
    else:
        sidx = idxs_strs[1]
        assert sidx.startswith('s')
        session_index = int(sidx.strip('s'))
    return trial_index, session_index


def prepath_to_predir(prepath):
    tail = prepath.split('data/')[-1]
    prefolder = tail.split('/')[0]
    predir = f'data/{prefolder}'
    return predir


def prepath_to_spec_name(prepath):
    predir = prepath_to_predir(prepath)
    tail = prepath.split('data/')[-1]
    prefolder = tail.split('/')[0]
    experiment_ts = prepath_to_experiment_ts(prepath)
    spec_name = prefolder.replace(experiment_ts, '').strip('_')
    return spec_name


def prepath_to_spec(prepath):
    '''Create spec from prepath such that it returns the same prepath with info_space'''
    prepath = prepath.strip('_')
    pre_spec_path = '_'.join(prepath.split('_')[:-1])
    spec_path = f'{pre_spec_path}_spec.json'
    # read the spec of prepath
    spec = read(spec_path)
    return spec


def prepath_to_spec_info_space(prepath):
    '''
    Given a prepath, read the correct spec and craete the info_space that will return the same prepath
    This is used for lab_mode: enjoy
    example: data/a2c_cartpole_2018_06_13_220436/a2c_cartpole_t0_s0
    '''
    spec = prepath_to_spec(prepath)
    info_space = prepath_to_info_space(prepath)
    check_prepath = get_prepath(spec, info_space, unit='session')
    assert check_prepath in prepath, f'{check_prepath}, {prepath}'
    return spec, info_space


def read(data_path, **kwargs):
    '''
    Universal data reading method with smart data parsing
    - {.csv, .xlsx, .xls} to DataFrame
    - {.json} to dict, list
    - {.yml} to dict
    - {*} to str
    - TODO {db-query} to dict, DataFrame
    @param {str} data_path The data path to read from
    @returns {data} The read data in sensible format
    @example

    data_df = util.read('test/fixture/lib/util/test_df.csv')
    data_df = util.read('test/fixture/lib/util/test_df.xls')
    data_df = util.read('test/fixture/lib/util/test_df.xlsx')
    # => <DataFrame>

    data_dict = util.read('test/fixture/lib/util/test_dict.json')
    data_dict = util.read('test/fixture/lib/util/test_dict.yml')
    # => <dict>

    data_list = util.read('test/fixture/lib/util/test_list.json')
    # => <list>

    data_str = util.read('test/fixture/lib/util/test_str.txt')
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
    if ext in ['.xlsx', '.xls']:
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
    from_idx = ps.find_index(SPACE_PATH, lambda s: from_class_name in (s, s.replace('_', '')))
    from_idx = max(from_idx, 0)
    attr_path = attr_path.split('.')
    to_idx = SPACE_PATH.index(attr_path[0])
    assert -1 not in (from_idx, to_idx)
    if from_idx < to_idx:
        path_link = SPACE_PATH[from_idx: to_idx]
    else:
        path_link = ps.reverse(SPACE_PATH[to_idx: from_idx])

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
            desc_v = v['name']
        elif ps.is_dict(v) or ps.is_dict(ps.head(v)):
            desc_v = to_json(v)
        else:
            desc_v = v
        desc_list.append(f'- {k} = {desc_v}')
    desc = '\n'.join(desc_list)
    return desc


def session_df_to_data(session_df):
    '''
    Convert a multi_index session_df (df) with column levels (a,e,b,col) to session_data[aeb] = aeb_df
    @example

    session_df = util.read(filepath, header=[0, 1, 2, 3])
    session_data = util.session_df_to_data(session_df)
    '''
    session_data = {}
    fix_multi_index_dtype(session_df)
    aeb_list = get_df_aeb_list(session_df)
    for aeb in aeb_list:
        aeb_df = session_df.loc[:, aeb]
        session_data[aeb] = aeb_df
    return session_data


def set_attr(obj, attr_dict, keys=None):
    '''Set attribute of an object from a dict'''
    if keys is not None:
        attr_dict = ps.pick(attr_dict, keys)
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
    return obj


def set_module_seed(random_seed):
    '''Set all the module random seeds'''
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def set_session_logger(spec, info_space, logger):
    '''Set the logger for a session give its spec and info_space'''
    os.environ['PREPATH'] = get_prepath(spec, info_space, unit='session')
    reload(logger)  # to set session-specific logger


def _sizeof(obj, seen=None):
    '''Recursively finds size of objects'''
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([_sizeof(v, seen) for v in obj.values()])
        size += sum([_sizeof(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += _sizeof(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([_sizeof(i, seen) for i in obj])
    return size


def sizeof(obj, divisor=1e6):
    '''Return the size of object, in MB by default'''
    return _sizeof(obj) / divisor


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


def to_one_hot(data, max_val):
    '''Convert an int list of data into one-hot vectors'''
    return np.eye(max_val)[np.array(data)]


def to_render():
    return get_lab_mode() in ('dev', 'enjoy') and os.environ.get('RENDER', 'true') == 'true'


def to_torch_batch(batch, device, is_episodic):
    '''Mutate a batch (dict) to make its values from numpy into PyTorch tensor'''
    for k in batch:
        if is_episodic:  # for episodic format
            batch[k] = np.concatenate(batch[k])
        elif ps.is_list(batch[k]):
            batch[k] = np.array(batch[k])
        batch[k] = torch.from_numpy(batch[k].astype('float32')).to(device)
    return batch


def to_tuple_list(l):
    '''Returns a copy of the list with its elements as tuples'''
    return [tuple(row) for row in l]


def track_mem(obj):
    '''Debug method to track memory footprint of object and its attributes'''
    global MEMTRACKER
    if not isinstance(MEMTRACKER, dict):
        MEMTRACKER = {}
    obj_name = get_class_name(obj)
    for k in dir(obj):
        if not k.startswith('_'):
            hash_k = f'{obj_name}.{k}'
            size = sizeof(getattr(obj, k))
            if hash_k not in MEMTRACKER:
                MEMTRACKER[hash_k] = size
            else:
                diff = size - MEMTRACKER[hash_k]
                MEMTRACKER[hash_k] = size
                if (diff > 1e-4) or (size > 1.0):
                    print(f'{hash_k} diff: {diff:.6f}, size: {size:.6f}')


def try_set_cuda_id(spec, info_space):
    '''Use trial and session id to hash and modulo cuda device count for a cuda_id to maximize device usage. Sets the net_spec for the base Net class to pick up.'''
    # Don't trigger any cuda call if not using GPU. Otherwise will break multiprocessing on machines with CUDA.
    # see issues https://github.com/pytorch/pytorch/issues/334 https://github.com/pytorch/pytorch/issues/3491 https://github.com/pytorch/pytorch/issues/9996
    for agent_spec in spec['agent']:
        if not agent_spec['net'].get('gpu'):
            return
    trial_idx = info_space.get('trial') or 0
    session_idx = info_space.get('session') or 0
    job_idx = trial_idx * spec['meta']['max_session'] + session_idx
    device_count = torch.cuda.device_count()
    if device_count == 0:
        cuda_id = None
    else:
        cuda_id = job_idx % device_count
        cuda_id += int(os.environ.get('CUDA_ID_OFFSET', 0))

    for agent_spec in spec['agent']:
        agent_spec['net']['cuda_id'] = cuda_id


def write(data, data_path):
    '''
    Universal data writing method with smart data parsing
    - {.csv, .xlsx, .xls} from DataFrame
    - {.json} from dict, list
    - {.yml} from dict
    - {*} from str(*)
    - TODO {db-query} from dict, DataFrame
    @param {*} data The data to write
    @param {str} data_path The data path to write to
    @returns {data_path} The data path written to
    @example

    data_path = util.write(data_df, 'test/fixture/lib/util/test_df.csv')
    data_path = util.write(data_df, 'test/fixture/lib/util/test_df.xls')
    data_path = util.write(data_df, 'test/fixture/lib/util/test_df.xlsx')

    data_path = util.write(data_dict, 'test/fixture/lib/util/test_dict.json')
    data_path = util.write(data_dict, 'test/fixture/lib/util/test_dict.yml')

    data_path = util.write(data_list, 'test/fixture/lib/util/test_list.json')

    data_path = util.write(data_str, 'test/fixture/lib/util/test_str.txt')
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
        df.to_csv(data_path)
    return data_path


def write_as_plain(data, data_path):
    '''Submethod to write data as plain type'''
    open_file = open(data_path, 'w')
    ext = get_file_ext(data_path)
    if ext == '.json':
        json.dump(data, open_file, indent=2, cls=LabJsonEncoder)
    elif ext == '.yml':
        yaml.dump(data, open_file)
    else:
        open_file.write(str(data))
    open_file.close()
    return data_path


def resize_image(im):
    return sp.misc.imresize(im, (110, 84))


def crop_image(im):
    return im[-84:, :]


def normalize_image(im):
    return np.divide(im, 255.0)


def transform_image(im):
    '''
    Image preprocessing from the paper "Playing Atari with Deep Reinforcement Learning, 2013, Mnih et al"
    Takes an RGB image and converts it to grayscale, downsizes to 110 x 84 and crops to square 84 x 84, taking bottomost rows of the image.
    '''
    if im.ndim != 3:
        print(f'Unexpected image dimension: {im.ndim}, {im.shape}')
    im = np.dot(im[..., :3], [0.299, 0.587, 0.114])
    im = resize_image(im)
    im = crop_image(im)
    im = normalize_image(im)
    return im


def debug_image(im):
    '''Use this method to render image the agent sees; waits for a key press before continuing'''
    cv2.imshow('image', im)
    cv2.waitKey(0)
