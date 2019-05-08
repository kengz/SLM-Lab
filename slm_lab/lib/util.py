from collections import deque
from contextlib import contextmanager
from datetime import datetime
from importlib import reload
from pprint import pformat
from slm_lab import ROOT_DIR, EVAL_MODES
import cv2
import json
import numpy as np
import operator
import os
import pandas as pd
import pydash as ps
import regex as re
import subprocess
import sys
import time
import torch
import torch.multiprocessing as mp
import ujson
import yaml

NUM_CPUS = mp.cpu_count()
FILE_TS_FORMAT = '%Y_%m_%d_%H%M%S'
RE_FILE_TS = re.compile(r'(\d{4}_\d{2}_\d{2}_\d{6})')
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


def clear_periodic_ckpt(prepath):
    '''Clear periodic (with -epi) ckpt files in prepath'''
    if '-epi' in prepath:
        run_cmd(f'rm {prepath}*')


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


def cond_multiget(arr, idxs):
    '''Get multi-idxs from an array depending if it's a python list or np.array'''
    if isinstance(arr, (list, deque)):
        return np.array(operator.itemgetter(*idxs)(arr))
    else:
        return arr[idxs]


def count_nonan(arr):
    try:
        return np.count_nonzero(~np.isnan(arr))
    except Exception:
        return len(filter_nonan(arr))


def downcast_float32(df):
    '''Downcast any float64 col to float32 to allow safer pandas comparison'''
    for col in df.columns:
        if df[col].dtype == 'float':
            df[col] = df[col].astype('float32')
    return df


def epi_done(done):
    '''
    General method to check if episode is done for both single and vectorized env
    Only return True for singleton done since vectorized env does not have a natural episode boundary
    '''
    return np.isscalar(done) and done


def find_ckpt(prepath):
    '''Find the ckpt-lorem-ipsum in a string and return lorem-ipsum'''
    if 'ckpt' in prepath:
        ckpt_str = ps.find(prepath.split('_'), lambda s: s.startswith('ckpt'))
        ckpt = ckpt_str.replace('ckpt-', '')
    else:
        ckpt = None
    return ckpt


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


def get_file_ext(data_path):
    '''get the `.ext` of file.ext'''
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
    t_str = '' if trial_index is None else f'_t{trial_index}'
    s_str = '' if session_index is None else f'_s{session_index}'
    if unit == 'trial':
        prename += t_str
    elif unit == 'session':
        prename += f'{t_str}{s_str}'
    ckpt = ps.get(info_space, 'ckpt')
    if ckpt is not None:
        prename += f'_ckpt-{ckpt}'
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


def in_eval_lab_modes():
    '''Check if lab_mode is one of EVAL_MODES'''
    return get_lab_mode() in EVAL_MODES


def is_jupyter():
    '''Check if process is in Jupyter kernel'''
    try:
        get_ipython().config
        return True
    except NameError:
        return False
    return False


@contextmanager
def ctx_lab_mode(lab_mode):
    '''
    Creates context to run method with a specific lab_mode
    @example
    with util.ctx_lab_mode('eval'):
        run_eval()
    '''
    prev_lab_mode = os.environ.get('lab_mode')
    os.environ['lab_mode'] = lab_mode
    yield
    if prev_lab_mode is None:
        del os.environ['lab_mode']
    else:
        os.environ['lab_mode'] = prev_lab_mode


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


def parallelize_fn(fn, args, num_cpus=NUM_CPUS):
    '''
    Parallelize a method fn, args and return results with order preserved per args.
    fn should take only a single arg.
    @returns {list} results Order preserved output from fn.
    '''
    pool = mp.Pool(num_cpus, maxtasksperchild=1)
    results = pool.map(fn, args)
    pool.close()
    pool.join()
    return results


def prepath_split(prepath):
    '''
    Split prepath into useful names. Works with predir (prename will be None)
    prepath: data/dqn_pong_2018_12_02_082510/dqn_pong_t0_s0
    predir: data/dqn_pong_2018_12_02_082510
    prefolder: dqn_pong_2018_12_02_082510
    prename: dqn_pong_t0_s0
    spec_name: dqn_pong
    experiment_ts: 2018_12_02_082510
    ckpt: ckpt-best of dqn_pong_t0_s0_ckpt-best if available
    '''
    prepath = prepath.strip('_')
    tail = prepath.split('data/')[-1]
    ckpt = find_ckpt(tail)
    if ckpt is not None:  # separate ckpt
        tail = tail.replace(f'_ckpt-{ckpt}', '')
    if '/' in tail:  # tail = prefolder/prename
        prefolder, prename = tail.split('/')
    else:
        prefolder, prename = tail, None
    predir = f'data/{prefolder}'
    spec_name = RE_FILE_TS.sub('', prefolder).strip('_')
    experiment_ts = RE_FILE_TS.findall(prefolder)[0]
    return predir, prefolder, prename, spec_name, experiment_ts, ckpt


def prepath_to_idxs(prepath):
    '''Extract trial index and session index from prepath if available'''
    _, _, prename, spec_name, _, _ = prepath_split(prepath)
    idxs_tail = prename.replace(spec_name, '').strip('_')
    idxs_strs = ps.compact(idxs_tail.split('_')[:2])
    if ps.is_empty(idxs_strs):
        return None, None
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


def prepath_to_spec(prepath):
    '''Create spec from prepath such that it returns the same prepath with info_space'''
    predir, _, prename, _, _, _ = prepath_split(prepath)
    sidx_res = re.search('_s\d+', prename)
    if sidx_res:  # replace the _s0 if any
        prename = prename.replace(sidx_res[0], '')
    spec_path = f'{predir}/{prename}_spec.json'
    # read the spec of prepath
    spec = read(spec_path)
    return spec


def prepath_to_info_space(prepath):
    '''Create info_space from prepath such that it returns the same prepath with spec'''
    from slm_lab.experiment.monitor import InfoSpace
    _, _, _, _, experiment_ts, ckpt = prepath_split(prepath)
    trial_index, session_index = prepath_to_idxs(prepath)
    # create info_space for prepath
    info_space = InfoSpace()
    info_space.experiment_ts = experiment_ts
    info_space.ckpt = ckpt
    info_space.set('experiment', 0)
    info_space.set('trial', trial_index)
    info_space.set('session', session_index)
    return info_space


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
    - {.csv} to DataFrame
    - {.json} to dict, list
    - {.yml} to dict
    - {*} to str
    @param {str} data_path The data path to read from
    @returns {data} The read data in sensible format
    @example

    data_df = util.read('test/fixture/lib/util/test_df.csv')
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
    if ext == '.csv':
        data = read_as_df(data_path, **kwargs)
    else:
        data = read_as_plain(data_path, **kwargs)
    return data


def read_as_df(data_path, **kwargs):
    '''Submethod to read data as DataFrame'''
    ext = get_file_ext(data_path)
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


def run_cmd(cmd):
    '''Run shell command'''
    print(f'+ {cmd}')
    proc = subprocess.Popen(cmd, cwd=ROOT_DIR, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    return proc


def run_cmd_wait(proc):
    '''Wait on a running process created by util.run_cmd and print its stdout'''
    for line in proc.stdout:
        print(line.decode(), end='')
    output = proc.communicate()[0]
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.args, proc.returncode, output)
    else:
        return output


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
            desc_v = pformat(v)
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
        aeb_df.reset_index(inplace=True, drop=True)  # guard for eval append-row
        session_data[aeb] = aeb_df
    return session_data


def set_attr(obj, attr_dict, keys=None):
    '''Set attribute of an object from a dict'''
    if keys is not None:
        attr_dict = ps.pick(attr_dict, keys)
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
    return obj


def set_cuda_id(spec, info_space):
    '''Use trial and session id to hash and modulo cuda device count for a cuda_id to maximize device usage. Sets the net_spec for the base Net class to pick up.'''
    # Don't trigger any cuda call if not using GPU. Otherwise will break multiprocessing on machines with CUDA.
    # see issues https://github.com/pytorch/pytorch/issues/334 https://github.com/pytorch/pytorch/issues/3491 https://github.com/pytorch/pytorch/issues/9996
    for agent_spec in spec['agent']:
        if not agent_spec['net'].get('gpu'):
            return
    trial_idx = info_space.get('trial') or 0
    session_idx = info_space.get('session') or 0
    job_idx = trial_idx * spec['meta']['max_session'] + session_idx
    job_idx += int(os.environ.get('CUDA_ID_OFFSET', 0))
    device_count = torch.cuda.device_count()
    if device_count == 0:
        cuda_id = None
    else:
        cuda_id = job_idx % device_count

    for agent_spec in spec['agent']:
        agent_spec['net']['cuda_id'] = cuda_id


def set_logger(spec, info_space, logger, unit=None):
    '''Set the logger for a lab unit give its spec and info_space'''
    os.environ['PREPATH'] = get_prepath(spec, info_space, unit=unit)
    reload(logger)  # to set session-specific logger


def set_random_seed(trial, session, spec):
    '''Generate and set random seed for relevant modules, and record it in spec.meta.random_seed'''
    random_seed = int(1e5 * (trial or 0) + 1e3 * (session or 0) + time.time())
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    spec['meta']['random_seed'] = random_seed
    return random_seed


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


def split_minibatch(batch, mb_size):
    '''Split a batch into minibatches of mb_size or smaller, without replacement'''
    size = len(batch['rewards'])
    assert mb_size < size, f'Minibatch size {mb_size} must be < batch size {size}'
    idxs = np.arange(size)
    np.random.shuffle(idxs)
    chunks = int(size / mb_size)
    nested_idxs = np.array_split(idxs, chunks)
    mini_batches = []
    for minibatch_idxs in nested_idxs:
        minibatch = {k: v[minibatch_idxs] for k, v in batch.items()}
        mini_batches.append(minibatch)
    return mini_batches


def to_json(d, indent=2):
    '''Shorthand method for stringify JSON with indent'''
    return json.dumps(d, indent=indent, cls=LabJsonEncoder)


def to_render():
    return get_lab_mode() in ('dev', 'enjoy') and os.environ.get('RENDER', 'true') == 'true'


def to_torch_batch(batch, device, is_episodic):
    '''Mutate a batch (dict) to make its values from numpy into PyTorch tensor'''
    for k in batch:
        if is_episodic:  # for episodic format
            batch[k] = np.concatenate(batch[k])
        elif ps.is_list(batch[k]):
            batch[k] = np.array(batch[k])
        batch[k] = torch.from_numpy(batch[k].astype(np.float32)).to(device)
    return batch


def write(data, data_path):
    '''
    Universal data writing method with smart data parsing
    - {.csv} from DataFrame
    - {.json} from dict, list
    - {.yml} from dict
    - {*} from str(*)
    @param {*} data The data to write
    @param {str} data_path The data path to write to
    @returns {data_path} The data path written to
    @example

    data_path = util.write(data_df, 'test/fixture/lib/util/test_df.csv')

    data_path = util.write(data_dict, 'test/fixture/lib/util/test_dict.json')
    data_path = util.write(data_dict, 'test/fixture/lib/util/test_dict.yml')

    data_path = util.write(data_list, 'test/fixture/lib/util/test_list.json')

    data_path = util.write(data_str, 'test/fixture/lib/util/test_str.txt')
    '''
    data_path = smart_path(data_path)
    data_dir = os.path.dirname(data_path)
    os.makedirs(data_dir, exist_ok=True)
    ext = get_file_ext(data_path)
    if ext == '.csv':
        write_as_df(data, data_path)
    else:
        write_as_plain(data, data_path)
    return data_path


def write_as_df(data, data_path):
    '''Submethod to write data as DataFrame'''
    df = cast_df(data)
    ext = get_file_ext(data_path)
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


# Atari image preprocessing


def to_opencv_image(im):
    '''Convert to OpenCV image shape h,w,c'''
    shape = im.shape
    if len(shape) == 3 and shape[0] < shape[-1]:
        return im.transpose(1, 2, 0)
    else:
        return im


def to_pytorch_image(im):
    '''Convert to PyTorch image shape c,h,w'''
    shape = im.shape
    if len(shape) == 3 and shape[-1] < shape[0]:
        return im.transpose(2, 0, 1)
    else:
        return im


def grayscale_image(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def resize_image(im, w_h):
    return cv2.resize(im, w_h, interpolation=cv2.INTER_AREA)


def normalize_image(im):
    '''Normalizing image by dividing max value 255'''
    # NOTE: beware in its application, may cause loss to be 255 times lower due to smaller input values
    return np.divide(im, 255.0)


def preprocess_image(im):
    '''
    Image preprocessing using OpenAI Baselines method: grayscale, resize
    This resize uses stretching instead of cropping
    '''
    im = to_opencv_image(im)
    im = grayscale_image(im)
    im = resize_image(im, (84, 84))
    im = np.expand_dims(im, 0)
    return im


def debug_image(im):
    '''
    Renders an image for debugging; pauses process until key press
    Handles tensor/numpy and conventions among libraries
    '''
    if torch.is_tensor(im):  # if PyTorch tensor, get numpy
        im = im.cpu().numpy()
    im = to_opencv_image(im)
    im = im.astype(np.uint8)  # typecast guard
    if im.shape[0] == 3:  # RGB image
        # accommodate from RGB (numpy) to BGR (cv2)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow('debug image', im)
    cv2.waitKey(0)
