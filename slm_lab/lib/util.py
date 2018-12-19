from datetime import datetime
from importlib import reload
from slm_lab import ROOT_DIR
import cv2
import glob
import json
import numpy as np
import operator
import os
import pandas as pd
import pydash as ps
import regex as re
import shutil
import subprocess
import sys
import torch
import torch.multiprocessing as mp
import ujson
import yaml

NUM_CPUS = mp.cpu_count()
NUM_EVAL_EPISODES = 100
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
        run_cmd(f'rm {prepath}*', wait=False)


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
    if isinstance(arr, list):
        return np.array(operator.itemgetter(*idxs)(arr))
    else:
        return arr[idxs]


def copy_spec(original_trial_path, new_trial_path):
    '''Copies spec from original to new directory. Used in eval or enjoy mode.'''
    source = f'{original_trial_path}_spec.json'
    dest = f'{new_trial_path}_spec.json'
    shutil.copy(source, dest)


def copy_models(original_session_name, new_trial_name, number_sessions):
    '''Copies all model data from original_session to new_trial. Duplicates model x number_sessions. Used in eval or enjoy mode.'''
    for i in range(number_sessions):
        new_session_name = f'{new_trial_name}_s{i}'
        files = glob.glob(f'{original_session_name}*')
        new_files = [re.sub(original_session_name, new_session_name, f) for f in files]
        for f, nf in zip(files, new_files):
            shutil.copy(f, nf)


def copy_original_models(original_session_name, old_dir, new_dir):
    '''Copies the original checkpoint to the new directory. Used in eval or enjoy mode.'''
    files = glob.glob(f'{original_session_name}*')
    for f in files:
        nf = re.sub(old_dir, new_dir, f)
        shutil.copy(f, nf)


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
    if unit == 'trial':
        prename += f'_t{trial_index}'
    elif unit == 'session':
        prename += f'_t{trial_index}_s{session_index}'
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


def override_enjoy_spec(spec):
    spec['meta']['max_session'] = 1
    spec['meta']['max_trial'] = 1
    return spec


def override_eval_spec(spec):
    spec['meta']['max_session'] = 6
    spec['meta']['max_trial'] = 1
    for agent_spec in spec['agent']:
        if 'max_size' in agent_spec['memory']:
            agent_spec['memory']['max_size'] = 1000
    for env_spec in spec['env']:
        if 'max_total_t' in env_spec:
            del env_spec['max_total_t']
        env_spec['max_epi'] = NUM_EVAL_EPISODES
    return spec


def override_test_spec(spec):
    for agent_spec in spec['agent']:
        # covers episodic and timestep
        agent_spec['algorithm']['training_frequency'] = 1
        agent_spec['algorithm']['training_start_step'] = 1
        agent_spec['algorithm']['training_epoch'] = 1
        agent_spec['algorithm']['training_batch_epoch'] = 1
    for env_spec in spec['env']:
        env_spec['max_epi'] = 3
        env_spec['max_t'] = 20
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


def prepare_directory(new_spec, new_info_space, original_spec, original_info_space, original_prepath):
    '''Prepares a clean directory to evaluate or enjoy a particular model. Leaves original experiment directory untouched.'''
    assert new_spec['meta']['max_trial'] == 1
    predir, _, _, spec_name, _, ckpt = prepath_split(original_prepath)
    trial, session = prepath_to_idxs(original_prepath)
    new_prepath = get_prepath(new_spec, new_info_space, 'experiment')
    new_predir, _, _, _, _, _ = prepath_split(new_prepath)
    new_trial_name = f'{new_prepath}_t0'
    original_trial_name = f'{predir}/{spec_name}_t{trial}'
    original_session_name = f'{original_trial_name}_s{session}'
    if ckpt is not None:
        original_session_name = f'{original_session_name}_ckpt{ckpt}'
    copy_spec(original_trial_name, new_trial_name)
    copy_original_models(original_session_name, predir, new_predir)
    copy_models(original_session_name, new_trial_name, new_spec['meta']['max_session'])


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


def run_cmd(cmd, wait=False):
    '''Run shell command, with wait or without'''
    if wait:
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
    else:
        stdout = stderr = None
    print(f'+ {cmd}')
    proc = subprocess.Popen(cmd, cwd=ROOT_DIR, shell=True, stdout=stdout, stderr=stderr, close_fds=True)
    if wait:
        for line in proc.stdout:
            print(line.decode(), end='')
        output = proc.communicate()[0]
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(cmd, proc.returncode, output)
        else:
            return output
    else:
        return proc


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


def set_rand_seed(random_seed, env_space):
    '''Set all the module random seeds'''
    torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    envs = env_space.envs if hasattr(env_space, 'envs') else [env_space]
    for env in envs:
        try:
            env.u_env.seed(random_seed)
        except Exception as e:
            pass


def set_logger(spec, info_space, logger, unit=None):
    '''Set the logger for a lab unit give its spec and info_space'''
    os.environ['PREPATH'] = get_prepath(spec, info_space, unit=unit)
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


# Atari image transformation

def grayscale_image(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def resize_image(im, w_h):
    return cv2.resize(im, w_h, interpolation=cv2.INTER_AREA)


def crop_image(im):
    '''Crop away the unused top-bottom game borders of Atari'''
    return im[18:102, :]


def normalize_image(im):
    # NOTE: beware in its application, may cause loss to be 256 times lower due to smaller input values
    return np.divide(im, 255.0)


def nature_transform_image(im):
    '''
    Image preprocessing from the paper "Playing Atari with Deep Reinforcement Learning, 2013, Mnih et al"
    Takes an RGB image and converts it to grayscale, downsizes to 110 x 84 and crops to square 84 x 84 without the game border
    '''
    im = grayscale_image(im)
    im = resize_image(im, (84, 110))
    im = crop_image(im)
    return im


def openai_transform_image(im):
    '''
    Image transformation using OpenAI's baselines method: greyscale, resize
    Instead of cropping as done in nature_transform_image(), this resizes and stretches the image.
    '''
    im = grayscale_image(im)
    im = resize_image(im, (84, 84))
    return im


def transform_image(im, method='openai'):
    '''Apply image transformation using nature or openai method'''
    if method == 'nature':
        return nature_transform_image(im)
    elif method == 'openai':
        return openai_transform_image(im)
    else:
        raise ValueError('method must be one of: nature, openai')


def debug_image(im):
    '''Use this method to render image the agent sees; waits for a key press before continuing'''
    cv2.imshow('image', im)
    cv2.waitKey(0)


def mpl_debug_image(im):
    '''Uses matplotlib to plot image with bigger size, axes, and false color on greyscaled images'''
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(im)
    plt.show()
