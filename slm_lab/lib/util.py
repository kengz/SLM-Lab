"""Core utilities for SLM-Lab.

This module contains non-ML utilities that work in minimal install mode.
ML-dependent utilities (torch, numpy, cv2) are in ml_util.py and are
re-exported here for backward compatibility when ML deps are available.
"""
from contextlib import contextmanager
from datetime import datetime
from loguru import logger as loguru_logger
from slm_lab import ROOT_DIR, EVAL_MODES, TRAIN_MODES
from slm_lab.lib import logger
from slm_lab.lib.env_var import lab_mode
import json
import os
import pandas as pd
import pydash as ps
import regex as re
import subprocess
import sys
import time
import ujson
import yaml

FILE_TS_FORMAT = '%Y_%m_%d_%H%M%S'
RE_FILE_TS = re.compile(r'(\d{4}_\d{2}_\d{2}_\d{6})')


def format_metrics(metrics: dict) -> list[str]:
    """Format metrics dict into clean key:value strings for logging.

    Handles numpy types, NaN values, and applies appropriate precision:
    - frame: scientific notation (1.00e+07)
    - total_reward, total_reward_ma: 2 decimal places
    - other floats: 4 significant figures
    """
    items = []
    for k, v in metrics.items():
        # Convert numpy types to Python types
        if hasattr(v, 'item'):
            v = v.item()

        if str(v).lower() == 'nan':
            items.append(f'{k}:nan')
        elif k == 'frame':
            items.append(f'{k}:{v:.2e}')
        elif k in ('total_reward', 'total_reward_ma'):
            items.append(f'{k}:{v:.2f}')
        elif isinstance(v, float) and not v.is_integer():
            items.append(f'{k}:{v:.4g}')
        elif isinstance(v, (int, float)):
            items.append(f'{k}:{v:g}')
        else:
            items.append(f'{k}:{v}')
    return items


def calc_srs_mean_std(sr_list):
    '''Given a list of series, calculate their mean and std. Truncates to min length to handle mismatched sizes.'''
    min_len = min(len(sr) for sr in sr_list)
    truncated = [sr.iloc[:min_len].reset_index(drop=True) for sr in sr_list]
    cat_df = pd.DataFrame(dict(enumerate(truncated)))
    mean_sr = cat_df.mean(axis=1)
    std_sr = cat_df.std(axis=1)
    return mean_sr, std_sr


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


def downcast_float32(df):
    '''Downcast any float64 col to float32 to allow safer pandas comparison'''
    for col in df.columns:
        if df[col].dtype == 'float':
            df[col] = df[col].astype('float32')
    return df


def frame_mod(frame, frequency, num_envs):
    '''
    Generic mod for (frame % frequency == 0) for when num_envs is 1 or more,
    since frame will increase multiple ticks for vector env, use the remainder'''
    remainder = num_envs or 1
    return (frame % frequency < remainder)


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


def get_class_name(obj, lower=False):
    '''Get the class name of an object'''
    class_name = obj.__class__.__name__
    if lower:
        class_name = class_name.lower()
    return class_name


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
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], close_fds=True, cwd=ROOT_DIR).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'  # git not available (e.g., in minimal/remote env)


def get_port():
    '''Get a unique port number for a run time as 4xxx, where xxx is the last 3 digits from the PID, front-padded with 0'''
    # get 3 digits from pid
    xxx = ps.pad_start(str(os.getpid())[-3:], 3, 0)
    port = int(f'4{xxx}')
    return port


def get_predir(spec):
    """Get the parent directory for experiment data."""
    spec_name = spec["name"]
    meta_spec = spec["meta"]
    predir = f"data/{spec_name}_{meta_spec['experiment_ts']}"
    return predir

def get_experiment_ts(path):
    """Extract experiment timestamp from a path like 'data/exp_name_2025_09_26_123456'."""
    matches = RE_FILE_TS.findall(path)
    return matches[0] if matches else None


def get_prepath(spec, unit='experiment'):
    spec_name = spec['name']
    meta_spec = spec['meta']
    predir = f'data/{spec_name}_{meta_spec["experiment_ts"]}'
    prename = f'{spec_name}'
    trial_index = meta_spec['trial']
    session_index = meta_spec['session']
    t_str = '' if trial_index is None else f'_t{trial_index}'
    s_str = '' if session_index is None else f'_s{session_index}'
    if unit == 'trial':
        prename += t_str
    elif unit == 'session':
        prename += f'{t_str}{s_str}'
    prepath = f'{predir}/{prename}'
    return prepath


def get_session_df_path(session_spec, df_mode):
    '''Method to return standard filepath for session_df (agent.mt.train_df/eval_df) for saving and loading'''
    info_prepath = session_spec['meta']['info_prepath']
    return f'{info_prepath}_session_df_{df_mode}.csv'


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


def insert_folder(prepath, folder):
    '''Insert a folder into prepath'''
    split_path = prepath.split('/')
    prename = split_path.pop()
    split_path += [folder, prename]
    return '/'.join(split_path)


def in_eval_lab_mode():
    '''Check if lab_mode is one of EVAL_MODES'''
    return lab_mode() in EVAL_MODES


def in_train_lab_mode():
    '''Check if lab_mode is one of TRAIN_MODES'''
    return lab_mode() in TRAIN_MODES


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
        foo()

    @util.ctx_lab_mode('eval')
    def foo():
        ...
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


def prepath_to_idxs(prepath):
    '''Extract trial index and session index from prepath if available'''
    tidxs = re.findall(r'_t(\d+)', prepath)
    trial_index = int(tidxs[0]) if tidxs else None
    sidxs = re.findall(r'_s(\d+)', prepath)
    session_index = int(sidxs[0]) if sidxs else None
    return trial_index, session_index


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
    data = pd.read_csv(data_path, **kwargs)
    return data




def read_as_plain(data_path, **kwargs):
    '''Submethod to read data as plain type'''
    open_file = open(data_path, 'r')
    ext = get_file_ext(data_path)
    if ext == '.json':
        data = ujson.load(open_file, **kwargs)
    elif ext == '.yml':
        data = yaml.load(open_file, Loader=yaml.FullLoader, **kwargs)
    else:
        data = open_file.read()
    open_file.close()
    return data


def log_dict(data: dict, title: str = None):
    '''Log dict as clean YAML format.'''
    lines = [f'{title}:'] if title else []
    for k, v in data.items():
        if isinstance(v, dict):
            yaml_str = yaml.dump({k: v}, default_flow_style=False, indent=2, sort_keys=False).rstrip()
            lines.append(yaml_str)
        elif v is not None and not ps.reg_exp_js_match(str(v), "/<.+>/"):
            lines.append(f'{k}: {v}')
    logger.info('\n'.join(lines))


def log_self_desc(cls, omit=None):
    '''Log self description in YAML-style format.'''
    try:
        from slm_lab.lib.ml_util import get_class_attr
        obj_dict = get_class_attr(cls)
    except ImportError:
        # Fallback for minimal install (no torch)
        obj_dict = {k: str(v) for k, v in cls.__dict__.items() if not k.startswith('_')}
    if omit:
        obj_dict = ps.omit(obj_dict, omit)
    log_dict(obj_dict, get_class_name(cls))


def set_attr(obj, attr_dict, keys=None):
    '''Set attribute of an object from a dict'''
    if keys is not None:
        attr_dict = ps.pick(attr_dict, keys)
    for attr, val in attr_dict.items():
        setattr(obj, attr, val)
    return obj


def set_logger(spec, logger, unit=None):
    '''Set the logger for a lab unit give its spec'''
    os.environ['LOG_PREPATH'] = insert_folder(get_prepath(spec, unit=unit), 'log')
    log_filepath = os.path.join(ROOT_DIR, os.environ['LOG_PREPATH'] + '.log')
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    # Remove existing file handlers (stdout remains)
    while len(loguru_logger._core.handlers) > 1:
        loguru_logger.remove(list(loguru_logger._core.handlers.keys())[-1])

    loguru_logger.add(
        log_filepath,
        format=logger.LOG_FORMAT,
        level='INFO',
        backtrace=True,
        diagnose=True
    )


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
        data_path = os.path.join(ROOT_DIR, data_path)
    if as_dir:
        data_path = os.path.dirname(data_path)
    return os.path.normpath(data_path)


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
    df.to_csv(data_path, index=False)
    return data_path




def write_as_plain(data, data_path):
    '''Submethod to write data as plain type'''
    open_file = open(data_path, 'w')
    ext = get_file_ext(data_path)
    if ext == '.json':
        try:
            from slm_lab.lib.ml_util import LabJsonEncoder
            json.dump(data, open_file, indent=2, cls=LabJsonEncoder)
        except ImportError:
            # Fallback for minimal install (no numpy)
            json.dump(data, open_file, indent=2)
    elif ext == '.yml':
        yaml.dump(data, open_file)
    else:
        open_file.write(str(data))
    open_file.close()
    return data_path


# Re-export ML utilities for backward compatibility
# These are only available when ML dependencies (torch, numpy, cv2) are installed
try:
    from slm_lab.lib.ml_util import (
        NUM_CPUS,
        LabJsonEncoder,
        batch_get,
        concat_batches,
        debug_image,
        epi_done,
        get_class_attr,
        grayscale_image,
        normalize_image,
        parallelize,
        preprocess_image,
        resize_image,
        set_cuda_id,
        set_random_seed,
        split_minibatch,
        to_json,
        to_opencv_image,
        to_pytorch_image,
        to_torch_batch,
        use_gpu,
    )
except ImportError:
    pass  # ML deps not available (minimal install mode)
