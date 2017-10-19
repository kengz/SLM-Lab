import os
import pandas as pd
import pydash as _
import regex as re
import ujson
import yaml
from datetime import datetime


DF_FILE_EXT = ['.csv', '.xlsx', '.xls']
FILE_TS_FORMAT = '%Y_%m_%d_%H%M%S'
RE_FILE_TS = re.compile(r'(\d{4}_\d{2}_\d{2}_\d{6})')
ROOT_DIR = os.getcwd()


def cast_df(val):
    '''missing pydash method to cast value as DataFrame'''
    if isinstance(val, pd.DataFrame):
        return val
    return pd.DataFrame(val)


def cast_list(val):
    '''missing pydash method to cast value as list'''
    if isinstance(val, list):
        return val
    else:
        return [val]

# TODO logger, with dedent
# TODO refer to old exp util
# TODO experiment controller/util vs generic util
# TODO new exp id needs to reflect DAG structure: its ancestors; use neo4j to store
# TODO unit tests
# use simple tree level notation? will get out of hand, but store as metadata
# try first experiments as pytorch intro and simple dqn implementations, then ways to store experiments data in neo4j


def smart_path(data_path, as_dir=False):
    '''
    Resolve data_path into abspath with fallback to join from ROOT_DIR
    @param {str} data_path The input data path to resolve
    @param {bool} as_dir Whether to return as dirname
    @returns {str} The normalized absolute data_path
    @example

    smart_path('unity_lab/lib')
    # => '/Users/ANON/Documents/unity_lab/unity_lab/lib'

    smart_path('/tmp')
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


def get_file_ext(data_path):
    return os.path.splitext(data_path)[-1]


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


def smart_read(data_path):
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

    smart_read('test/fixture/lib/util/sample.csv')
    smart_read('test/fixture/lib/util/sample.xlsx')
    smart_read('test/fixture/lib/util/sample.xls')
    # => <DataFrame>

    smart_read('test/fixture/lib/util/sample.json')
    # => <dict or list>
    smart_read('test/fixture/lib/util/sample.yml')
    # => <dict>
    smart_read('test/fixture/lib/util/sample.txt')
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


def write_as_df(data, data_path):
    '''Submethod to write data as DataFrame'''
    df = cast_df(data)
    ext = get_file_ext(data_path)
    if ext in ['.xlsx', 'xls']:
        write = pd.ExcelWriter(data_path)
        df.to_excel(writer)
        writer.save()
        writer.close()
    else:  # .csv
        df.to_csv(data_path)
    return data_path


def write_as_plain(data_path):
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


def smart_write(data, data_path):
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


def get_timestamp(pattern=FILE_TS_FORMAT):
    '''
    Get current timestamp, defaults to format used for filename
    @param {str} pattern To format the timestamp
    @returns {str} timestamp
    @example

    get_timestamp()
    # => '2017_10_17_084739'
    '''
    timestamp_obj = datetime.now()
    timestamp = timestamp_obj.strftime(pattern)
    assert RE_FILE_TS.search(timestamp)
    return timestamp


def calc_timestamp_diff(ts2, ts1):
    '''
    Calculate the time from timestamps ts1 to ts2
    @param {str} ts2 Later timestamp in the FILE_TS_FORMAT
    @param {str} ts1 Earlier timestamp in the FILE_TS_FORMAT
    @returns {string} delta_t in %H:%M:%S format
    @example

    ts1 = get_timestamp()
    ts2 = get_timestamp()
    ts_diff = calc_timestamp_diff(ts2, ts1)
    # => '00:00:01'
    '''
    delta_t = datetime.strptime(
        ts2, FILE_TS_FORMAT) - datetime.strptime(
        ts1, FILE_TS_FORMAT)
    return str(delta_t)
