from slm_lab.lib import util
import colorlog
import logging
import os
import pandas as pd
import sys
import warnings

# extra debugging level deeper than the default debug
DEBUG2 = 9
DEBUG3 = 8
logging.addLevelName(DEBUG2, 'DEBUG2')
logging.addLevelName(DEBUG3, 'DEBUG3')
setattr(logging, 'DEBUG2', DEBUG2)
setattr(logging, 'DEBUG3', DEBUG3)

LOG_FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
color_formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(asctime)s %(levelname)s]%(reset)s %(message)s')
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(color_formatter)
lab_logger = logging.getLogger()
lab_logger.addHandler(sh)
lab_logger.propagate = False


# this will trigger from Experiment init on reload(logger)
if os.environ.get('PREPATH') is not None:
    # mute the competing loggers
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings(
        'ignore', category=pd.io.pytables.PerformanceWarning)

    logging.getLogger('gym').setLevel(logging.WARN)
    logging.getLogger('requests').setLevel(logging.WARN)
    logging.getLogger('unityagents').setLevel(logging.WARN)

    log_filepath = os.environ['PREPATH'] + '.log'
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    # create file handler
    formatter = logging.Formatter(LOG_FORMAT)
    fh = logging.FileHandler(log_filepath)
    fh.setFormatter(formatter)
    # remove old handlers to prevent repeated logging
    for handler in lab_logger.handlers[:]:
        lab_logger.removeHandler(handler)
    # add stream and file handler
    lab_logger.addHandler(sh)
    lab_logger.addHandler(fh)

if os.environ.get('LOG_LEVEL'):
    lab_logger.setLevel(os.environ['LOG_LEVEL'])
else:
    lab_logger.setLevel('INFO')


class DedentFormatter(logging.Formatter):
    '''The formatter to dedent broken python multiline string'''

    def format(self, record):
        record.msg = util.dedent(record.msg)
        return super(DedentFormatter, self).format(record)


def to_init(info_space, spec):
    '''
    Whether the lab's logger had been initialized:
    - prepath present in env
    - importlib.reload(logger) had been called
    '''
    return os.environ.get('prepath') is None


def set_level(lvl):
    lab_logger.setLevel(lvl)
    os.environ['LOG_LEVEL'] = lvl


def critical(msg, *args, **kwargs):
    return lab_logger.critical(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    return lab_logger.debug(msg, *args, **kwargs)


def debug2(msg, *args, **kwargs):
    return lab_logger.log(DEBUG2, msg, *args, **kwargs)


def debug3(msg, *args, **kwargs):
    return lab_logger.log(DEBUG3, msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    return lab_logger.error(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    return lab_logger.exception(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    return lab_logger.info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    return lab_logger.warn(msg, *args, **kwargs)
