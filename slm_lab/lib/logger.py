from slm_lab.lib import util
import colorlog
import logging
import os
import pandas as pd
import sys
import warnings


class FixedList(list):
    '''fixed-list to restrict addition to root logger handler'''

    def append(self, e):
        pass


LOG_FORMAT = '[%(asctime)s PID:%(process)d %(levelname)s %(filename)s %(funcName)s] %(message)s'
color_formatter = colorlog.ColoredFormatter('%(log_color)s[%(asctime)s PID:%(process)d %(levelname)s %(filename)s %(funcName)s]%(reset)s %(message)s')
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(color_formatter)
lab_logger = logging.getLogger()
lab_logger.handlers = FixedList([sh])
logging.getLogger('ray').propagate = False  # hack to mute poorly designed ray TF warning log

# this will trigger from Experiment init on reload(logger)
if os.environ.get('PREPATH') is not None:
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

    log_filepath = os.environ['PREPATH'] + '.log'
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    # create file handler
    formatter = logging.Formatter(LOG_FORMAT)
    fh = logging.FileHandler(log_filepath)
    fh.setFormatter(formatter)
    # add stream and file handler
    lab_logger.handlers = FixedList([sh, fh])

if os.environ.get('LOG_LEVEL'):
    lab_logger.setLevel(os.environ['LOG_LEVEL'])
else:
    lab_logger.setLevel('INFO')


def set_level(lvl):
    lab_logger.setLevel(lvl)
    os.environ['LOG_LEVEL'] = lvl


def critical(msg, *args, **kwargs):
    return lab_logger.critical(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    return lab_logger.debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    return lab_logger.error(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    return lab_logger.exception(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    return lab_logger.info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    return lab_logger.warning(msg, *args, **kwargs)


def get_logger(__name__):
    '''Create a child logger specific to a module'''
    return logging.getLogger(__name__)


def toggle_debug(modules, level='DEBUG'):
    '''Turn on module-specific debugging using their names, e.g. algorithm, actor_critic, at the desired debug level.'''
    logger_names = list(logging.Logger.manager.loggerDict.keys())
    for module in modules:
        name = module.strip()
        for logger_name in logger_names:
            if name in logger_name.split('.'):
                module_logger = logging.getLogger(logger_name)
                module_logger.setLevel(getattr(logging, level))
