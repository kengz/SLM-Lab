import logging
import os
import sys
from slm_lab.lib import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # mute tf warnings on optimized setup
LOG_FILEPATH = util.smart_path(
    f'log/{os.environ.get("PY_ENV")}_{util.get_timestamp()}_slm_lab.log')
LOG_FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
LOG_LEVEL = logging.DEBUG if bool(os.environ.get('DEBUG')) else logging.INFO


class DedentFormatter(logging.Formatter):
    '''The formatter to dedent broken python multiline string'''

    def format(self, record):
        record.msg = util.dedent(record.msg)
        return super(DedentFormatter, self).format(record)


os.makedirs(os.path.dirname(LOG_FILEPATH), exist_ok=True)
dedent_formatter = DedentFormatter(LOG_FORMAT)
fh = logging.FileHandler(LOG_FILEPATH)
fh.setFormatter(dedent_formatter)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(dedent_formatter)

lab_logger = logging.getLogger('slm')
lab_logger.setLevel(LOG_LEVEL)
lab_logger.addHandler(fh)
lab_logger.addHandler(sh)
lab_logger.propagate = False


def set_level(lvl):
    lab_logger.setLevel(lvl)


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
    return lab_logger.warn(msg, *args, **kwargs)
