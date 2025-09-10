import os
import sys
import warnings
import pandas as pd
from loguru import logger as loguru_logger

# Remove default handler and configure loguru
loguru_logger.remove()

# Format: consistent between console and file (colors only for console)
LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level} | "
    "PID:{process} | "
    "{name}:{function} | "
    "{message}"
)

CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level}</level> | "
    "<cyan>PID:{process}</cyan> | "
    "<cyan>{name}:{function}</cyan> | "
    "<level>{message}</level>"
)

# Setup console logging with shorter tracebacks
log_level = os.environ.get('LOG_LEVEL', 'INFO')
loguru_logger.add(
    sys.stdout, 
    format=CONSOLE_FORMAT, 
    level=log_level, 
    colorize=True,
    backtrace=False,  # Disable deep traceback for cleaner errors
    diagnose=False    # Disable variable inspection
)

# Setup file logging if LOG_PREPATH is set
if os.environ.get('LOG_PREPATH'):
    warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
    log_filepath = os.environ['LOG_PREPATH'] + '.log'
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    loguru_logger.add(
        log_filepath, 
        format=LOG_FORMAT, 
        level=log_level,
        backtrace=True,   # Keep full traceback in files for debugging
        diagnose=True     # Keep variable inspection in files
    )

# Disable noisy loggers
loguru_logger.disable("ray")

# Backward compatibility API
def set_level(lvl):
    os.environ['LOG_LEVEL'] = lvl

def critical(msg, *args, **kwargs):
    return loguru_logger.critical(msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    return loguru_logger.debug(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    return loguru_logger.error(msg, *args, **kwargs)

def exception(msg, *args, **kwargs):
    return loguru_logger.exception(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    return loguru_logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    return loguru_logger.warning(msg, *args, **kwargs)

def get_logger(name):
    return loguru_logger.bind(name=name)

def toggle_debug(modules, level='DEBUG'):
    if modules:
        set_level(level)