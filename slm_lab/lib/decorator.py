import time
from functools import wraps
from slm_lab.lib import logger


def lab_api(fn):
    '''
    Function decorator to label and check Lab API methods
    @examples

    from slm_lab.lib.decorator import lab_api
    @lab_api
    def foo():
        print('foo')
    '''
    @wraps(fn)
    def check_api(*args, **kwargs):
        res = fn(*args, **kwargs)
        logger.debug('API method')
        return res
    return check_api


def timeit(fn):
    '''
    Function decorator to measure execution time
    @examples

    from slm_lab.lib.decorator import timeit
    @timeit
    def foo(sec):
        time.sleep(sec)
        print('foo')

    foo(1)
    # => foo
    # => Timed: foo 1000.9971ms
    '''
    @wraps(fn)
    def time_fn(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        end = time.time()
        logger.debug(
            f'Timed: {fn.__name__} {round((end - start) * 1000, 4)}ms')
        return res
    return time_fn
