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
        # TODO name-based data check for api methods
        output = fn(*args, **kwargs)
        logger.debug(f'API method: {fn.__name__}, output: {output}')
        return output
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
        output = fn(*args, **kwargs)
        end = time.time()
        logger.debug(
            f'Timed: {fn.__name__} {round((end - start) * 1000, 4)}ms')
        return output
    return time_fn
