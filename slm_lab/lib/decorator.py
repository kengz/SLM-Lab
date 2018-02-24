import time
from functools import wraps
from slm_lab.lib import logger


def lab_api(fn):
    '''
    Function decorator to label and check Lab API methods
    @example

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


def ray_init_dc(fn):
    '''
    Function decorator for RaySearch to run ray.init() and ray.disconnect()
    @example

    from slm_lab.lib.decorator import ray_init_dc
    # method of RaySearch
    @ray_init_dc
    def run(self):
        return trial_data_dict
    '''
    @wraps(fn)
    def init_dc(*args, **kwargs):
        from slm_lab.experiment.control import Experiment
        from slm_lab.experiment.monitor import InfoSpace
        import pandas as pd
        import ray
        ray.init()
        # serialize here as ray is not thread safe outside
        ray.register_custom_serializer(Experiment, use_pickle=True)
        ray.register_custom_serializer(InfoSpace, use_pickle=True)
        ray.register_custom_serializer(pd.DataFrame, use_pickle=True)
        ray.register_custom_serializer(pd.Series, use_pickle=True)
        output = fn(*args, **kwargs)
        ray.disconnect()
        return output
    return init_dc


def timeit(fn):
    '''
    Function decorator to measure execution time
    @example

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
