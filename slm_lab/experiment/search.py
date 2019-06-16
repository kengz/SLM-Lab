from copy import deepcopy
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import numpy as np
import pydash as ps
import random
import ray
import ray.tune as tune
import torch

logger = logger.get_logger(__name__)


def build_config_space(spec):
    '''
    Build ray config space from flattened spec.search
    Specify a config space in spec using `"{key}__{space_type}": {v}`.
    Where `{space_type}` is `grid_search` of `ray.tune`, or any function name of `np.random`:
    - `grid_search`: str/int/float. v = list of choices
    - `choice`: str/int/float. v = list of choices
    - `randint`: int. v = [low, high)
    - `uniform`: float. v = [low, high)
    - `normal`: float. v = [mean, stdev)

    For example:
    - `"explore_anneal_epi__randint": [10, 60],` will sample integers uniformly from 10 to 60 for `explore_anneal_epi`,
    - `"lr__uniform": [0.001, 0.1]`, and it will sample `lr` using `np.random.uniform(0.001, 0.1)`

    If any key uses `grid_search`, it will be combined exhaustively in combination with other random sampling.
    '''
    space_types = ('grid_search', 'choice', 'randint', 'uniform', 'normal')
    config_space = {}
    for k, v in util.flatten_dict(spec['search']).items():
        key, space_type = k.split('__')
        assert space_type in space_types, f'Please specify your search variable as {key}__<space_type> in one of {space_types}'
        if space_type == 'grid_search':
            config_space[key] = tune.grid_search(v)
        elif space_type == 'choice':
            config_space[key] = tune.sample_from(lambda spec, v=v: random.choice(v))
        else:
            np_fn = getattr(np.random, space_type)
            config_space[key] = tune.sample_from(lambda spec, v=v: np_fn(*v))
    return config_space


def infer_trial_resources(spec):
    '''Infer the resources_per_trial for ray from spec'''
    meta_spec = spec['meta']
    num_cpus = min(util.NUM_CPUS, meta_spec['max_session'])

    use_gpu = any(agent_spec['net'].get('gpu') for agent_spec in spec['agent'])
    requested_gpu = meta_spec['max_session'] if use_gpu else 0
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_gpus = min(gpu_count, requested_gpu)
    resources_per_trial = {'cpu': num_cpus, 'gpu': num_gpus}
    return resources_per_trial


def inject_config(spec, config):
    '''Inject flattened config into SLM Lab spec.'''
    spec = deepcopy(spec)
    spec.pop('search', None)
    for k, v in config.items():
        ps.set_(spec, k, v)
    return spec


def ray_trainable(config, reporter):
    '''
    Create an instance of a trainable function for ray: https://ray.readthedocs.io/en/latest/tune-usage.html#training-api
    Lab needs a spec and a trial_index to be carried through config, pass them with config in ray.run() like so:
    config = {
        'spec': spec,
        'trial_index': tune.sample_from(lambda spec: gen_trial_index()),
        ... # normal ray config with sample, grid search etc.
    }
    '''
    from slm_lab.experiment.control import Trial
    import os
    # restore data carried from ray.run() config
    spec = config.pop('spec')
    spec = inject_config(spec, config)
    # tick trial_index with proper offset
    trial_index = config.pop('trial_index')
    spec['meta']['trial'] = trial_index - 1
    spec_util.tick(spec, 'trial')
    # run SLM Lab trial
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # remove CUDA id restriction from ray
    metrics = Trial(spec).run()
    metrics.update(config)  # carry config for analysis too
    # ray report to carry data in ray trial.last_result
    reporter(trial_data={trial_index: metrics})


def run_ray_search(spec):
    '''
    Method to run ray search from experiment. Uses RandomSearch now.
    TODO support for other ray search algorithms: https://ray.readthedocs.io/en/latest/tune-searchalg.html
    '''
    logger.info(f'Running ray search for spec {spec["name"]}')
    # generate trial index to pass into Lab Trial
    global trial_index  # make gen_trial_index passable into ray.run
    trial_index = -1

    def gen_trial_index():
        global trial_index
        trial_index += 1
        return trial_index

    ray.init()

    ray_trials = tune.run(
        ray_trainable,
        name=spec['name'],
        config={
            'spec': spec,
            'trial_index': tune.sample_from(lambda spec: gen_trial_index()),
            **build_config_space(spec)
        },
        resources_per_trial=infer_trial_resources(spec),
        num_samples=spec['meta']['max_trial'],
        reuse_actors=True,
    )
    trial_data_dict = {}  # data for Lab Experiment to analyze
    for ray_trial in ray_trials:
        ray_trial_data = ray_trial.last_result['trial_data']
        trial_data_dict.update(ray_trial_data)

    ray.shutdown()
    return trial_data_dict


def run_param_specs(param_specs):
    '''Run the given param_specs in parallel trials using ray. Used for benchmarking.'''
    ray.init()
    ray_trials = tune.run(
        ray_trainable,
        name='param_specs',
        config={
            'spec': tune.grid_search(param_specs),
            'trial_index': 0,
        },
        resources_per_trial=infer_trial_resources(param_specs[0]),
        num_samples=1,
        reuse_actors=True,
    )
    ray.shutdown()
