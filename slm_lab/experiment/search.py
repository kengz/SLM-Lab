from abc import ABC, abstractmethod
from copy import deepcopy
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from slm_lab.spec import spec_util
import logging
import numpy as np
import pydash as ps
import random
import ray
import ray.tune
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
            config_space[key] = ray.tune.grid_search(v)
        elif space_type == 'choice':
            config_space[key] = lambda spec, v=v: random.choice(v)
        else:
            np_fn = getattr(np.random, space_type)
            config_space[key] = lambda spec, v=v: np_fn(*v)
    return config_space


def spec_from_config(spec, config):
    '''Helper to create spec from config - variables in spec.'''
    spec = deepcopy(spec)
    spec.pop('search', None)
    for k, v in config.items():
        ps.set_(spec, k, v)
    return spec


def create_remote_fn(spec):
    ray_gpu = int(bool(ps.get(spec, 'agent.0.net.gpu') and torch.cuda.device_count()))
    # TODO fractional ray_gpu is broken

    @ray.remote(num_gpus=ray_gpu)  # hack around bad Ray design of hard-coding
    def run_trial(init_trial_and_run, spec, config):
        trial_index = config.pop('trial_index')
        spec = spec_from_config(spec, config)
        spec['meta']['trial'] = trial_index  # inject trial index
        metrics = init_trial_and_run(spec)
        trial_data = {**config, **metrics, 'trial_index': spec['meta']['trial']}
        return trial_data
    return run_trial


def get_ray_results(pending_ids, ray_id_to_config):
    '''Helper to wait and get ray results into a new trial_data_dict, or handle ray error'''
    trial_data_dict = {}
    for _t in range(len(pending_ids)):
        ready_ids, pending_ids = ray.wait(pending_ids, num_returns=1)
        ready_id = ready_ids[0]
        try:
            trial_data = ray.get(ready_id)
            trial_index = trial_data.pop('trial_index')
            trial_data_dict[trial_index] = trial_data
        except:
            logger.exception(f'Trial failed: {ray_id_to_config[ready_id]}')
    return trial_data_dict


class RaySearch(ABC):
    '''RaySearch module for Experiment - Ray API integration with Lab'''

    def __init__(self, spec):
        self.spec = spec
        self.config_space = build_config_space(self.spec)
        logger.info(f'Running {util.get_class_name(self)}, with meta spec:\n{self.spec["meta"]}')

    @abstractmethod
    def generate_config(self):
        '''Generate the next config given config_space'''
        raise NotImplementedError
        return config

    @abstractmethod
    @lab_api
    def run(self):
        '''Implement the main run_trial loop.'''
        ray.init()
        # loop for max_trial: generate_config(); run_trial.remote(config)
        ray.shutdown()
        raise NotImplementedError
        return trial_data_dict


class RandomSearch(RaySearch):

    def generate_config(self):
        configs = []  # to accommodate for grid_search
        for resolved_vars, config in ray.tune.suggest.variant_generator._generate_variants(self.config_space):
            # inject trial_index for tracking in Ray
            config['trial_index'] = spec_util.tick(self.spec, 'trial')['meta']['trial']
            configs.append(config)
        return configs

    @lab_api
    def run(self, init_trial_and_run):
        run_trial = create_remote_fn(self.spec)
        meta_spec = self.spec['meta']
        logging.getLogger('ray').propagate = True
        ray.init(**meta_spec.get('search_resources', {}))
        max_trial = meta_spec['max_trial']
        trial_data_dict = {}
        ray_id_to_config = {}
        pending_ids = []

        for _t in range(max_trial):
            configs = self.generate_config()
            for config in configs:
                ray_id = run_trial.remote(init_trial_and_run, self.spec, config)
                ray_id_to_config[ray_id] = config
                pending_ids.append(ray_id)

        trial_data_dict.update(get_ray_results(pending_ids, ray_id_to_config))
        ray.shutdown()
        return trial_data_dict
