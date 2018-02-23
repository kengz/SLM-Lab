from copy import deepcopy
from ray.tune import register_trainable, grid_search, variant_generator, run_experiments
from slm_lab.experiment import analysis
from slm_lab.experiment.monitor import InfoSpace
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import os
import pandas as pd
import pydash as _
import random
import ray


def build_config_space(experiment):
    '''
    Build ray config space from flattened spec.search for ray spec passed to run_experiments()
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
    config_space = {}
    for k, v in util.flatten_dict(experiment.spec['search']).items():
        if '__' in k:
            key, space_type = k.split('__')
        else:
            key, space_type = k, 'grid_search'
        if space_type == 'grid_search':
            config_space[key] = grid_search(v)
        elif space_type == 'choice':
            config_space[key] = lambda spec, v=v: random.choice(v)
        else:
            np_fn = getattr(np.random, space_type)
            config_space[key] = lambda spec, v=v: np_fn(*v)
    return config_space


def spec_from_config(experiment, config):
    '''Helper to create spec from config - variables in spec.'''
    spec = deepcopy(experiment.spec)
    spec.pop('search', None)
    for k, v in config.items():
        _.set_(spec, k, v)
    return spec


@ray.remote
def run_trial(experiment, config):
    trial_index = config.pop('trial_index')
    spec = spec_from_config(experiment, config)
    info_space = deepcopy(experiment.info_space)
    info_space.set('trial', trial_index)
    trial_fitness_df = experiment.init_trial_and_run(spec, info_space)
    fitness_vec = trial_fitness_df.iloc[0].to_dict()
    fitness = analysis.calc_fitness(trial_fitness_df)
    trial_data = {
        **config, **fitness_vec, 'fitness': fitness, 'trial_index': trial_index,
    }
    prepath = analysis.get_prepath(spec, info_space, unit='trial')
    util.write(trial_data, f'{prepath}_trial_data.json')
    return trial_data


class RaySearch(ABC):
    '''
    RaySearch module for Experiment - Ray API integration with Lab
    Abstract class ancestor to all RaySearch (using Ray).
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, experiment):
        self.experiment = experiment
        self.config_space = build_config_space(experiment)

    @abstractmethod
    def generate_config(self):
        '''
        Generate the next config given config_space, may update belief first.
        Remember to update trial_index in config.
        '''
        # use self.config_space to build config
        config['trial_index'] = self.experiment.info_space.tick('trial')[
            'trial']
        raise NotImplementedError
        return config

    @lab_api
    @abstractmethod
    def run(self):
        '''
        Implement the main run_trial loop.
        Remember to call ray init and disconnect before and after loop.
        '''
        ray.init()
        # serialize here as ray is not thread safe outside
        ray.register_custom_serializer(InfoSpace, use_pickle=True)
        ray.register_custom_serializer(pd.DataFrame, use_pickle=True)
        ray.register_custom_serializer(pd.Series, use_pickle=True)
        # loop for max_trial: generate_config(); run_trial.remote(config)
        ray.disconnect()
        raise NotImplementedError
        return trial_data_dict


# TODO implement different class extending this
class RandomSearch(RaySearch):

    def generate_config(self):
        # TODO pick first only, so ban grid search
        # TODO restore grid search here
        for resolved_vars, config in variant_generator._generate_variants(self.config_space):
            # RaySearch.lab_trial on ray.remote is not thread-safe, we have to carry the trial_index from config, created at the top level on the same thread, ticking once a trial (one samping).
            config['trial_index'] = self.experiment.info_space.tick('trial')[
                'trial']
            return config

    @lab_api
    def run(self):
        ray.init()
        # serialize here as ray is not thread safe outside
        ray.register_custom_serializer(InfoSpace, use_pickle=True)
        ray.register_custom_serializer(pd.DataFrame, use_pickle=True)
        ray.register_custom_serializer(pd.Series, use_pickle=True)

        max_trial = self.experiment.spec['meta']['max_trial']
        pending_ids = []
        trial_data_dict = {}

        for t in range(max_trial):
            config = self.generate_config()
            pending_ids.append(run_trial.remote(self.experiment, config))

        for t in range(max_trial):
            ready_ids, pending_ids = ray.wait(pending_ids, num_returns=1)
            try:
                trial_data = ray.get(ready_ids[0])
                trial_index = trial_data.pop('trial_index')
                trial_data_dict[trial_index] = trial_data
            except:
                logger.exception(f'Trial at ray id {ready_ids[0]} failed.')

        ray.disconnect()
        return trial_data_dict
