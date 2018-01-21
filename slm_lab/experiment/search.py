from copy import deepcopy
from ray.tune import register_trainable, grid_search, run_experiments
from ray.tune.trial import Trial
from slm_lab.experiment import analysis
from slm_lab.experiment.monitor import InfoSpace
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pandas as pd
import pydash as _
import ray


class RaySearch:
    '''Search module for Experiment - Ray.tune API integration with Lab'''

    def __init__(self, experiment):
        self.experiment = experiment

        '''
        Since every call of RaySearch.lab_trial starts on ray.remote on a new thread, we have to carry the trial_index from config at ray.Trial init.
        Then, in lab_trial, pop this trial_index, update a thread copy of info_space, and init lab trial.
        '''
        Trial._init = Trial.__init__

        def hack_trial_init(*args, **kwargs):
            self.experiment.info_space.tick('trial')
            trial_index = self.experiment.info_space.get('trial')
            kwargs['config'].update({'trial_index': trial_index})
            return Trial._init(*args, **kwargs)

        Trial.__init__ = hack_trial_init

    def build_config_space(self):
        '''
        Build ray config space from flattened spec.search for ray spec passed to run_experiments()
        Specify a config space in spec using `"{key}__{space_type}": {v}`.
        Where {space_type} is 'grid_search' of ray.tune, or any function name of np.random.
        - grid_search: str, int, float. v = list of choices
        - choice: str, int, float. v = list of choices
        - randint: int. v = [low, high)
        - uniform: float. v = [low, high)
        - normal: float. v = [mean, stdev)

        E.g. `"lr__uniform": [0.001, 0.1]`, and it will sample `lr` using np.random.uniform(0.001, 0.1)
        If any key uses 'grid_search', it will be combined exhaustively in combination with other random sampling.
        '''
        config_space = {}
        for k, v in util.flatten_dict(self.experiment.spec['search']).items():
            if '__' in k:
                key, space_type = k.split('__')
            else:
                key, space_type = k, 'grid_search'

            if space_type == 'grid_search':
                config_space[key] = grid_search(v)
            else:
                np_fn = getattr(np.random, space_type)
                config_space[key] = lambda spec: np_fn(*v)
        return config_space

    def spec_from_config(self, config):
        '''Helper to create spec from config - variables in spec.'''
        spec = deepcopy(self.experiment.spec)
        spec.pop('search', None)
        for k, v in config.items():
            _.set_(spec, k, v)
        return spec

    @lab_api
    def run(self):
        # serialize here as ray is not thread safe outside
        ray.register_custom_serializer(InfoSpace, use_pickle=True)
        ray.register_custom_serializer(pd.DataFrame, use_pickle=True)
        ray.register_custom_serializer(pd.Series, use_pickle=True)

        def lab_trial(config, reporter):
            '''Trainable method to run a trial given ray config and reporter'''
            trial_index = config.pop('trial_index')
            spec = self.spec_from_config(config)
            info_space = deepcopy(self.experiment.info_space)
            info_space.set('trial', trial_index)
            trial_fitness_df = self.experiment.init_trial_and_run(
                spec, info_space)
            fitness_vec = trial_fitness_df.iloc[0].to_dict()
            fitness = analysis.calc_fitness(trial_fitness_df)
            trial_index = trial_fitness_df.index[0]
            trial_data = {
                **config, **fitness_vec, 'fitness': fitness, 'trial_index': trial_index,
            }
            done = True
            # TODO timesteps = episode len or total_t from space_clock
            # call reporter from inside trial/session loop
            reporter(timesteps_total=-1, done=done, info=trial_data)

        register_trainable('lab_trial', lab_trial)

        # TODO use hyperband
        # TODO parallelize on trial sessions
        # TODO use advanced conditional config space via lambda func
        config_space = self.build_config_space()
        spec = self.experiment.spec
        ray_trials = run_experiments({
            spec['name']: {
                'run': 'lab_trial',
                # 'resources': {'cpu': 2, 'gpu': 0},
                'stop': {'done': True},
                'config': config_space,
                'repeat': spec['meta']['max_trial'],
            }
        })
        logger.info('Ray.tune experiment.search.run() done.')
        # compose data format for experiment analysis
        trial_data_dict = {}
        for ray_trial in ray_trials:
            exp_trial_data = ray_trial.last_result.info
            trial_index = exp_trial_data.pop('trial_index')
            trial_data_dict[trial_index] = exp_trial_data
        return trial_data_dict
