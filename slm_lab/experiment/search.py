from copy import deepcopy
from ray.tune import register_trainable, grid_search, run_experiments
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

    def build_config_space(self):
        '''Build ray config space from flattened spec.search for ray spec passed to run_experiments()'''
        # space_types = {
        #     'default': CategoricalHyperparameter,
        #     'categorical': CategoricalHyperparameter,
        #     'uniform_float': UniformFloatHyperparameter,
        #     'uniform_integer': UniformIntegerHyperparameter,
        #     'normal_float': NormalFloatHyperparameter,
        #     'normal_integer': NormalIntegerHyperparameter,
        # }
        for k, v in util.flatten_dict(self.experiment.spec['search']).items():
            if '__' in k:
                key, space_type = k.split('__')
            else:
                key, space_type = k, 'default'
            # TODO yield smth like:
            # config_space = {
            # "config": {
            #     "alpha": lambda spec: np.random.uniform(100),
            #     "beta": lambda spec: spec.config.alpha * np.random.normal(),
            #     "nn_layers": [
            #         grid_search([16, 64, 256]),
            #         grid_search([16, 64, 256]),
            #     ],
            # },
            # "repeat": 10,
            # }
            # param_cls = SMACSearch.space_types[space_type]
            # if space_type in ('default', 'categorical'):
            #     ck = param_cls(key, v)
            # else:
            #     ck = param_cls(key, *v)
            # cs.add_hyperparameter(ck)
        config_space = {
            "config": {
                "lr": grid_search([0.2, 0.4]),
            },
        }
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

        def run_trial(config, reporter):
            '''Trainable method to run a trial given ray config and reporter'''
            spec = self.spec_from_config(config)
            trial_fitness_df = self.experiment.init_trial_and_run(spec)
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

        register_trainable('run_trial', run_trial)

        # TODO use hyperband
        # TODO parallelize on trial sessions
        # TODO use ray API on normal experiment call to run trial
        # TODO use advanced conditional config space via lambda func
        config_space = self.build_config_space()
        exp_name = self.experiment.spec['name']
        ray_trials = run_experiments({
            exp_name: {
                'run': 'run_trial',
                'resources': {'cpu': 2, 'gpu': 0},
                'stop': {'done': True},
                **config_space,
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
