from slm_lab.experiment import analysis
from slm_lab.lib import logger, util, viz
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
)
from ConfigSpace.conditions import InCondition
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
import numpy as np
import pandas as pd
import pydash as _


class SMACSearch:
    cls_dict = {
        'default': CategoricalHyperparameter,
        'categorical': CategoricalHyperparameter,
        'uniform_float': UniformFloatHyperparameter,
        'uniform_integer': UniformIntegerHyperparameter,
        'normal_float': NormalFloatHyperparameter,
        'normal_integer': NormalIntegerHyperparameter,
    }

    def trial_wrapper(self, cs_spec):
        # spec = ...
        logger.info(f'SMAC spec {cs_spec}')
        spec = self.spec.copy()
        spec.pop('search', None)
        # TODO collect var dict for plotting later
        var_dict = {}
        for k in cs_spec:
            var_dict[k] = cs_spec[k]
            _.set_(spec, k, cs_spec[k])
        self.init_trial(spec)
        trial_df, trial_fitness_df = self.trial.run()
        # TODO need to recover data to experiment
        # t = self.trial.index
        # self.trial_df_dict[t] = trial_df
        # self.trial_fitness_df_dict[t] = trial_fitness_df
        fitness = analysis.calc_fitness(trial_fitness_df)
        cost = -fitness
        print(f'Optimized fitness: {fitness}, cost: {cost}')
        return cost

    def build_cs(self):
        '''Build SMAC config space from spec.search, flattened. Use this to set on copy spec from cs_spec later.'''
        cs = ConfigurationSpace()
        for k, v in util.flatten_dict(self.spec['search']).items():
            if '__' in k:
                key, space_type = k.split('__')
            else:
                key, space_type = k, 'default'
            param_cls = SMACSearch.cls_dict[space_type]
            if space_type in ('default', 'categorical'):
                ck = param_cls(key, v)
            else:
                ck = param_cls(key, *v)
            cs.add_hyperparameter(ck)
        return cs

    def run_smac(self):
        cs = self.build_cs()

        scenario = Scenario({
            "run_obj": "quality",  # or "runtime" with "cutoff_time"
            "runcount-limit": 1,  # max trials
            "cs": cs,
            "deterministic": False,
            # "output_dir": 'smac3/',
            "shared_model": True,  # parallel
            "input_psmac_dirs": 'smac3-output*',
        })

        smac = SMAC(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=self.trial_wrapper)
        incumbent = smac.optimize()
