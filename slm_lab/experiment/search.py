from slm_lab.experiment import analysis
from slm_lab.lib import logger, util, viz
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import InCondition
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
import numpy as np
import pandas as pd
import pydash as _


class ExperimentExt:
    def trial_wrapper(self, cfg_spec):
        # spec = ...
        logger.info(f'SMAC spec {cfg_spec}')
        spec = self.spec.copy()
        spec['agent'][0]['algorithm']['explore_anneal_epi'] = cfg_spec['explore_anneal_epi']
        spec['agent'][0]['net']['hid_layers_activation'] = cfg_spec['hid_layers_activation']
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
        cs = ConfigurationSpace()

        c1 = CategoricalHyperparameter(
            "hid_layers_activation", ["relu", "sigmoid"])
        cs.add_hyperparameter(c1)
        c2 = UniformIntegerHyperparameter(
            "explore_anneal_epi", 10, 60)
        cs.add_hyperparameter(c2)
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
