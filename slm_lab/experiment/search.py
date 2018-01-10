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
    space_types = {
        'default': CategoricalHyperparameter,
        'categorical': CategoricalHyperparameter,
        'uniform_float': UniformFloatHyperparameter,
        'uniform_integer': UniformIntegerHyperparameter,
        'normal_float': NormalFloatHyperparameter,
        'normal_integer': NormalIntegerHyperparameter,
    }

    def __init__(self, experiment):
        self.experiment = experiment

    def build_cs(self):
        '''Build SMAC config space from spec.search, flattened. Use this to set on copy spec from cs_spec later.'''
        cs = ConfigurationSpace()
        for k, v in util.flatten_dict(self.experiment.spec['search']).items():
            if '__' in k:
                key, space_type = k.split('__')
            else:
                key, space_type = k, 'default'
            param_cls = SMACSearch.space_types[space_type]
            if space_type in ('default', 'categorical'):
                ck = param_cls(key, v)
            else:
                ck = param_cls(key, *v)
            cs.add_hyperparameter(ck)
        return cs

    def run_trial(self, cs_spec):
        '''Wrapper for SMAC's tae_runner to run trial with a cs_spec given by ConfigSpace'''
        spec = self.experiment.spec.copy()
        spec.pop('search', None)
        var_spec = {}
        for k in cs_spec:
            var_spec[k] = cs_spec[k]
            _.set_(spec, k, cs_spec[k])
        # TODO proper id from top level
        trial = self.experiment.init_trial(spec)
        trial_df, trial_fitness_df = trial.run()
        # trial fitness already avg over sessions and bodies
        fitness_vec = trial_fitness_df.loc[0].to_dict()
        fitness = analysis.calc_fitness(trial_fitness_df)
        cost = -fitness
        logger.info(
            f'Optimized cost: {cost}, fitness: {fitness}\n{fitness_vec}')
        exp_trial_data = {
            **var_spec,
            **fitness_vec,
            'fitness': fitness,
        }
        return cost, exp_trial_data

    def get_experiment_df(self, smac):
        '''
        Recover the trial_id from smac history RunKeys,
        and the var_spec, fitness_vec, fitness from RunValues
        Format and return into experiment_df with index = trial_id and columns = [*var_spec, *fitness_vec, fitness]
        '''
        sr_list = []
        smac_hist = smac.get_runhistory()
        for trial_id, rv in enumerate(smac_hist.data.values()):
            exp_trial_data = rv.additional_info
            exp_sr = pd.Series(exp_trial_data, name=trial_id)
            sr_list.append(exp_sr)
        experiment_df = pd.DataFrame(sr_list)
        return experiment_df

    def run(self):
        cs = self.build_cs()
        scenario = Scenario({
            'run_obj': 'quality',  # or 'runtime' with 'cutoff_time'
            'runcount-limit': self.experiment.spec['meta']['max_trial'],
            'cs': cs,
            'deterministic': True,  # fitness roughly so
            'output_dir': 'smac3/',
            'shared_model': True,  # parallel
            'input_psmac_dirs': 'smac3-output*',
        })
        smac = SMAC(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=self.run_trial)
        incumbent = smac.optimize()  # best var_spec
        return self.get_experiment_df(smac)
