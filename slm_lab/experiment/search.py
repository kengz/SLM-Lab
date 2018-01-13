from copy import deepcopy
from slm_lab.experiment import analysis
from slm_lab.lib import logger, util
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
        '''Build SMAC config space from flattened spec.search. Use this to set on copy spec from cfg later.'''
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

    def spec_from_cfg(self, cfg):
        '''Helper to create spec from cfg'''
        spec = deepcopy(self.experiment.spec)
        spec.pop('search', None)
        var_spec = cfg.get_dictionary()
        for k, v in var_spec.items():
            _.set_(spec, k, v)
        return spec

    def run_trial(self, cfg):
        '''Wrapper for SMAC's tae_runner to run trial with a var_spec given by ConfigSpace'''
        # TODO proper id from top level
        spec = self.spec_from_cfg(cfg)
        var_spec = cfg.get_dictionary()
        trial_fitness_df = self.experiment.init_trial_and_run(spec)
        # trial fitness already avg over sessions and bodies
        fitness_vec = trial_fitness_df.loc[0].to_dict()
        fitness = analysis.calc_fitness(trial_fitness_df)
        cost = -fitness
        logger.info(
            f'Optimized cost: {cost}, fitness: {fitness}\n{fitness_vec}')
        exp_trial_data = {
            **var_spec, **fitness_vec, 'fitness': fitness,
        }
        return cost, exp_trial_data

    def get_trial_data_dict(self, smac):
        '''
        Recover the trial_id from smac history RunKeys,
        and the var_spec, fitness_vec, fitness from RunValues
        Format and return into trial_data_dict with index = trial_id and columns = [*var_spec, *fitness_vec, fitness]
        '''
        smac_hist = smac.get_runhistory()
        trial_data_dict = {}
        for trial_id, rv in enumerate(smac_hist.data.values()):
            exp_trial_data = rv.additional_info
            trial_data_dict[trial_id] = exp_trial_data
        return trial_data_dict

    def run(self):
        '''
        Interface method for Experiment class to run trials.
        Ensure trial is init properly with its trial_id.
        Search module must return best_spec and trial_data_dict with format {trial_id: exp_trial_data},
        where trial_data = {**var_spec, **fitness_vec, fitness}.
        '''
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
        best_cfg = smac.optimize()  # best var_spec
        best_spec = self.spec_from_cfg(best_cfg)
        trial_data_dict = self.get_trial_data_dict(smac)
        return best_spec, trial_data_dict
