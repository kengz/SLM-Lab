from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from deap import creator, base, tools, algorithms
from ray.tune import grid_search, variant_generator
from slm_lab.experiment import analysis
from slm_lab.experiment.monitor import InfoSpace
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import json
import numpy as np
import pandas as pd
import pydash as _
import random
import ray


def build_config_space(experiment):
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


def calc_population_size(experiment):
    '''Calculate the population size for RandomSearch or EvolutionarySearch'''
    pop_size = 2  # start with x2 for better sampling coverage
    for k, v in util.flatten_dict(experiment.spec['search']).items():
        if '__' in k:
            key, space_type = k.split('__')
        else:
            key, space_type = k, 'grid_search'
        if space_type in ('grid_search', 'choice'):
            pop_size *= len(v)
        elif space_type == 'randint':
            pop_size *= (v[1] - v[0])
        else:
            pop_size *= 5
    return pop_size


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
    '''
    RaySearch module for Experiment - Ray API integration with Lab
    Abstract class ancestor to all RaySearch (using Ray).
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''

    def __init__(self, experiment):
        from slm_lab.experiment.control import Experiment
        ray.register_custom_serializer(Experiment, use_pickle=True)
        ray.register_custom_serializer(InfoSpace, use_pickle=True)
        ray.register_custom_serializer(pd.DataFrame, use_pickle=True)
        ray.register_custom_serializer(pd.Series, use_pickle=True)
        self.experiment = experiment
        self.config_space = build_config_space(experiment)
        logger.info(
            f'Running {util.get_class_name(self)}, with meta spec:\n{self.experiment.spec["meta"]}')

    @abstractmethod
    def generate_config(self):
        '''
        Generate the next config given config_space, may update belief first.
        Remember to update trial_index in config here, since run_trial() on ray.remote is not thread-safe.
        '''
        # use self.config_space to build config
        config['trial_index'] = self.experiment.info_space.tick('trial')[
            'trial']
        raise NotImplementedError
        return config

    @abstractmethod
    @lab_api
    def run(self):
        '''
        Implement the main run_trial loop.
        Remember to call ray init and disconnect before and after loop.
        '''
        ray.init()
        # loop for max_trial: generate_config(); run_trial.remote(config)
        ray.disconnect()
        raise NotImplementedError
        return trial_data_dict


class RandomSearch(RaySearch):

    def generate_config(self):
        configs = []  # to accommodate for grid_search
        for resolved_vars, config in variant_generator._generate_variants(self.config_space):
            config['trial_index'] = self.experiment.info_space.tick('trial')[
                'trial']
            configs.append(config)
        return configs

    @lab_api
    def run(self):
        ray.init()
        max_trial = self.experiment.spec['meta']['max_trial']
        trial_data_dict = {}
        ray_id_to_config = {}
        pending_ids = []

        for _t in range(max_trial):
            configs = self.generate_config()
            for config in configs:
                ray_id = run_trial.remote(self.experiment, config)
                ray_id_to_config[ray_id] = config
                pending_ids.append(ray_id)

        trial_data_dict.update(get_ray_results(pending_ids, ray_id_to_config))
        ray.disconnect()
        return trial_data_dict


class EvolutionarySearch(RaySearch):

    def generate_config(self):
        for resolved_vars, config in variant_generator._generate_variants(self.config_space):
            # trial_index is set at population level
            return config

    def mutate(self, individual, indpb):
        '''
        Deap implementation for dict individual (config),
        mutate an attribute with some probability - resample using the generate_config method and ensuring the new value is different.
        @param {dict} individual Individual to be mutated.
        @param {float} indpb Independent probability for each attribute to be mutated.
        @returns A tuple of one individual.
        '''
        for k, v in individual.items():
            if random.random() < indpb:
                while True:
                    new_ind = self.generate_config()
                    if new_ind[k] != v:
                        individual[k] = new_ind[k]
                        break
        return individual,

    def cx_uniform(cls, ind1, ind2, indpb):
        '''
        Deap implementation for dict individual (config),
        do a uniform crossover that modify in place the two individuals. The attributes are swapped with probability indpd.
        @param {dict} ind1 The first individual participating in the crossover.
        @param {dict} ind2 The second individual participating in the crossover.
        @param {float} indpb Independent probabily for each attribute to be exchanged.
        @returns A tuple of two individuals.
        '''
        for k in ind1:
            if random.random() < indpb:
                ind1[k], ind2[k] = ind2[k], ind1[k]
        return ind1, ind2

    def init_deap(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', dict, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register('attr', self.generate_config)
        toolbox.register('individual', tools.initIterate,
                         creator.Individual, toolbox.attr)
        toolbox.register('population', tools.initRepeat,
                         list, toolbox.individual)

        toolbox.register('mate', self.cx_uniform, indpb=0.5)
        toolbox.register('mutate', self.mutate, indpb=1 /
                         len(toolbox.individual()))
        toolbox.register('select', tools.selTournament, tournsize=3)
        return toolbox

    @lab_api
    def run(self):
        ray.init()
        meta_spec = self.experiment.spec['meta']
        max_generation = meta_spec['max_generation']
        pop_size = meta_spec['max_trial'] or calc_population_size(
            self.experiment)
        logger.info(
            f'EvolutionarySearch max_generation: {max_generation}, population size: {pop_size}')
        trial_data_dict = {}
        config_hash = {}  # config hash_str to trial_index

        toolbox = self.init_deap()
        population = toolbox.population(n=pop_size)
        for gen in range(1, max_generation + 1):
            logger.info(f'Running generation: {gen}/{max_generation}')
            ray_id_to_config = {}
            pending_ids = []
            for individual in population:
                config = dict(individual.items())
                hash_str = util.to_json(config, indent=0)
                if hash_str not in config_hash:
                    trial_index = self.experiment.info_space.tick('trial')[
                        'trial']
                    config_hash[hash_str] = config['trial_index'] = trial_index
                    ray_id = run_trial.remote(self.experiment, config)
                    ray_id_to_config[ray_id] = config
                    pending_ids.append(ray_id)
                individual['trial_index'] = config_hash[hash_str]

            trial_data_dict.update(get_ray_results(
                pending_ids, ray_id_to_config))

            for individual in population:
                trial_index = individual.pop('trial_index')
                trial_data = trial_data_dict.get(
                    trial_index, {'fitness': 0})  # if trial errored
                individual.fitness.values = trial_data['fitness'],

            preview = 'Fittest of population preview:'
            for individual in tools.selBest(population, k=min(10, pop_size)):
                preview += f'\nfitness: {individual.fitness.values[0]}, {individual}'
            logger.info(preview)

            # prepare offspring for next generation
            if gen < max_generation:
                population = toolbox.select(population, len(population))
                # Vary the pool of individuals
                population = algorithms.varAnd(
                    population, toolbox, cxpb=0.5, mutpb=0.5)

        ray.disconnect()
        return trial_data_dict
