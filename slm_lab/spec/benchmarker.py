'''
The benchmarker
Run the benchmark of agent vs environments, or environment vs agents, or both.
Generate benchmark specs like so:
- take a spec
- for each in benchmark envs
    - use the template env spec to update spec
    - append to benchmark specs
Interchange agent and env for the reversed benchmark.
'''
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import os
import pydash as _

AGENT_TEMPLATES = util.read(f'{spec_util.SPEC_DIR}/_agent.json')
ENV_TEMPLATES = util.read(f'{spec_util.SPEC_DIR}/_env.json')
BENCHMARK = util.read(f'{spec_util.SPEC_DIR}/_benchmark.json')


def generate_specs(spec, const='agent'):
    '''
    Generate benchmark specs with compatible  discrete/continuous/both types:
    - take a spec
    - for each in benchmark envs
        - use the template env spec to update spec
        - append to benchmark specs
    Interchange agent and env for the reversed benchmark.
    '''
    if const == 'agent':
        const_name = _.get(spec, 'agent.0.algorithm.name')
        variant = 'env'
    else:
        const_name = _.get(spec, 'env.0.name')
        variant = 'agent'

    filepath = f'{spec_util.SPEC_DIR}/benchmark_{const_name}.json'
    if os.path.exists(filepath):
        logger.info(
            f'Benchmark for {const_name} exists at {filepath} already, not overwriting.')
        benchmark_specs = util.read(filepath)
        return benchmark_specs

    logger.info(f'Generating benchmark for {const_name}')
    benchmark_variants = []
    benchmark_specs = {}
    for dist_cont, const_names in BENCHMARK[const].items():
        if const_name in const_names:
            benchmark_variants.extend(BENCHMARK[variant][dist_cont])
    for vary_name in benchmark_variants:
        vary_spec = ENV_TEMPLATES[vary_name]
        benchmark_spec = spec.copy()
        benchmark_spec[variant] = [vary_spec]
        spec_name = f'{const_name}_{vary_name}'
        benchmark_specs[spec_name] = benchmark_spec

    util.write(benchmark_specs, filepath)
    logger.info(
        f'Benchmark for {const_name} written to {filepath}.')
    return benchmark_specs
