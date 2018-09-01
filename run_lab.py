'''
The entry point of SLM Lab
Specify what to run in `config/experiments.json`
Then run `yarn start` or `python run_lab.py`
'''
import os
# NOTE increase if needed. Pytorch thread overusage https://github.com/pytorch/pytorch/issues/975
os.environ['OMP_NUM_THREADS'] = '1'
from importlib import reload
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.experiment.monitor import InfoSpace
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util, benchmarker


debug_modules = [
    # 'algorithm',
]
debug_level = 'DEBUG'
logger.toggle_debug(debug_modules, debug_level)


def run_benchmark(spec, const):
    benchmark_specs = benchmarker.generate_specs(spec, const)
    logger.info('Running benchmark')
    for spec_name, benchmark_spec in benchmark_specs.items():
        # run only if not already exist; benchmark mode only
        if not any(spec_name in filename for filename in os.listdir('data')):
            info_space = InfoSpace()
            info_space.tick('experiment')
            Experiment(benchmark_spec, info_space).run()


def run_by_mode(spec_file, spec_name, lab_mode):
    logger.info(f'Running lab in mode: {lab_mode}')
    spec = spec_util.get(spec_file, spec_name)
    info_space = InfoSpace()
    os.environ['PREPATH'] = util.get_prepath(spec, info_space)
    reload(logger)  # to set PREPATH properly
    # expose to runtime, '@' is reserved for 'enjoy@{prepath}'
    os.environ['lab_mode'] = lab_mode.split('@')[0]
    if lab_mode == 'search':
        info_space.tick('experiment')
        Experiment(spec, info_space).run()
    elif lab_mode == 'train':
        info_space.tick('trial')
        Trial(spec, info_space).run()
    elif lab_mode.startswith('enjoy'):
        prepath = lab_mode.split('@')[1]
        spec, info_space = util.prepath_to_spec_info_space(prepath)
        Session(spec, info_space).run()
    elif lab_mode == 'generate_benchmark':
        benchmarker.generate_specs(spec, const='agent')
    elif lab_mode == 'benchmark':
        # TODO allow changing const to env
        run_benchmark(spec, const='agent')
    elif lab_mode == 'dev':
        spec = util.override_dev_spec(spec)
        info_space.tick('trial')
        Trial(spec, info_space).run()
    else:
        logger.warn('lab_mode not recognized; must be one of `search, train, enjoy, benchmark, dev`.')


def main():
    experiments = util.read('config/experiments.json')
    for spec_file in experiments:
        for spec_name, lab_mode in experiments[spec_file].items():
            run_by_mode(spec_file, spec_name, lab_mode)


if __name__ == '__main__':
    main()
