'''
The entry point of SLM Lab
Specify what to run in `config/experiments.json`
Then run `yarn start` or `python run_lab.py`
'''
import os
# NOTE increase if needed. Pytorch thread overusage https://github.com/pytorch/pytorch/issues/975
os.environ['OMP_NUM_THREADS'] = '1'
from importlib import reload
from slm_lab.experiment import analysis
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.experiment.monitor import InfoSpace
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
from xvfbwrapper import Xvfb
import sys
import torch.multiprocessing as mp


debug_modules = [
    # 'algorithm',
]
debug_level = 'DEBUG'
logger.toggle_debug(debug_modules, debug_level)


def run_benchmark(spec_file):
    logger.info('Running benchmark')
    spec_dict = util.read(f'{spec_util.SPEC_DIR}/{spec_file}')
    for spec_name in spec_dict:
        # run only if not already exist; benchmark mode only
        if not any(spec_name in filename for filename in os.listdir('data')):
            run_by_mode(spec_file, spec_name, 'search')
        else:
            logger.info(f'{spec_name} is already ran and present in data/')


def run_by_mode(spec_file, spec_name, lab_mode):
    logger.info(f'Running lab in mode: {lab_mode}')
    spec = spec_util.get(spec_file, spec_name)
    info_space = InfoSpace()
    analysis.save_spec(spec, info_space, unit='experiment')

    # '@' is reserved for 'enjoy@{prepath}'
    os.environ['lab_mode'] = lab_mode.split('@')[0]
    os.environ['PREPATH'] = util.get_prepath(spec, info_space)
    reload(logger)  # to set PREPATH properly

    if lab_mode == 'search':
        info_space.tick('experiment')
        Experiment(spec, info_space).run()
    elif lab_mode.startswith('train'):
        if '@' in lab_mode:
            prepath = lab_mode.split('@')[1]
            spec, info_space = util.prepath_to_spec_info_space(prepath)
        else:
            info_space.tick('trial')
        Trial(spec, info_space).run()
    elif lab_mode.startswith('enjoy'):
        prepath = lab_mode.split('@')[1]
        spec, info_space = util.prepath_to_spec_info_space(prepath)
        Session(spec, info_space).run()
    elif lab_mode == 'dev':
        spec = util.override_dev_spec(spec)
        info_space.tick('trial')
        Trial(spec, info_space).run()
    else:
        logger.warn('lab_mode not recognized; must be one of `search, train, enjoy, benchmark, dev`.')


def main():
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        assert len(args) == 3, f'To use sys args, specify spec_file, spec_name, lab_mode'
        run_by_mode(*args)
        return

    experiments = util.read('config/experiments.json')
    for spec_file in experiments:
        for spec_name, lab_mode in experiments[spec_file].items():
            if lab_mode == 'benchmark':
                run_benchmark(spec_file)
            else:
                run_by_mode(spec_file, spec_name, lab_mode)


if __name__ == '__main__':
    mp.set_start_method('spawn')  # for distributed pytorch to work
    if sys.platform == 'darwin':
        # avoid xvfb for MacOS: https://github.com/nipy/nipype/issues/1400
        main()
    else:
        with Xvfb() as xvfb:  # safety context for headless machines
            main()
