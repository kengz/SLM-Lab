# The SLM Lab entrypoint
# to run scheduled set of specs:
# python run_lab.py job/experiments.json
# to run a single spec:
# python run_lab.py slm_lab/spec/experimental/a2c_pong.json a2c_pong train
from slm_lab import EVAL_MODES, TRAIN_MODES
from slm_lab.experiment import search
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
from xvfbwrapper import Xvfb
import os
import pydash as ps
import sys
import torch
import torch.multiprocessing as mp


debug_modules = [
    # 'algorithm',
]
debug_level = 'DEBUG'
logger.toggle_debug(debug_modules, debug_level)
logger = logger.get_logger(__name__)


def run_spec(spec, lab_mode):
    '''Run a spec in lab_mode'''
    os.environ['lab_mode'] = lab_mode
    if lab_mode in TRAIN_MODES:
        spec_util.save(spec)  # first save the new spec
        if lab_mode == 'dev':
            spec = spec_util.override_dev_spec(spec)
        if lab_mode == 'search':
            spec_util.tick(spec, 'experiment')
            Experiment(spec).run()
        else:
            spec_util.tick(spec, 'trial')
            Trial(spec).run()
    elif lab_mode in EVAL_MODES:
        spec = spec_util.override_enjoy_spec(spec)
        Session(spec).run()
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}')


def read_spec_and_run(spec_file, spec_name, lab_mode):
    '''Read a spec and run it in lab mode'''
    logger.info(f'Running lab spec_file:{spec_file} spec_name:{spec_name} in mode:{lab_mode}')
    if lab_mode in TRAIN_MODES:
        spec = spec_util.get(spec_file, spec_name)
    else:  # eval mode
        lab_mode, prename = lab_mode.split('@')
        spec = spec_util.get_eval_spec(spec_file, prename)

    if 'spec_params' not in spec:
        run_spec(spec, lab_mode)
    else:  # spec is parametrized; run them in parallel using ray
        param_specs = spec_util.get_param_specs(spec)
        search.run_param_specs(param_specs)


def main():
    '''Main method to run jobs from scheduler or from a spec directly'''
    args = sys.argv[1:]
    if len(args) <= 1:  # use scheduler
        job_file = args[0] if len(args) == 1 else 'job/experiments.json'
        for spec_file, spec_and_mode in util.read(job_file).items():
            for spec_name, lab_mode in spec_and_mode.items():
                read_spec_and_run(spec_file, spec_name, lab_mode)
    else:  # run single spec
        assert len(args) == 3, f'To use sys args, specify spec_file, spec_name, lab_mode'
        read_spec_and_run(*args)


if __name__ == '__main__':
    torch.set_num_threads(1)  # prevent multithread slowdown
    try:
        mp.set_start_method('spawn')  # for distributed pytorch to work
    except RuntimeError:
        pass

    if sys.platform == 'darwin':  # MacOS is not headless
        main()
    else:
        with Xvfb() as xvfb:  # to run on headless machines, e.g. plotly orca
            main()
