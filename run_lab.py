# The SLM Lab entrypoint
from glob import glob
from slm_lab import EVAL_MODES, TRAIN_MODES
from slm_lab.experiment import search
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import os
import sys
import torch.multiprocessing as mp


debug_modules = [
    # 'algorithm',
]
debug_level = 'DEBUG'
logger.toggle_debug(debug_modules, debug_level)
logger = logger.get_logger(__name__)


def get_spec(spec_file, spec_name, lab_mode, pre_):
    '''Get spec using args processed from inputs'''
    if lab_mode in TRAIN_MODES:
        if pre_ is None:  # new train trial
            spec = spec_util.get(spec_file, spec_name)
        else:
            # for resuming with train@{predir}
            # e.g. train@latest (fill find the latest predir)
            # e.g. train@data/reinforce_cartpole_2020_04_13_232521
            predir = pre_
            if predir == 'latest':
                predir = sorted(glob(f'data/{spec_name}*/'))[-1]  # get the latest predir with spec_name
            _, _, _, _, experiment_ts = util.prepath_split(predir)  # get experiment_ts to resume train spec
            logger.info(f'Resolved to train@{predir}')
            spec = spec_util.get(spec_file, spec_name, experiment_ts)
    elif lab_mode == 'enjoy':
        # for enjoy@{session_spec_file}
        # e.g. enjoy@data/reinforce_cartpole_2020_04_13_232521/reinforce_cartpole_t0_s0_spec.json
        session_spec_file = pre_
        assert session_spec_file is not None, 'enjoy mode must specify a `enjoy@{session_spec_file}`'
        spec = util.read(f'{session_spec_file}')
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}')
    return spec


def run_spec(spec, lab_mode):
    '''Run a spec in lab_mode'''
    os.environ['lab_mode'] = lab_mode  # set lab_mode
    spec = spec_util.override_spec(spec, lab_mode)  # conditionally override spec
    if lab_mode in TRAIN_MODES:
        spec_util.save(spec)  # first save the new spec
        if lab_mode == 'search':
            spec_util.tick(spec, 'experiment')
            Experiment(spec).run()
        else:
            spec_util.tick(spec, 'trial')
            Trial(spec).run()
    elif lab_mode in EVAL_MODES:
        Session(spec).run()
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}')


def get_spec_and_run(spec_file, spec_name, lab_mode):
    '''Read a spec and run it in lab mode'''
    logger.info(f'Running lab spec_file:{spec_file} spec_name:{spec_name} in mode:{lab_mode}')
    if '@' in lab_mode:  # process lab_mode@{predir/prename}
        lab_mode, pre_ = lab_mode.split('@')
    else:
        pre_ = None
    spec = get_spec(spec_file, spec_name, lab_mode, pre_)

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
                get_spec_and_run(spec_file, spec_name, lab_mode)
    else:  # run single spec
        assert len(args) == 3, f'To use sys args, specify spec_file, spec_name, lab_mode'
        get_spec_and_run(*args)


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')  # for distributed pytorch to work
    except RuntimeError:
        pass
    main()
