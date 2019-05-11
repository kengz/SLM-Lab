'''
The entry point of SLM Lab
# to run scheduled set of specs
python run_lab.py config/experiments.json
# to run a single spec
python run_lab.py slm_lab/spec/experimental/a2c_pong.json a2c_pong train
'''
from slm_lab import EVAL_MODES, TRAIN_MODES
from slm_lab.experiment import analysis, retro_analysis
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.experiment.monitor import InfoSpace
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


def run_train_mode(spec, lab_mode):
    '''Run to generate new data with `search, train, dev`'''
    info_space = InfoSpace()
    analysis.save_spec(spec, info_space)  # first save the new spec
    if lab_mode == 'search':
        spec_util.tick(spec, 'experiment')
        Experiment(spec, info_space).run()
    elif lab_mode in TRAIN_MODES:
        if lab_mode == 'dev':
            spec = spec_util.override_dev_spec(spec)
        spec_util.tick(spec, 'trial')
        Trial(spec, info_space).run()
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {TRAIN_MODES}')


def run_eval_mode(spec, lab_mode):
    '''Run using existing data with `enjoy, eval`. The eval mode is also what train mode's online eval runs in a subprocess via bash command'''
    # reconstruct spec and info_space from existing data
    lab_mode, prename = lab_mode.split('@')
    predir, _, _, _, _, _ = util.prepath_split(spec_file)
    prepath = f'{predir}/{prename}'
    spec, info_space = util.prepath_to_spec_info_space(prepath)
    # see InfoSpace def for more on these
    spec['meta']['ckpt'] = 'eval'
    spec['meta']['eval_model_prepath'] = prepath

    # no info_space.tick() as they are reconstructed
    if lab_mode in EVAL_MODES:
        spec = spec_util.override_enjoy_spec(spec)
        Session(spec, info_space).run()
        if lab_mode == 'eval':
            util.clear_periodic_ckpt(prepath)  # cleanup after itself
            retro_analysis.analyze_eval_trial(spec, info_space, predir)
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {EVAL_MODES}')


# TODO unify these later
# def run_by_mode(spec_file, spec_name, lab_mode):
#     '''The main run lab function for all lab_modes'''
#     logger.info(f'Running lab: spec_file {spec_file} spec_name {spec_name} in mode: {lab_mode}')
#     # '@' is reserved for EVAL_MODES
#     os.environ['lab_mode'] = lab_mode.split('@')[0]
#     if lab_mode in TRAIN_MODES:
#         run_train_mode(spec_file, spec_name, lab_mode)
#     else:
#         run_eval_mode(spec_file, spec_name, lab_mode)


def run_by_mode(spec_file, spec_name, lab_mode):
    '''Read a spec and run it in lab mode'''
    logger.info(f'Running lab: spec_file {spec_file} spec_name {spec_name} in mode: {lab_mode}')
    # '@' is reserved for EVAL_MODES
    os.environ['lab_mode'] = lab_mode.split('@')[0]
    spec = spec_util.get(spec_file, spec_name)
    if 'spec_params' not in spec:
        run_train_mode(spec, lab_mode)
    else:  # spec is parametrized; run them in parallel
        param_specs = spec_util.get_param_specs(spec)
        num_pro = spec['meta']['param_spec_process']
        # can't use Pool since it cannot spawn nested Process, which is needed for VecEnv and parallel sessions. So these will run and wait by chunks
        workers = [mp.Process(target=run_train_mode, args=(spec, lab_mode)) for spec in param_specs]
        for chunk_w in ps.chunk(workers, num_pro):
            for w in chunk_w:
                w.start()
            for w in chunk_w:
                w.join()


def main():
    '''Main method to run jobs from scheduler or from a spec directly'''
    args = sys.argv[1:]
    if len(args) <= 1:  # use scheduler
        job_file = args[0] if len(args) == 1 else 'config/experiments.json'
        for spec_file, spec_and_mode in util.read(job_file).items():
            for spec_name, lab_mode in spec_and_mode.items():
                run_by_mode(spec_file, spec_name, lab_mode)
    else:  # run single spec
        assert len(args) == 3, f'To use sys args, specify spec_file, spec_name, lab_mode'
        run_by_mode(*args)


if __name__ == '__main__':
    torch.set_num_threads(1)  # prevent multithread slowdown
    mp.set_start_method('spawn')  # for distributed pytorch to work
    if sys.platform == 'darwin':
        # avoid xvfb on MacOS: https://github.com/nipy/nipype/issues/1400
        main()
    else:
        with Xvfb() as xvfb:  # safety context for headless machines
            main()
