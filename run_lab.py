'''
The entry point of SLM Lab
Specify what to run in `config/experiments.json`
Then run `python run_lab.py` or `yarn start`
'''
import os
# NOTE increase if needed. Pytorch thread overusage https://github.com/pytorch/pytorch/issues/975
os.environ['OMP_NUM_THREADS'] = '1'
from slm_lab import EVAL_MODES, TRAIN_MODES
from slm_lab.experiment import analysis, retro_analysis
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


def run_new_mode(spec_file, spec_name, lab_mode):
    '''Run to generate new data with `search, train, dev`'''
    spec = spec_util.get(spec_file, spec_name)
    info_space = InfoSpace()
    analysis.save_spec(spec, info_space, unit='experiment')  # first save the new spec
    if lab_mode == 'search':
        info_space.tick('experiment')
        Experiment(spec, info_space).run()
    elif lab_mode.startswith('train'):
        info_space.tick('trial')
        Trial(spec, info_space).run()
    elif lab_mode == 'dev':
        spec = spec_util.override_dev_spec(spec)
        info_space.tick('trial')
        Trial(spec, info_space).run()
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {TRAIN_MODES}')


def run_old_mode(spec_file, spec_name, lab_mode):
    '''Run using existing data with `enjoy, eval`. The eval mode is also what train mode's online eval runs in a subprocess via bash command'''
    # reconstruct spec and info_space from existing data
    lab_mode, prename = lab_mode.split('@')
    predir, _, _, _, _, _ = util.prepath_split(spec_file)
    prepath = f'{predir}/{prename}'
    spec, info_space = util.prepath_to_spec_info_space(prepath)
    # see InfoSpace def for more on these
    info_space.ckpt = 'eval'
    info_space.eval_model_prepath = prepath

    # no info_space.tick() as they are reconstructed
    if lab_mode == 'enjoy':
        spec = spec_util.override_enjoy_spec(spec)
        Session(spec, info_space).run()
    elif lab_mode == 'eval':
        # example eval command:
        # python run_lab.py data/dqn_cartpole_2018_12_19_224811/dqn_cartpole_t0_spec.json dqn_cartpole eval@dqn_cartpole_t0_s1_ckpt-epi10-totalt1000
        spec = spec_util.override_eval_spec(spec)
        Session(spec, info_space).run()
        util.clear_periodic_ckpt(prepath)  # cleanup after itself
        retro_analysis.analyze_eval_trial(spec, info_space, predir)
    else:
        raise ValueError(f'Unrecognizable lab_mode not of {EVAL_MODES}')


def run_by_mode(spec_file, spec_name, lab_mode):
    '''The main run lab function for all lab_modes'''
    logger.info(f'Running lab in mode: {lab_mode}')
    # '@' is reserved for 'enjoy@{prename}'
    os.environ['lab_mode'] = lab_mode.split('@')[0]
    if lab_mode in TRAIN_MODES:
        run_new_mode(spec_file, spec_name, lab_mode)
    else:
        run_old_mode(spec_file, spec_name, lab_mode)


def main():
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        assert len(args) == 3, f'To use sys args, specify spec_file, spec_name, lab_mode'
        run_by_mode(*args)
        return

    experiments = util.read('config/experiments.json')
    for spec_file in experiments:
        for spec_name, lab_mode in experiments[spec_file].items():
            run_by_mode(spec_file, spec_name, lab_mode)


if __name__ == '__main__':
    mp.set_start_method('spawn')  # for distributed pytorch to work
    if sys.platform == 'darwin':
        # avoid xvfb for MacOS: https://github.com/nipy/nipype/issues/1400
        main()
    else:
        with Xvfb() as xvfb:  # safety context for headless machines
            main()
