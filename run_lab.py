'''
The entry point of SLM Lab
Specify what to run in `config/experiments.json`
Then run `yarn start` or `python run_lab.py`
'''
from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import os


def run_by_mode(spec_file, spec_name, run_mode):
    spec = spec_util.get(spec_file, spec_name)
    # TODO remove when analysis can save all plotly plots
    os.environ['run_mode'] = run_mode
    if run_mode == 'search':
        Experiment(spec).run()
    elif run_mode == 'train':
        Trial(spec).run()
    elif run_mode == 'enjoy':
        # TODO turn on save/load model mode
        # Session(spec).run()
        pass
    elif run_mode == 'benchmark':
        # TODO need to spread benchmark over spec on Experiment
        pass
    elif run_mode == 'dev':
        os.environ['PY_ENV'] = 'test'  # to not save in viz
        spec = util.override_dev_spec(spec)
        Trial(spec).run()
    else:
        logger.warn(
            'run_mode not recognized; must be one of `search, train, enjoy, benchmark, dev`.')


def main():
    experiments = util.read('config/experiments.json')
    for spec_file in experiments:
        for spec_name, run_mode in experiments[spec_file].items():
            run_by_mode(spec_file, spec_name, run_mode)


if __name__ == '__main__':
    main()
