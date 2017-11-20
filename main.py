from slm_lab import spec
from slm_lab.experiment import analysis
from slm_lab.experiment.control import Session, Trial
from slm_lab.lib import logger, util


def main():
    # Ghetto ass run method for now, only runs base case (1 agent 1 env 1 body)
    # TODO metaspec to specify specs to run, can be source from evolution suggestion
    # TODO set proper pattern
    logger.set_level('DEBUG')
    exp_spec = spec.get('default.json', 'base_case')
    # session = Session(exp_spec)
    # session_data = session.run()
    trial = Trial(exp_spec)
    trial_data = trial.run()


if __name__ == '__main__':
    main()
