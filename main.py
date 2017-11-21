from slm_lab.spec import spec_util
from slm_lab.experiment import analysis
from slm_lab.experiment.control import Session, Trial
from slm_lab.lib import logger, util


def main():
    # Ghetto ass run method for now, only runs base case (1 agent 1 env 1 body)
    # TODO metaspec to specify specs to run, can be source from evolution suggestion
    # TODO set proper pattern
    logger.set_level('DEBUG')
    spec = spec_util.get('base.json', 'base_case')
    # session = Session(spec)
    # session_data = session.run()
    trial = Trial(spec)
    trial_data = trial.run()


if __name__ == '__main__':
    main()
