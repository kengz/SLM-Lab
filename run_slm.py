from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util


def main():
    # TODO streamline to grunt from external in the next release
    # logger.set_level('DEBUG')
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    Trial(spec).run()
    # spec = spec_util.get('dqn.json', 'dqn_cartpole')
    # Experiment(spec).run()


if __name__ == '__main__':
    main()
