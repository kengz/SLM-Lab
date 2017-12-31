from slm_lab.experiment import analysis
from slm_lab.experiment.control import Session, Trial
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util


def main():
    # logger.set_level('DEBUG')
    # spec = spec_util.get('dqn.json', 'dqn_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_cartpole_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_cartpole_cartpole_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_acrobot_cartpole')
    spec = spec_util.get('dqn.json', 'dqn_2dball_cartpole')
    Session(spec).run()


if __name__ == '__main__':
    main()
