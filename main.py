from slm_lab.experiment.control import Session, Trial
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util


def main():
    # logger.set_level('DEBUG')
    spec = spec_util.get('demo.json', 'dqn_cartpole')
    Trial(spec).run()


if __name__ == '__main__':
    main()
