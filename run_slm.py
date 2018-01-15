from slm_lab.experiment.control import Session, Trial, Experiment
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util


def main():
    # logger.set_level('DEBUG')
    # spec = spec_util.get('dqn.json', 'dqn_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_cartpole_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_cartpole_multitask')
    # spec = spec_util.get('dqn.json', 'dqn_cartpole_cartpole_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_acrobot_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_2dball_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_cartpole_gridworld')
    # spec = spec_util.get('reinforce.json', 'reinforce_spec_template')
    # spec = spec_util.get('reinforce.json', 'reinforce_cartpole')
    # spec = spec_util.get('actor_critic.json', 'actor_critic_spec_template')
    spec = spec_util.get('actor_critic.json', 'actor_critic_cartpole')
    # spec = spec_util.get('dqn.json', 'dqn_2dball_cartpole')
    Trial(spec).run()
    # Experiment(spec).run()


if __name__ == '__main__':
    main()
