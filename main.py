from slm_lab.experiment import analysis
from slm_lab.experiment.control import Monitor, Session
from slm_lab.lib import logger, util


def main():
    # Ghetto ass run method for now, only runs base case (1 agent 1 env 1 body)
    logger.set_level('DEBUG')
    # TODO metaspec to specify specs to run, can be source from evolution suggestion
    # TODO set proper pattern
    demo_spec = util.read('slm_lab/spec/demo.json')
    monitor = Monitor(demo_spec)
    session_spec = demo_spec['base_case']
    # TODO temp set monitor method in session
    sess = Session(session_spec, monitor)
    session_data = sess.run()


if __name__ == '__main__':
    main()
