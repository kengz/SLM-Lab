# the control module
# creates and runs control loops at levels: Experiment, Trial, Session
from copy import deepcopy
from importlib import reload
from slm_lab.agent import AgentSpace, Agent
from slm_lab.agent.net import net_util
from slm_lab.env import EnvSpace, make_env
from slm_lab.experiment import analysis, search
from slm_lab.experiment.monitor import AEBSpace, Body, enable_aeb_space
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import torch.multiprocessing as mp


class Session:
    '''
    The base lab unit to run a RL session for a spec.
    Given a spec, it creates the agent and env, runs the RL loop,
    then gather data and analyze it to produce session data.
    '''

    def __init__(self, spec, global_nets=None):
        self.spec = spec
        self.index = self.spec['meta']['session']
        util.set_random_seed(self.spec)
        util.set_cuda_id(self.spec)
        util.set_logger(self.spec, logger, 'session')
        spec_util.save(spec, unit='session')

        # init agent and env
        self.env = make_env(self.spec)
        with util.ctx_lab_mode('eval'):  # env for eval
            self.eval_env = make_env(self.spec)
        body = Body(self.env, self.spec['agent'])
        self.agent = Agent(self.spec, body=body, global_nets=global_nets)

        enable_aeb_space(self)  # to use lab's data analysis framework
        logger.info(util.self_desc(self))

    def to_ckpt(self, env, mode='eval'):
        '''Check with clock whether to run log/eval ckpt: at the start, save_freq, and the end'''
        if mode == 'eval' and util.in_eval_lab_modes():  # avoid double-eval: eval-ckpt in eval mode
            return False
        clock = env.clock
        frame = clock.get()
        frequency = env.eval_frequency if mode == 'eval' else env.log_frequency
        if frame == 0 or clock.get('opt_step') == 0:  # avoid ckpt at init
            to_ckpt = False
        elif frequency is None:  # default episodic
            to_ckpt = env.done
        else:  # normal ckpt condition by mod remainder (general for venv)
            rem = env.num_envs or 1
            to_ckpt = (frame % frequency < rem) or frame == clock.max_frame
        return to_ckpt

    def try_ckpt(self, agent, env):
        '''Check then run checkpoint log/eval'''
        body = agent.body
        if self.to_ckpt(env, 'log'):
            body.train_ckpt()
            body.log_summary('train')

        if self.to_ckpt(env, 'eval'):
            avg_return = analysis.gen_avg_return(agent, self.eval_env)
            body.eval_ckpt(self.eval_env, avg_return)
            body.log_summary('eval')
            if analysis.new_best(agent):
                agent.save(ckpt='best')
            if len(body.eval_df) > 1:  # need > 1 row to calculate stability
                analysis.analyze_session(self.spec, body.eval_df, 'eval')
            if len(body.train_df) > 1:  # need > 1 row to calculate stability
                analysis.analyze_session(self.spec, body.train_df, 'train')

    def run_rl(self):
        '''Run the main RL loop until clock.max_frame'''
        logger.info(f'Running RL loop for trial {self.spec["meta"]["trial"]} session {self.index}')
        clock = self.env.clock
        state = self.env.reset()
        done = False
        while True:
            if util.epi_done(done):  # before starting another episode
                self.try_ckpt(self.agent, self.env)
                if clock.get() < clock.max_frame:  # reset and continue
                    clock.tick('epi')
                    state = self.env.reset()
                    done = False
            self.try_ckpt(self.agent, self.env)
            if clock.get() >= clock.max_frame:  # finish
                break
            clock.tick('t')
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state = next_state

    def close(self):
        '''Close session and clean up. Save agent, close env.'''
        self.agent.close()
        self.env.close()
        self.eval_env.close()
        logger.info(f'Session {self.index} done')

    def run(self):
        self.run_rl()
        metrics = analysis.analyze_session(self.spec, self.agent.body.eval_df, 'eval')
        self.close()
        return metrics


class SpaceSession(Session):
    '''Session for multi-agent/env setting'''

    def __init__(self, spec, global_nets=None):
        self.spec = spec
        self.index = self.spec['meta']['session']
        util.set_random_seed(self.spec)
        util.set_cuda_id(self.spec)
        util.set_logger(self.spec, logger, 'session')
        spec_util.save(spec, unit='session')

        self.aeb_space = AEBSpace(self.spec)
        self.env_space = EnvSpace(self.spec, self.aeb_space)
        self.aeb_space.init_body_space()
        self.agent_space = AgentSpace(self.spec, self.aeb_space, global_nets)

        logger.info(util.self_desc(self))

    def try_ckpt(self, agent_space, env_space):
        '''Try to checkpoint agent at the start, save_freq, and the end'''
        # TODO ckpt and eval not implemented for SpaceSession
        pass
        # for agent in agent_space.agents:
        #     for body in agent.nanflat_body_a:
        #         env = body.env
        #         super().try_ckpt(agent, env)

    def run_all_episodes(self):
        '''
        Continually run all episodes, where each env can step and reset at its own clock_speed and timeline.
        Will terminate when all envs done are done.
        '''
        all_done = self.aeb_space.tick('epi')
        state_space = self.env_space.reset()
        while not all_done:
            self.try_ckpt(self.agent_space, self.env_space)
            all_done = self.aeb_space.tick()
            action_space = self.agent_space.act(state_space)
            next_state_space, reward_space, done_space, info_v = self.env_space.step(action_space)
            self.agent_space.update(state_space, action_space, reward_space, next_state_space, done_space)
            state_space = next_state_space
        self.try_ckpt(self.agent_space, self.env_space)

    def close(self):
        '''Close session and clean up. Save agent, close env.'''
        self.agent_space.close()
        self.env_space.close()
        logger.info('Session done')

    def run(self):
        self.run_all_episodes()
        space_metrics_dict = analysis.analyze_session(self)
        self.close()
        return space_metrics_dict


def mp_run_session(spec, global_nets, mp_dict):
    '''Wrap for multiprocessing with shared variable'''
    session = Session(spec, global_nets)
    metrics = session.run()
    mp_dict[session.index] = metrics


class Trial:
    '''
    The lab unit which runs repeated sessions for a same spec, i.e. a trial
    Given a spec and number s, trial creates and runs s sessions,
    then gathers session data and analyze it to produce trial data.
    '''

    def __init__(self, spec):
        self.spec = spec
        self.index = self.spec['meta']['trial']
        util.set_logger(self.spec, logger, 'trial')
        spec_util.save(spec, unit='trial')

    def parallelize_sessions(self, global_nets=None):
        mp_dict = mp.Manager().dict()
        workers = []
        for _s in range(self.spec['meta']['max_session']):
            spec_util.tick(self.spec, 'session')
            w = mp.Process(target=mp_run_session, args=(deepcopy(self.spec), global_nets, mp_dict))
            w.start()
            workers.append(w)
        for w in workers:
            w.join()
        session_metrics_list = [mp_dict[idx] for idx in sorted(mp_dict.keys())]
        return session_metrics_list

    def run_sessions(self):
        logger.info('Running sessions')
        session_metrics_list = self.parallelize_sessions()
        return session_metrics_list

    def init_global_nets(self):
        session = Session(deepcopy(self.spec))
        if self.is_singleton:
            session.env.close()  # safety
            global_nets = net_util.init_global_nets(session.agent.algorithm)
        else:
            session.env_space.close()  # safety
            global_nets = [net_util.init_global_nets(agent.algorithm) for agent in session.agent_space.agents]
        return global_nets

    def run_distributed_sessions(self):
        logger.info('Running distributed sessions')
        global_nets = self.init_global_nets()
        session_metrics_list = self.parallelize_sessions(global_nets)
        return session_metrics_list

    def close(self):
        logger.info(f'Trial {self.index} done')

    def run(self):
        if self.spec['meta'].get('distributed') == False:
            session_metrics_list = self.run_sessions()
        else:
            session_metrics_list = self.run_distributed_sessions()
        metrics = analysis.analyze_trial(self.spec, session_metrics_list)
        self.close()
        return metrics['scalar']


class Experiment:
    '''
    The lab unit to run experiments.
    It generates a list of specs to search over, then run each as a trial with s repeated session,
    then gathers trial data and analyze it to produce experiment data.
    '''

    def __init__(self, spec):
        self.spec = spec
        self.index = self.spec['meta']['experiment']
        util.set_logger(self.spec, logger, 'trial')
        spec_util.save(spec, unit='experiment')
        SearchClass = getattr(search, spec['meta'].get('search'))
        self.search = SearchClass(deepcopy(self.spec))

    def init_trial_and_run(self, spec):
        '''Method to run trial with the properly updated spec (trial_index) from experiment.search.lab_trial.'''
        trial = Trial(spec)
        trial_metrics = trial.run()
        return trial_metrics

    def close(self):
        reload(search)  # fixes ray consecutive run crashing due to bad cleanup
        logger.info('Experiment done')

    def run(self):
        trial_data_dict = self.search.run(self.init_trial_and_run)
        experiment_df = analysis.analyze_experiment(self.spec, trial_data_dict)
        self.close()
        return experiment_df
