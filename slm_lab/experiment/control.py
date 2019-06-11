# The control module
# Creates and runs control loops at levels: Experiment, Trial, Session
from copy import deepcopy
from slm_lab.agent import Agent, Body
from slm_lab.agent.net import net_util
from slm_lab.env import make_env
from slm_lab.experiment import analysis, search
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import torch.multiprocessing as mp


def make_agent_env(spec, global_nets=None):
    '''Helper to create agent and env given spec'''
    env = make_env(spec)
    body = Body(env, spec['agent'])
    agent = Agent(spec, body=body, global_nets=global_nets)
    return agent, env


def mp_run_session(spec, global_nets, mp_dict):
    '''Wrap for multiprocessing with shared variable'''
    session = Session(spec, global_nets)
    metrics = session.run()
    mp_dict[session.index] = metrics


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

        self.agent, self.env = make_agent_env(self.spec, global_nets)
        with util.ctx_lab_mode('eval'):  # env for eval
            self.eval_env = make_env(self.spec)
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
            to_ckpt = util.frame_mod(frame, frequency, env.num_envs) or frame == clock.max_frame
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
            if body.eval_reward_ma >= body.best_reward_ma:
                body.best_reward_ma = body.eval_reward_ma
                agent.save(ckpt='best')
            if len(body.train_df) > 1:  # need > 1 row to calculate stability
                metrics = analysis.analyze_session(self.spec, body.train_df, 'train')
                body.log_metrics(metrics['scalar'], 'train')
            if len(body.eval_df) > 1:  # need > 1 row to calculate stability
                metrics = analysis.analyze_session(self.spec, body.eval_df, 'eval')
                body.log_metrics(metrics['scalar'], 'eval')

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
        self.agent.body.log_metrics(metrics['scalar'], 'eval')
        self.close()
        return metrics


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
        spec = deepcopy(self.spec)
        for _s in range(spec['meta']['max_session']):
            spec_util.tick(spec, 'session')
            w = mp.Process(target=mp_run_session, args=(spec, global_nets, mp_dict))
            w.start()
            workers.append(w)
        for w in workers:
            w.join()
        session_metrics_list = [mp_dict[idx] for idx in sorted(mp_dict.keys())]
        return session_metrics_list

    def run_sessions(self):
        logger.info('Running sessions')
        if self.spec['meta']['max_session'] == 1:
            spec = deepcopy(self.spec)
            spec_util.tick(spec, 'session')
            session_metrics_list = [Session(spec).run()]
        else:
            session_metrics_list = self.parallelize_sessions()
        return session_metrics_list

    def init_global_nets(self):
        session = Session(deepcopy(self.spec))
        session.env.close()  # safety
        global_nets = net_util.init_global_nets(session.agent.algorithm)
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

    def close(self):
        logger.info('Experiment done')

    def run(self):
        trial_data_dict = search.run_ray_search(self.spec)
        experiment_df = analysis.analyze_experiment(self.spec, trial_data_dict)
        self.close()
        return experiment_df
