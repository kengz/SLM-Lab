# The control module
# Creates and runs control loops at levels: Experiment, Trial, Session
from copy import deepcopy
from slm_lab.agent import Agent, Body
from slm_lab.agent import world as World
from slm_lab.agent.net import net_util
from slm_lab.env import make_env
from slm_lab.experiment import analysis, search
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util
import pydash as ps
import torch
import torch.multiprocessing as mp
from collections.abc import Iterable


def make_env_agents_world(spec, global_nets_list=None):
    '''Helper to create world (with its agents) and env given spec'''
    env = make_env(spec)

    # TODO replace by world and select world type in config. Default is SimpleSingleAgentWorld
    if "world" not in spec.keys():
        logger.info("'world' key not in spec. Thus using the default single or multi agent world class.")
        if len(spec['agent']) > 1:
            world = World.DefaultMultiAgentWorld(spec, env=env, global_nets_list=global_nets_list)
        else:
            world = World.DefaultSingleAgentWorld(spec, env=env, global_nets_list=global_nets_list)
    else:
        WorldClass = getattr(World, spec['world']['name'])
        world = WorldClass(spec, env=env, global_nets_list=global_nets_list)
    return world, env


def mp_run_session(spec, global_nets_list, mp_dict):
    '''Wrap for multiprocessing with shared variable'''
    session = Session(spec, global_nets_list)
    metrics = session.run()
    mp_dict[session.index] = metrics


class Session:
    '''
    The base lab unit to run a RL session for a spec.
    Given a spec, it creates the agent and env, runs the RL loop,
    then gather data and analyze it to produce session data.
    '''

    def __init__(self, spec, global_nets_list=None):
        self.spec = spec
        self.index = self.spec['meta']['session']
        util.set_random_seed(self.spec)
        util.set_cuda_id(self.spec)
        util.set_logger(self.spec, logger, 'session')
        spec_util.save(spec, unit='session')

        self.world, self.env = make_env_agents_world(self.spec, global_nets_list)
        if ps.get(self.spec, 'meta.rigorous_eval'):
            with util.ctx_lab_mode('eval'):
                self.eval_env = make_env(self.spec)
        else:
            self.eval_env = self.env
        logger.info(util.self_desc(self))
        logger.debug("End of Session __init__")

    def to_ckpt(self, env, mode='eval'):
        '''Check with clock whether to run log/eval ckpt: at the start, save_freq, and the end'''
        if mode == 'eval' and util.in_eval_lab_modes():  # avoid double-eval: eval-ckpt in eval mode
            return False
        clock = env.clock
        frame = clock.get()
        frequency = env.eval_frequency if mode == 'eval' else env.log_frequency
        to_ckpt = util.frame_mod(frame, frequency, env.num_envs) or frame == clock.max_frame
        return to_ckpt

    def try_ckpt(self, world, env):
        '''Check then run checkpoint log/eval'''
        if self.to_ckpt(env, 'log'):
            world.ckpt()
            if world.total_rewards_ma >= world.best_total_rewards_ma:
                world.best_total_rewards_ma = world.total_rewards_ma
                world.save(ckpt='best')

            # TODO support eval
            if ps.get(self.spec, 'meta.rigorous_eval') and self.to_ckpt(env, 'eval'):
                logger.info('Running eval ckpt')

                analysis.gen_avg_return(world, self.eval_env)
                for body in world.bodies:
                    body.ckpt(self.eval_env, 'eval')
                    body.log_summary('eval')
                    if len(body.eval_df) > 2:  # need more rows to calculate metrics
                        metrics = analysis.analyze_session(self.spec, body.eval_df, 'eval', plot=False)
                        body.log_metrics(metrics['scalar'], 'eval')

    def run_rl(self):
        '''Run the main RL loop until clock.max_frame'''
        logger.info(f'Running RL loop for trial {self.spec["meta"]["trial"]} session {self.index}')
        clock = self.env.clock
        logger.debug("first reset env")
        state = self.env.reset()
        done = False
        while True:
            if util.epi_done(done):  # before starting another episode
                if clock.get() < clock.max_frame:  # reset and continue
                    clock.tick('epi')
                    state = self.env.reset()
                    done = False
            self.try_ckpt(self.world, self.env)
            if clock.get() >= clock.max_frame:  # finish
                break
            clock.tick('t')
            with torch.no_grad():
                action = self.world.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.world.update(state, action, reward, next_state, done)
            state = next_state

    def close(self):
        '''Close session and clean up. Save agent, close env.'''
        self.world.close()
        self.env.close()
        self.eval_env.close()
        torch.cuda.empty_cache()
        logger.info(f'Session {self.index} done')

    def run(self):
        self.run_rl()
        world_and_agents_metrics = self.world.compute_and_log_session_metrics()
        self.close()
        return world_and_agents_metrics


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

    def parallelize_sessions(self, global_nets_list=None):
        mp_dict = mp.Manager().dict()
        workers = []
        spec = deepcopy(self.spec)
        for _s in range(spec['meta']['max_session']):
            spec_util.tick(spec, 'session')
            w = mp.Process(target=mp_run_session, args=(spec, global_nets_list, mp_dict))
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
            sessions_metrics = [Session(spec).run()]
        else:
            sessions_metrics = self.parallelize_sessions()
        return sessions_metrics

    def init_global_nets(self):
        session = Session(deepcopy(self.spec))
        session.env.close()  # safety
        global_nets_list = net_util.init_global_nets(session.world.algorithms)
        return global_nets_list

    def run_distributed_sessions(self):
        logger.info('Running distributed sessions')
        global_nets_list = self.init_global_nets()
        session_metrics_list = self.parallelize_sessions(global_nets_list)
        return session_metrics_list

    def close(self):
        logger.info(f'Trial {self.index} done')

    def run(self):
        if self.spec['meta'].get('distributed') == False:
            sessions_agents_metrics = self.run_sessions()
        else:
            sessions_agents_metrics = self.run_distributed_sessions()
        trial_agents_metrics = analysis.analyze_trial(self.spec, sessions_agents_metrics)
        self.close()
        return trial_agents_metrics
        # return metrics['scalar']


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
