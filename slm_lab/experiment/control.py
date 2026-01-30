# The control module
# Creates and runs control loops at levels: Experiment, Trial, Session
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pydash as ps
import torch
import torch.multiprocessing as mp

from slm_lab.agent import Agent, MetricsTracker
from slm_lab.agent.net import net_util
from slm_lab.env import make_env
from slm_lab.experiment import analysis, search
from slm_lab.lib import logger, util
from slm_lab.lib.env_var import lab_mode
from slm_lab.lib.perf import log_perf_setup, optimize
from slm_lab.spec import spec_util


def make_agent_env(spec, global_nets=None):
    """Helper to create agent and env given spec"""
    env = make_env(spec)
    mt = MetricsTracker(env, spec)
    agent = Agent(spec, mt=mt, global_nets=global_nets)
    return agent, env


def mp_run_session(spec, global_nets, mp_dict):
    """Wrap for multiprocessing with shared variable"""
    session = Session(spec, global_nets)
    metrics = session.run()
    mp_dict[session.index] = metrics


class Session:
    """
    The base lab unit to run a RL session for a spec.
    Given a spec, it creates the agent and env, runs the RL loop,
    then gather data and analyze it to produce session data.
    """

    def __init__(self, spec: dict, global_nets=None):
        self.spec = spec
        self.index = self.spec["meta"]["session"]
        util.set_random_seed(self.spec)
        util.set_cuda_id(self.spec)
        util.set_logger(self.spec, logger, "session")

        # Apply perf optimizations for all sessions
        self.perf_setup = optimize()

        self.agent, self.env = make_agent_env(self.spec, global_nets)
        if ps.get(self.spec, "meta.rigorous_eval"):
            with util.ctx_lab_mode("eval"):
                self.eval_env = make_env(self.spec)
        else:
            self.eval_env = self.env
        if self.index == 0:
            log_perf_setup()
            util.log_self_desc(
                self.agent.algorithm, omit=["net_spec", "explore_var_spec"]
            )

    def to_ckpt(self, env: gym.Env, mode: str = "eval") -> bool:
        """Check with clock whether to run log/eval ckpt: at the start, save_freq, and the end"""
        if (
            mode == "eval" and util.in_eval_lab_mode()
        ):  # avoid double-eval: eval-ckpt in eval mode
            return False
        # ClockWrapper provides direct access to clock methods
        frame = env.get()
        frequency = env.eval_frequency if mode == "eval" else env.log_frequency
        is_final_frame = frame == env.max_frame
        to_ckpt = util.frame_mod(frame, frequency, env.num_envs) or is_final_frame
        return to_ckpt

    def try_ckpt(self, agent: Agent, env: gym.Env):
        """Check then run checkpoint log/eval"""
        mt = agent.mt
        if self.to_ckpt(env, "log"):
            mt.ckpt(self.env, "train")
            mt.log_summary("train")
            mt.calc_log_metrics(self.spec, "train")
            search.report(mt)

            agent.save()
            if mt.total_reward_ma >= mt.best_total_reward_ma:
                mt.best_total_reward_ma = mt.total_reward_ma
                agent.save(ckpt="best")

            # Plot: session graphs and trial graphs at checkpoints
            if len(mt.train_df) > 2:
                analysis.analyze_session(self.spec, mt.train_df, "train", plot=True)
                if self.index == 0:
                    analysis.analyze_trial(self.spec)

        if ps.get(self.spec, "meta.rigorous_eval") and self.to_ckpt(env, "eval"):
            logger.info("Running eval ckpt")
            analysis.gen_avg_return(agent, self.eval_env)
            mt.ckpt(self.eval_env, "eval")
            mt.log_summary("eval")
            mt.calc_log_metrics(self.spec, "eval")

    def run_rl(self):
        """Run the main RL loop until clock.max_frame"""
        state, info = self.env.reset()

        while self.env.get() < self.env.max_frame:
            with torch.no_grad():
                action = self.agent.act(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            done = np.logical_or(terminated, truncated)
            self.agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                terminated=terminated,
                truncated=truncated
            )
            self.try_ckpt(self.agent, self.env)

            if util.epi_done(done):
                state, info = self.env.reset()
            else:
                state = next_state

    def close(self):
        """Close session and clean up. Save agent, close env."""
        self.agent.close()
        self.env.close()
        self.eval_env.close()
        torch.cuda.empty_cache()
        logger.info(f"Session {self.index} done")

    def run(self):
        self.run_rl()
        metrics = analysis.analyze_session(self.spec, self.agent.mt.eval_df, "eval")
        self.agent.mt.log_metrics(metrics["scalar"], "eval")
        self.close()
        return metrics


class Trial:
    """
    The lab unit which runs repeated sessions for a same spec, i.e. a trial
    Given a spec and number s, trial creates and runs s sessions,
    then gathers session data and analyze it to produce trial data.
    """

    def __init__(self, spec):
        self.spec = spec
        self.index = self.spec["meta"]["trial"]
        util.set_logger(self.spec, logger, "trial")
        spec_util.save(spec, unit="trial")

    def parallelize_sessions(self, global_nets=None):
        mp_dict = mp.Manager().dict()
        workers = []
        spec = deepcopy(self.spec)
        for _s in range(spec["meta"]["max_session"]):
            spec_util.tick(spec, "session")
            w = mp.Process(target=mp_run_session, args=(spec, global_nets, mp_dict))
            w.start()
            workers.append(w)
        for w in workers:
            w.join()
        session_metrics_list = [mp_dict[idx] for idx in sorted(mp_dict.keys())]
        return session_metrics_list

    def run_sessions(self):
        max_session = self.spec["meta"]["max_session"]
        if max_session == 1:
            spec = deepcopy(self.spec)
            spec_util.tick(spec, "session")
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
        logger.info("Running distributed sessions")
        global_nets = self.init_global_nets()
        session_metrics_list = self.parallelize_sessions(global_nets)
        return session_metrics_list

    def close(self):
        logger.info(f"Trial {self.index} done")

    def run(self):
        if self.spec["meta"].get("distributed") is False:
            session_metrics_list = self.run_sessions()
        else:
            session_metrics_list = self.run_distributed_sessions()
        metrics = analysis.analyze_trial(self.spec, session_metrics_list)
        # Log final trial metrics for easy extraction from dstack logs
        logger.info(f"trial_metrics: {' | '.join(util.format_metrics(metrics['scalar']))}")
        self.close()
        return metrics["scalar"]


class Experiment:
    """
    The lab unit to run experiments.
    It generates a list of specs to search over, then run each as a trial with s repeated session,
    then gathers trial data and analyze it to produce experiment data.
    """

    def __init__(self, spec, keep_trials: int = 3):
        self.spec = spec
        self.index = self.spec["meta"]["experiment"]
        self.keep_trials = keep_trials
        util.set_logger(self.spec, logger, "experiment")
        spec_util.save(spec, unit="experiment")

    def close(self):
        logger.info("Experiment done")

    def run(self):
        trial_data_dict = search.run_ray_search(self.spec)
        experiment_df = analysis.analyze_experiment(self.spec, trial_data_dict)
        # Cleanup trial models after search to reduce disk usage (if keep_trials != -1)
        if self.keep_trials != -1:
            search.cleanup_trial_models(self.spec, experiment_df, keep_top_n=self.keep_trials)
        self.close()
        return experiment_df
