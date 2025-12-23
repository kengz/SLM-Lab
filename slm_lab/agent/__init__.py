# The agent module
import os
import warnings
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import pydash as ps
import torch
from torch.utils.tensorboard import SummaryWriter

from slm_lab.agent import algorithm, memory
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.net import net_util
from slm_lab.experiment import analysis
from slm_lab.lib import logger, util, viz
from slm_lab.lib.decorator import lab_api
from slm_lab.lib.env_var import lab_mode, log_extra

logger = logger.get_logger(__name__)


class Agent:
    """
    Agent abstraction; implements the API to interface with Env in SLM Lab
    Contains algorithm, memory, tracker
    """

    def __init__(
        self,
        spec: dict[str, Any],
        mt: "MetricsTracker",
        global_nets: Optional[dict[str, Any]] = None,
    ):
        self.spec = spec
        self.agent_spec = spec["agent"]
        self.name = self.agent_spec["name"]
        assert not ps.is_list(global_nets), (
            f"single agent global_nets must be a dict, got {global_nets}"
        )
        # set components
        self.mt = mt
        mt.agent = self
        # Add direct references for simplified access
        self.env = mt.env

        # Move space attributes from Tracker to Agent where they belong
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.is_discrete = self.env.is_discrete
        # Set the ActionPD class for sampling action
        self.action_type = policy_util.get_action_type(self.env)
        self.action_pdtype = ps.get(self.agent_spec, "algorithm.action_pdtype")
        if self.action_pdtype in (None, "default"):
            self.action_pdtype = policy_util.ACTION_PDS[self.action_type][0]
        self.ActionPD = policy_util.get_action_pd_cls(
            self.action_pdtype, self.action_type
        )

        # Initialize algorithm-specific variables with defaults first
        self.explore_var = np.nan  # action exploration: epsilon or tau
        self.entropy_coef = np.nan  # entropy for exploration
        self.entropy = np.nan  # entropy for tracking

        MemoryClass = getattr(memory, ps.get(self.agent_spec, "memory.name"))
        self.memory = MemoryClass(self.agent_spec["memory"], self)
        AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, "algorithm.name"))
        self.algorithm = AlgorithmClass(self, global_nets)

    @lab_api
    def act(self, state: np.ndarray) -> np.ndarray:
        """Standard act method from algorithm."""
        with torch.no_grad():  # for efficiency, only calc grad in algorithm.train
            action = self.algorithm.act(state)
        return action

    @lab_api
    def update(
        self,
        state: np.ndarray,
        action: Union[int, float, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        terminated: bool,
        truncated: bool,
    ) -> Optional[dict[str, float]]:
        """Update per timestep after env transitions"""
        self.mt.update(state, action, reward, next_state, done)
        if util.in_eval_lab_mode():
            return
        self.memory.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            terminated=terminated,
            truncated=truncated
        )
        loss = self.algorithm.train()
        if not np.isnan(loss):  # set for log_summary()
            self.mt.loss = loss
        explore_var = self.algorithm.update()
        return loss, explore_var

    @lab_api
    def save(self, ckpt: Optional[str] = None) -> None:
        """Save agent"""
        if util.in_eval_lab_mode():  # eval does not save new models
            return
        self.algorithm.save(ckpt=ckpt)

    @lab_api
    def close(self) -> None:
        """Close and cleanup agent at the end of a session, e.g. save model"""
        self.save()


class MetricsTracker:
    """
    Metrics tracker for single-agent-single-env architecture.

    Handles training and evaluation metrics collection, logging, and analysis
    data management for RL experiments.
    """

    def __init__(self, env: "gym.Env", spec: dict[str, Any]):
        # essential reference variables
        self.agent = None  # set later
        self.env = env
        self.spec = spec

        # debugging/logging variables, set in train or loss function
        self.loss = np.nan

        # total_reward_ma from eval for model checkpoint saves
        self.best_total_reward_ma = -np.inf
        self.total_reward_ma = np.nan

        # dataframes to track data - start with core columns in priority order
        core_columns = [
            "total_reward",
            "total_reward_ma",
            "loss",
            "fps",
            "frame",
            "wall_t",
            "t",
            "epi",
            "opt_step",
            "lr",
        ]
        self.train_df = pd.DataFrame(columns=core_columns)

        # Dynamic variables registry
        self.algo_vars = {}

        # Register grad_norm if in dev/test mode
        if net_util.to_check_train_step():
            self.register_algo_var("grad_norm", self)

        # in train@ mode, override from saved train_df if exists
        if util.in_train_lab_mode() and self.spec["meta"]["resume"]:
            train_df_filepath = util.get_session_df_path(self.spec, "train")
            if os.path.exists(train_df_filepath):
                self.train_df = util.read(train_df_filepath)
                self.env.load(self.train_df)

        # track eval data within run_eval. the same as train_df except for reward
        if self.spec["meta"]["rigorous_eval"]:
            self.eval_df = self.train_df.copy()
        else:
            self.eval_df = self.train_df

        self.metrics = {}  # store scalar metrics for Ray Tune reporting

    def register_algo_var(self, var_name: str, source_obj: object) -> None:
        """Register a variable for logging. Expects source_obj to have an attribute named var_name."""
        self.algo_vars[var_name] = source_obj

    def update(
        self,
        state: np.ndarray,
        action: Union[int, float, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Interface update method for tracker at agent.update()"""
        if lab_mode() == "dev":  # log tensorboard only on dev mode
            self.track_tensorboard(action)

    def __str__(self) -> str:
        class_attr = util.get_class_attr(self)
        class_attr.pop("spec")
        return f"mt: {util.to_json(class_attr)}"

    def calc_df_row(self, env: "gym.Env") -> pd.Series:
        """Calculate a row for updating train_df or eval_df."""
        frame = self.env.get("frame")
        wall_t = self.env.get_elapsed_wall_t()
        fps = 0 if wall_t == 0 else frame / wall_t
        with warnings.catch_warnings():  # mute np.nanmean warning
            warnings.filterwarnings("ignore")
            total_reward = np.nanmean(env.total_reward)  # guard for vec env

        # Calculate grad_norm only in dev/test mode when training
        if net_util.to_check_train_step():
            grad_norms = net_util.get_grad_norms(self.agent.algorithm)
            if not ps.is_empty(grad_norms):
                self.grad_norm = np.mean(grad_norms)
            else:
                self.grad_norm = np.nan

        # Core metrics in priority order (most important first for display)
        row_data = {
            # t and reward are measured from a given env or eval_env
            "total_reward": total_reward,
            "total_reward_ma": np.nan,  # update outside
            "loss": self.loss,
            "fps": fps,
            # epi and frame are always measured from training env
            "frame": frame,
            "wall_t": wall_t,
            "t": env.get("t"),
            "epi": self.env.get("epi"),
            "opt_step": self.env.get("opt_step"),
            "lr": self.get_mean_lr(),
        }

        # Add all dynamic variables
        for var_name, source_obj in self.algo_vars.items():
            if hasattr(source_obj, var_name):
                row_data[var_name] = getattr(source_obj, var_name)

        # Convert any tensors to scalars
        for key, value in row_data.items():
            if hasattr(value, "cpu"):
                row_data[key] = value.cpu().item()

        row = pd.Series(row_data, dtype=np.float32)

        # Dynamically add any missing columns to dataframe
        for col in row.index:
            if col not in self.train_df.columns:
                self.train_df[col] = np.nan
                if hasattr(self, "eval_df"):
                    self.eval_df[col] = np.nan
        return row

    def get_last_row(self, df_mode: str) -> pd.Series:
        """Get the last row of train_df or eval_df."""
        df = getattr(self, f"{df_mode}_df")
        return df.iloc[-1]

    def ckpt(self, env: "gym.Env", df_mode: str) -> None:
        """
        Checkpoint to update train_df or eval_df data
        @param gym.Env:env self.env or self.eval_env
        @param str:df_mode 'train' or 'eval'
        """
        row = self.calc_df_row(env)
        df = getattr(self, f"{df_mode}_df")
        df.loc[len(df)] = row  # append efficiently to df
        # Use .loc for direct assignment to avoid chained assignment
        total_reward_ma = df[-viz.PLOT_MA_WINDOW :]["total_reward"].mean()
        df.loc[df.index[-1], "total_reward_ma"] = total_reward_ma
        df.drop_duplicates("frame", inplace=True)  # Remove duplicates
        self.total_reward_ma = total_reward_ma

    def get_mean_lr(self) -> float:
        """Gets the average current learning rate of the algorithm's nets."""
        if not hasattr(self.agent.algorithm, "net_names"):
            return np.nan
        lrs = []
        for attr, obj in self.agent.algorithm.__dict__.items():
            if attr.endswith("lr_scheduler"):
                lr = obj.get_last_lr()
                if hasattr(lr, "cpu"):
                    lr = lr.cpu().item()
                elif isinstance(lr, list):
                    lr = lr[0].cpu().item() if hasattr(lr[0], "cpu") else lr[0]
                lrs.append(lr)
        return np.mean(lrs) if lrs else np.nan

    def get_log_prefix(self) -> str:
        """Get the prefix for logging"""
        spec_name = self.spec["name"]
        trial_index = self.spec["meta"]["trial"]
        session_index = self.spec["meta"]["session"]
        prefix = f"Trial {trial_index} session {session_index} {spec_name}_t{trial_index}_s{session_index}"
        return prefix

    def log_metrics(self, metrics: dict[str, float], df_mode: str) -> None:
        """Log session metrics"""
        # Skip extra metrics unless enabled
        if not log_extra():
            return
        prefix = self.get_log_prefix()
        row_str = "  ".join([f"{k}: {v:g}" for k, v in metrics.items()])
        msg = f"{prefix} [{df_mode}_df metrics] {row_str}"
        logger.info(msg)

    def calc_log_metrics(self, spec: dict, df_mode: str) -> dict:
        """Calculate session metrics and store them for Ray Tune reporting."""
        from slm_lab.experiment.search import BASE_SCHEDULER_SPEC

        df = getattr(self, f"{df_mode}_df")
        if len(df) <= 2:
            return self.metrics

        time_attr, metric = ps.at(BASE_SCHEDULER_SPEC, "time_attr", "metric")
        self.metrics = ps.pick(self.get_last_row(df_mode).to_dict(), time_attr, metric)

        session_metrics = analysis.analyze_session(spec, df, df_mode, plot=False)
        self.metrics |= session_metrics["scalar"]

        if log_extra():
            self.log_metrics(self.metrics, df_mode)

        return self.metrics

    def log_summary(self, df_mode: str) -> None:
        """Log the summary for this tracker when its environment is done"""
        # Skip periodic logging in Ray Tune to reduce noise (file logs still capture everything)
        from slm_lab.experiment.search import in_ray_tune_context
        if in_ray_tune_context():
            return

        prefix = self.get_log_prefix()
        last_row = self.get_last_row(df_mode)
        items = util.format_metrics(last_row)

        # Simple grid: 4 per line with equal spacing
        w = max(len(item) for item in items)
        lines = [f"{prefix} [{df_mode}]"]
        for i in range(0, len(items), 4):
            # Pad items except the last one in each row to avoid trailing spaces
            chunk = items[i : i + 4]
            formatted = [f"{item:<{w}}" if j < len(chunk) - 1 else item for j, item in enumerate(chunk)]
            lines.append("  ".join(formatted))

        logger.info("\n".join(lines))
        if (
            lab_mode() == "dev" and df_mode == "train"
        ):  # log tensorboard only on dev mode and train df data
            self.log_tensorboard()

    def log_tensorboard(self) -> None:
        """
        Log summary and useful info to TensorBoard.
        NOTE this logging is comprehensive and memory-intensive, hence it is used in dev mode only
        """
        # initialize TensorBoard writer
        if not hasattr(self, "tb_writer"):
            log_prepath = self.spec["meta"]["log_prepath"]
            self.tb_writer = SummaryWriter(
                os.path.dirname(log_prepath),
                filename_suffix=os.path.basename(log_prepath),
            )
            self.tb_actions = []  # store actions for tensorboard
            logger.info(
                f"Using TensorBoard logging for dev mode. Run `tensorboard --logdir={log_prepath}` to start TensorBoard."
            )

        trial_index = self.spec["meta"]["trial"]
        session_index = self.spec["meta"]["session"]
        if session_index != 0:  # log only session 0
            return
        idx_suffix = f"trial{trial_index}_session{session_index}"
        frame = self.env.get("frame")
        # add main graph
        if self.env.get("frame") == 0 and hasattr(self.agent.algorithm, "net"):
            # can only log 1 net to tb now, and 8 is a good common length for stacked and rnn inputs
            net = self.agent.algorithm.net
            self.tb_writer.add_graph(net, torch.rand(ps.flatten([8, net.in_dim])))
        # add summary variables
        last_row = self.get_last_row("train")
        for k, v in last_row.items():
            self.tb_writer.add_scalar(f"{k}/{idx_suffix}", v, frame)
        # add network parameters
        for net_name in self.agent.algorithm.net_names:
            if net_name.startswith("global_") or net_name.startswith("target_"):
                continue
            net = getattr(self.agent.algorithm, net_name)
            for name, params in net.named_parameters():
                self.tb_writer.add_histogram(
                    f"{net_name}.{name}/{idx_suffix}", params, frame
                )
        # add action histogram and flush
        if not ps.is_empty(self.tb_actions):
            actions = np.array(self.tb_actions)
            if len(actions.shape) == 1:
                self.tb_writer.add_histogram(f"action/{idx_suffix}", actions, frame)
            else:  # multi-action
                for idx, subactions in enumerate(actions.T):
                    self.tb_writer.add_histogram(
                        f"action.{idx}/{idx_suffix}", subactions, frame
                    )
            self.tb_actions = []

    def track_tensorboard(self, action: np.ndarray) -> None:
        """Helper to track variables for tensorboard logging"""
        if not hasattr(self, "tb_actions"):
            self.tb_actions = []
        if self.env.is_venv:
            self.tb_actions.extend(action.tolist())
        else:
            self.tb_actions.append(action)
