"""torch.profiler Kineto trace collection for HTA analysis.

Collects GPU/CPU traces when --profile is enabled. Traces are saved as
gzipped TensorBoard files compatible with Holistic Trace Analysis (HTA).
"""

import os
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler

from slm_lab.lib import logger
from slm_lab.lib.env_var import profile

logger = logger.get_logger(__name__)


def _get_trace_dir() -> Path:
    """Resolve trace output directory from LOG_PREPATH."""
    log_prepath = os.environ.get("LOG_PREPATH", "data/profiler")
    return Path(log_prepath).parent.parent / "traces"


def create_torch_profiler() -> torch.profiler.profile | None:
    """Create a torch.profiler.profile instance for Kineto trace collection.

    Returns None if profiling is not enabled.

    Schedule is configurable via env vars (useful for different algorithms):
      PROF_SKIP: steps to skip before profiling (default 500)
      PROF_ACTIVE: steps to actively record (default 20)
    PPO needs high skip (trains every time_horizon ~2048 steps).
    SAC/CrossQ need moderate skip (train every step after training_start_step).
    """
    if not profile():
        return None

    trace_dir = _get_trace_dir()
    trace_dir.mkdir(parents=True, exist_ok=True)

    skip_first = int(os.environ.get("PROF_SKIP", "500"))
    active = int(os.environ.get("PROF_ACTIVE", "20"))
    logger.info(f"Torch profiler traces: {trace_dir} (skip={skip_first}, active={active})")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        schedule=schedule(skip_first=skip_first, wait=5, warmup=2, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir), use_gzip=True),
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    )


@contextmanager
def torch_profiler_context():
    """Context manager that wraps the RL loop with torch.profiler.

    Yields a step callback to call after each iteration.
    If profiling is disabled, yields a no-op callable.
    """
    prof = create_torch_profiler()
    if prof is None:
        yield lambda: None
        return

    with prof:
        yield prof.step
