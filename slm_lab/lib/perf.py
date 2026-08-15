"""Performance optimization utilities for SLM-Lab."""

import os
import platform

import torch

from slm_lab.lib import logger
from slm_lab.lib.env_var import optimize_perf, profile

logger = logger.get_logger(__name__)


def optimize(n_concurrent: int = 1):
    """Apply all perf optimizations. `n_concurrent` = sessions sharing this machine."""
    _perf_cpu_threads(n_concurrent)
    _perf_gpu()
    _perf_memory()
    return _get_perf_status()


def _perf_cpu_threads(n_concurrent: int = 1):
    """Give torch its SHARE of the cores, not all of them.

    This runs inside every session process, and a trial runs `meta.max_session` of them at once.
    Sizing the thread pool from `os.cpu_count()` therefore hands each session the whole machine:
    with the stock `max_session: 4` benchmark specs on a 4-core box, four processes each claimed
    4 threads, and OpenMP's spin-wait burned the cores that the env workers needed.

    Measured on that box, PPO Hopper: 10.5 frames/s per session before, ~500 after. The
    optimisation was costing ~50x on the configuration it was meant to help.
    """
    if torch.cuda.is_available() or not optimize_perf():
        return

    cpu_count = os.cpu_count() or 1
    optimal = max(1, min(cpu_count // max(1, n_concurrent), 32))

    if torch.get_num_threads() != optimal:
        torch.set_num_threads(optimal)
        for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLAS_NUM_THREADS"]:
            os.environ.setdefault(var, str(optimal))




def _perf_gpu():
    """Apply GPU optimizations."""
    if not torch.cuda.is_available() or not optimize_perf():
        return

    try:
        major, _ = torch.cuda.get_device_capability()

        if major >= 8:  # Ampere+
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        torch.backends.cudnn.benchmark = True

    except Exception as e:
        logger.warning(f"GPU optimization failed: {e}")


def _perf_memory():
    """Apply memory optimizations."""
    if torch.cuda.is_available() and optimize_perf():
        torch.backends.cuda.max_split_size_mb = 128


def _get_perf_status():
    """Get comprehensive performance status including what was optimized."""
    prof_enabled = profile()
    perf_enabled = optimize_perf()
    cpu_count = os.cpu_count() or 1
    current_threads = torch.get_num_threads()

    # Platform info and optimization details
    optimizations = []

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        major, minor = torch.cuda.get_device_capability()
        platform_info = f"GPU {device_name} (compute {major}.{minor})"

        # GPU optimizations
        if perf_enabled:
            gpu_opts = ["cuDNN benchmark", "memory management"]
            if major >= 8:  # Ampere+
                gpu_opts.append("TF32 acceleration")
            optimizations.extend([f"GPU {opt}: enabled" for opt in gpu_opts])
    else:
        cpu_info = f"{platform.processor()} ({platform.machine()})"
        platform_info = f"CPU {cpu_info}"

        # CPU optimizations - show before/after for threads
        if perf_enabled:
            optimal_threads = min(cpu_count, 32)
            if current_threads >= optimal_threads and current_threads > 4:
                optimizations.append(f"CPU threads: 4 → {current_threads} (optimized)")
            else:
                optimizations.append(f"CPU threads: {current_threads}")

    status = {"platform": platform_info}

    if prof_enabled:
        optimizations.append("profiler: enabled")

    if not perf_enabled:
        optimizations = ["disabled (--no-optimize-perf)"]

    status["optimizations"] = optimizations
    return status


def log_perf_setup():
    """Log performance setup."""
    status = _get_perf_status()
    lines = ["Performance setup:"]
    lines.append(f"Platform: {status['platform']}")

    for opt in status["optimizations"]:
        lines.append(f"• {opt}")

    logger.info("\n".join(lines))
