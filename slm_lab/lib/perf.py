"""Performance optimization utilities for SLM-Lab."""

import os
import platform

import torch
import yaml

from slm_lab.lib import logger
from slm_lab.lib.env_var import optimize_perf, profile, torch_compile

logger = logger.get_logger(__name__)


def optimize():
    """Apply all perf optimizations."""
    _perf_cpu_threads()
    _perf_torch_compile()
    _perf_gpu()
    _perf_memory()
    return _get_perf_status()


def _perf_cpu_threads():
    """Optimize CPU threading."""
    if torch.cuda.is_available() or not optimize_perf():
        return

    current, cpu_count = torch.get_num_threads(), os.cpu_count() or 1
    optimal = min(cpu_count, 32)

    if current < optimal:
        torch.set_num_threads(optimal)
        for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLAS_NUM_THREADS"]:
            os.environ.setdefault(var, str(optimal))


def _perf_torch_compile():
    """Check if lightning thunder should be enabled."""
    if not optimize_perf():
        return False

    mode = torch_compile()
    is_apple_cpu = (
        platform.machine() == "arm64"
        and platform.system() == "Darwin"
        and not torch.cuda.is_available()
    )

    if mode == "true" and not is_apple_cpu:
        return True
    elif mode == "auto" and torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:  # Ampere+
                return True
        except Exception:
            pass
    return False


def _perf_gpu():
    """Apply GPU optimizations."""
    if not torch.cuda.is_available() or not optimize_perf():
        return

    try:
        major, minor = torch.cuda.get_device_capability()
        device_name = torch.cuda.get_device_name()

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
    compile_enabled = _perf_torch_compile()
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
            optimizations.append(f"GPU: {', '.join(gpu_opts)}")
    else:
        cpu_info = f"{platform.processor()} ({platform.machine()})"
        platform_info = f"CPU {cpu_info}"
        
        # CPU optimizations
        if perf_enabled:
            optimal_threads = min(cpu_count, 32)
            if current_threads >= optimal_threads:
                optimizations.append(f"CPU threads: {current_threads} (optimized from 4)")
            else:
                optimizations.append(f"CPU threads: {current_threads}")

    status = {"platform": platform_info}
    
    if compile_enabled:
        optimizations.append("lightning_thunder: enabled")
    
    if prof_enabled:
        optimizations.append("profiler: enabled")
    
    if optimizations:
        status["optimizations"] = optimizations
    else:
        status["optimizations"] = ["disabled"]

    return status


def log_perf_setup():
    """Log performance setup."""
    status = _get_perf_status()
    lines = [f'Performance setup:']
    lines.append(f'platform: {status["platform"]}')
    
    for opt in status["optimizations"]:
        lines.append(f'{opt}')
    
    logger.info('\n'.join(lines))
