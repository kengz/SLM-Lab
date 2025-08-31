"""
Performance optimization utilities for SLM-Lab.
Centralizes all perf optimizations in one module.
"""

import os
import platform

import torch

from slm_lab.lib import logger

logger = logger.get_logger(__name__)


def optimize_perf():
    """Apply all perf optimizations based on environment variables and hardware detection."""

    # 1. CPU Threading Optimization
    _perf_cpu_threads()

    # 2. PyTorch Compile Optimization
    _perf_torch_compile()

    # 3. Platform-specific GPU optimizations
    _perf_gpu()

    # 4. Memory optimizations
    _perf_memory()

    return _get_perf_status()


def _perf_cpu_threads():
    """Optimize CPU threading for all platforms."""
    if torch.cuda.is_available():
        return  # Skip CPU optimization if GPU is primary compute

    optimize_perf = os.getenv("OPTIMIZE_PERF", "true").lower() == "true"
    if not optimize_perf:
        logger.info("CPU optimization disabled via OPTIMIZE_PERF=false")
        return

    current_threads = torch.get_num_threads()
    cpu_count = os.cpu_count() or 1

    # Intelligent threading: use all cores but cap at reasonable limit
    optimal_threads = min(cpu_count, 32)  # Cap at 32 to avoid diminishing returns

    if current_threads < optimal_threads:
        torch.set_num_threads(optimal_threads)
        # Set environment variables for other libraries
        for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLAS_NUM_THREADS"]:
            os.environ.setdefault(var, str(optimal_threads))
        logger.info(
            f"CPU optimization: {current_threads} → {optimal_threads} threads ({cpu_count} cores)"
        )


def _perf_torch_compile():
    """Apply torch.compile optimizations based on hardware."""
    # Respect optimize_perf setting first
    optimize_perf = os.getenv("OPTIMIZE_PERF", "true").lower() == "true"
    if not optimize_perf:
        return False
        
    compile_mode = os.getenv("TORCH_COMPILE", "auto").lower()

    # Skip on Apple Silicon CPU (known instability)
    is_apple_cpu = (
        platform.machine() == "arm64"
        and platform.system() == "Darwin"
        and not torch.cuda.is_available()
    )

    if compile_mode == "true" and not is_apple_cpu:
        return True  # Will be applied in network initialization
    elif compile_mode == "auto" and torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:  # Only Ampere+ for stability
                return True
        except Exception:
            pass

    return False


def _perf_gpu():
    """Apply GPU-specific optimizations."""
    if not torch.cuda.is_available():
        return

    try:
        major, minor = torch.cuda.get_device_capability()
        device_name = torch.cuda.get_device_name()

        # Enable TF32 on Ampere+ for speed (minimal precision loss for RL)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        logger.info(
            f"GPU optimization applied: {device_name} (compute {major}.{minor})"
        )

    except Exception as e:
        logger.warning(f"GPU optimization failed: {e}")


def _perf_memory():
    """Apply memory optimizations."""
    # Simple optimizations that are generally safe
    if torch.cuda.is_available():
        torch.backends.cuda.max_split_size_mb = 128  # Reduce memory fragmentation


def _get_perf_status():
    """Get current perf optimization status for logging."""
    # Check if torch.compile will actually be enabled (respects hierarchy)
    compile_will_be_enabled = _perf_torch_compile()
    profile_enabled = os.getenv("PROFILE", "false").lower() == "true"
    cpu_count = os.cpu_count() or 1

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        major, minor = torch.cuda.get_device_capability()
        platform_info = f"GPU {device_name} (compute {major}.{minor})"
    else:
        cpu_info = f"{platform.processor()} ({platform.machine()})"
        platform_info = f"CPU {cpu_info}"

    return {
        "torch_compile": "enabled" if compile_will_be_enabled else "disabled",
        "profiler": "enabled" if profile_enabled else "disabled",
        "platform": platform_info,
        "threads": torch.get_num_threads(),
        "cpu_cores": cpu_count,
    }
