"""Centralized environment variable configuration for CLI flags."""
import os


def profile():
    """Check if --profile flag is enabled."""
    return os.getenv("PROFILE", "false").lower() == "true"


def render():
    """Check if --render flag is enabled."""
    return os.environ.get('RENDER', 'false') == 'true'


def optimize_perf():
    """Check if --optimize-perf flag is enabled."""
    return os.getenv("OPTIMIZE_PERF", "true").lower() == "true"


def torch_compile():
    """Get --torch-compile flag value."""
    return os.getenv("TORCH_COMPILE", "auto").lower()


def lab_mode():
    """Get current lab mode."""
    return os.environ.get('lab_mode')