"""
Clean SLM-Lab profiler: cpu, ram, gpu, vram, time tracking with single visualization
Activate with --profile flag: slm-lab --profile spec.json spec_name mode
"""

import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, Thread

import GPUtil
import plotly.graph_objects as go
import psutil
from plotly.subplots import make_subplots

from slm_lab.lib import logger, util
from slm_lab.lib.env_var import profile

logger = logger.get_logger(__name__)

# Global profiler instance
PROFILER = None
LOCK = None
CLEANUP_REGISTERED = False


def _calc_stats(vals):
    """Calculate avg and peak from values."""
    if not vals:
        return {"avg": 0, "peak": 0}
    return {"avg": sum(vals) / len(vals), "peak": max(vals)}


def _log_profiler_setup():
    """Log comprehensive profiler setup information."""
    from slm_lab.lib.env_var import lab_mode
    
    lines = ["Profiler setup:"]
    
    # Mode info if forced to dev
    current_mode = lab_mode()
    if current_mode == "dev":
        lines.append("• Mode: dev (required for profiling)")
    
    # System monitoring
    lines.append("• Monitoring: CPU, memory, GPU utilization, function timing")
    
    # Output info
    log_path = Path(os.environ.get("LOG_PREPATH", "data/profiler"))
    out_dir = log_path.parent.parent / "profiler" if "LOG_PREPATH" in os.environ else Path("data/profiler")
    lines.append(f"• Results: {out_dir}/profiling_dashboard.html")
    
    logger.info('\n'.join(lines))


def get_profiler(spec=None):
    """Get global profiler instance if enabled."""
    global PROFILER, LOCK, CLEANUP_REGISTERED
    if not profile():
        return None

    if LOCK is None:
        LOCK = Lock()

    with LOCK:
        if PROFILER is None:
            PROFILER = Profiler(spec=spec)
            _log_profiler_setup()
        if not CLEANUP_REGISTERED:
            import atexit

            atexit.register(cleanup_profiler)
            CLEANUP_REGISTERED = True
        return PROFILER


def cleanup_profiler():
    """Clean up global profiler and save results."""
    global PROFILER, LOCK
    if LOCK and PROFILER:
        with LOCK:
            PROFILER.save_results()
            PROFILER.stop()
            PROFILER = None


class Profiler:
    """Simple profiler for cpu, ram, gpu, vram, time tracking on @lab_api functions."""

    def __init__(self, spec=None):
        self.spec = spec
        self.func_stats = defaultdict(list)
        self.keys = ["timestamps", "cpu", "memory", "gpu_util", "gpu_memory"]
        self.sys_stats = {k: deque(maxlen=2000) for k in self.keys}
        self.process = psutil.Process()
        self.active = True
        self.lock = Lock()
        Thread(target=self._monitor, daemon=True).start()

    def _monitor(self):
        while self.active:
            stats = [
                time.time(),
                self.process.cpu_percent(),
                self.process.memory_info().rss / 1024 / 1024,
                0,
                0,
            ]
            gpus = GPUtil.getGPUs()
            if gpus:
                stats[3] = gpus[0].load * 100
                stats[4] = gpus[0].memoryUsed
            with self.lock:
                for i, k in enumerate(self.keys):
                    self.sys_stats[k].append(stats[i])
            time.sleep(0.5)

    @contextmanager
    def profile_function(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            dur = (time.perf_counter() - start) * 1000
            mem_delta = 0
            if dur > 1.0:
                cur_mem = self.process.memory_info().rss / 1024 / 1024
                with self.lock:
                    if self.sys_stats["memory"]:
                        mem_delta = cur_mem - self.sys_stats["memory"][-1]

            with self.lock:
                cpu = list(self.sys_stats["cpu"])[-10:]
                gpu = list(self.sys_stats["gpu_util"])[-10:]
                self.func_stats[name].append(
                    {
                        "duration_ms": dur,
                        "memory_delta_mb": mem_delta,
                        "cpu_percent": sum(cpu) / len(cpu) if cpu else 0,
                        "gpu_util_percent": sum(gpu) / len(gpu) if gpu else 0,
                        "gpu_memory_delta_mb": 0,
                        "timestamp": time.time(),
                    }
                )

    def get_summary(self):
        with self.lock:
            sys = {}
            for name, key in [
                ("cpu", "cpu"),
                ("memory_mb", "memory"),
                ("gpu_util", "gpu_util"),
                ("gpu_memory_mb", "gpu_memory"),
            ]:
                for k, v in _calc_stats(self.sys_stats[key]).items():
                    sys[f"{name}_{k}"] = v

            funcs = {}
            for name, calls in self.func_stats.items():
                durs = [c["duration_ms"] for c in calls]
                funcs[name] = {
                    "call_count": len(calls),
                    "total_duration_ms": sum(durs),
                    "avg_duration_ms": sum(durs) / len(durs),
                    "max_duration_ms": max(durs),
                    "min_duration_ms": min(durs),
                }
                for k in [
                    "memory_delta_mb",
                    "cpu_percent",
                    "gpu_util_percent",
                    "gpu_memory_delta_mb",
                ]:
                    funcs[name][f"avg_{k}"] = sum(c[k] for c in calls) / len(calls)
                for k in ["memory_delta_mb", "gpu_memory_delta_mb"]:
                    funcs[name][f"total_{k}"] = sum(c[k] for c in calls)
            return {"system": sys, "functions": funcs}

    def create_visualization(self, out_dir):
        with self.lock:
            times = [
                (t - self.sys_stats["timestamps"][0])
                for t in self.sys_stats["timestamps"]
            ]
            top_funcs = sorted(
                self.func_stats.items(),
                key=lambda x: sum(c["duration_ms"] for c in x[1]),
                reverse=True,
            )[:10]

        fig = make_subplots(
            rows=5,
            cols=2,
            subplot_titles=[
                "System CPU %",
                "Function CPU %",
                "System Memory MB",
                "Function Memory MB",
                "System GPU %",
                "Function GPU %",
                "System GPU Memory MB",
                "Function GPU Memory MB",
                "Function Total Time (s)",
                "Function Avg Time (ms)",
            ],
        )

        # Left column: System aggregate plots
        for i, (k, title) in enumerate(
            [
                ("cpu", "CPU"),
                ("memory", "Memory"),
                ("gpu_util", "GPU"),
                ("gpu_memory", "GPU Mem"),
            ]
        ):
            fig.add_trace(
                go.Scatter(x=times, y=list(self.sys_stats[k]), name=title),
                row=i + 1,
                col=1,
            )

        # Right column: Function breakdown plots
        if top_funcs:
            names = [f[:10] for f, _ in top_funcs]
            for i, metric in enumerate(
                [
                    "cpu_percent",
                    "memory_delta_mb",
                    "gpu_util_percent",
                    "gpu_memory_delta_mb",
                ]
            ):
                values = [
                    sum(c[metric] for c in calls) / len(calls) for _, calls in top_funcs
                ]
                fig.add_trace(go.Bar(x=names, y=values), row=i + 1, col=2)

            # Total time (s) vs average time (ms)
            total_times = [
                sum(c["duration_ms"] for c in calls) / 1000.0 for _, calls in top_funcs
            ]
            avg_times = [
                sum(c["duration_ms"] for c in calls) / len(calls)
                for _, calls in top_funcs
            ]
            fig.add_trace(go.Bar(x=names, y=total_times), row=5, col=1)
            fig.add_trace(go.Bar(x=names, y=avg_times), row=5, col=2)

        fig.update_layout(
            height=1200,
            title_text="SLM-Lab Performance Profiler: System Resources & Function Analysis",
            showlegend=False,
        )
        
        # Rotate x-axis labels for better readability of function names
        fig.update_xaxes(tickangle=45)
        fig.write_html(Path(out_dir) / "profiling_dashboard.html")

    def save_results(self, out_dir=None):
        if out_dir is None:
            log_path = Path(os.environ["LOG_PREPATH"])
            out_dir = log_path.parent.parent / "profiler"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        util.write(self.get_summary(), Path(out_dir) / "profiler_summary.json")
        self.create_visualization(out_dir)

    def stop(self):
        self.active = False
