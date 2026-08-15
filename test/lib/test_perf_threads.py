"""Torch must get its SHARE of the cores, not all of them, once sessions run concurrently.

WHY THIS EXISTS. `_perf_cpu_threads` runs inside every session process, and `Trial` runs
`meta.max_session` of them at once. Sizing the pool from `os.cpu_count()` gave each session the
whole machine. On a 4-core box the stock `benchmark_arc/ppo/ppo_mujoco_arc.yaml` (`max_session: 4`)
put 4 torch threads in each of 4 processes; OpenMP's spin-wait then burned the cores the env
workers needed.

Measured, PPO Hopper on that box: **10.5 frames/s per session before, ~500 after** — the
optimisation was costing ~50x on exactly the configuration it exists to speed up.
"""
from unittest import mock

import pytest

from slm_lab.lib import perf


@pytest.fixture
def cpu_bound(monkeypatch):
    """No CUDA, optimisation enabled — the branch the fix lives on."""
    monkeypatch.setattr(perf.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(perf, "optimize_perf", lambda: True)


@pytest.mark.parametrize("n_cpu,n_concurrent,expected", [
    (4, 4, 1),    # the defect: 4 sessions on 4 cores got 4 threads each, now 1
    (4, 1, 4),    # a single session still gets the whole box
    (4, 8, 1),    # more sessions than cores never goes below 1
    (64, 4, 16),  # a big box divides evenly
    (64, 1, 32),  # the 32-thread cap still holds
])
def test_threads_are_divided_by_concurrent_sessions(cpu_bound, monkeypatch, n_cpu, n_concurrent, expected):
    monkeypatch.setattr(perf.os, "cpu_count", lambda: n_cpu)
    with mock.patch.object(perf.torch, "set_num_threads") as set_threads, \
            mock.patch.object(perf.torch, "get_num_threads", return_value=999):
        perf._perf_cpu_threads(n_concurrent)
    set_threads.assert_called_once_with(expected)


def test_it_can_now_LOWER_the_thread_count(cpu_bound, monkeypatch):
    """The old guard was `if current < optimal`, so it could only ever raise.

    That is why the defect was unreachable from inside a run: torch's default was already at or
    above the value the routine wanted, so the routine looked like a no-op while the damage came
    from it never lowering.
    """
    monkeypatch.setattr(perf.os, "cpu_count", lambda: 4)
    with mock.patch.object(perf.torch, "set_num_threads") as set_threads, \
            mock.patch.object(perf.torch, "get_num_threads", return_value=4):
        perf._perf_cpu_threads(n_concurrent=4)
    set_threads.assert_called_once_with(1)


def test_it_is_a_no_op_when_already_correct(cpu_bound, monkeypatch):
    monkeypatch.setattr(perf.os, "cpu_count", lambda: 4)
    with mock.patch.object(perf.torch, "set_num_threads") as set_threads, \
            mock.patch.object(perf.torch, "get_num_threads", return_value=1):
        perf._perf_cpu_threads(n_concurrent=4)
    set_threads.assert_not_called()


def test_default_is_unchanged_behaviour_for_a_single_session(cpu_bound, monkeypatch):
    """Callers that pass nothing keep the original semantics."""
    monkeypatch.setattr(perf.os, "cpu_count", lambda: 8)
    with mock.patch.object(perf.torch, "set_num_threads") as set_threads, \
            mock.patch.object(perf.torch, "get_num_threads", return_value=1):
        perf._perf_cpu_threads()
    set_threads.assert_called_once_with(8)
