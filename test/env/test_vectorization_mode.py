"""`async` must not be chosen when the workers cannot fit on the box.

WHY THIS EXISTS. Every `async` worker is a process that pays a full import of this package —
measured at ~450 MB RSS. The old rule chose `async` for any non-simple env with `num_envs >= 8`,
counting neither cores nor the `max_session` copies of the whole arrangement that run concurrently.
On a 4-core, 15 GB box the stock `benchmark_arc/ppo/ppo_mujoco_arc.yaml` (`num_envs: 16`,
`max_session: 4`) therefore asked for 64 workers and ~28 GB. Measured outcome: 14 GB resident,
13 GB of swap in use, load average 59, and 10.5 frames/s where sync delivers ~1500.

The bug was invisible from inside a run: it looked like a slow framework, not a misconfiguration.
"""
from unittest import mock

import pytest

from slm_lab.env import _get_vectorization_mode

MUJOCO = "Hopper-v5"
CLASSIC = "CartPole-v1"


@pytest.mark.parametrize("n_cpu,num_envs,expected", [
    (4, 16, "sync"),    # the defect: 16 workers on 4 cores
    (4, 8, "sync"),     # still oversubscribed
    (8, 8, "async"),    # fits exactly
    (64, 16, "async"),  # a big box is what async was written for
])
def test_async_only_when_the_workers_fit_the_cores(n_cpu, num_envs, expected):
    with mock.patch("slm_lab.env.os.cpu_count", return_value=n_cpu):
        assert _get_vectorization_mode(MUJOCO, num_envs) == expected


def test_the_shipped_mujoco_spec_on_this_class_of_box_is_sync():
    """The exact configuration that died in swap. Pinned, so it cannot regress silently."""
    with mock.patch("slm_lab.env.os.cpu_count", return_value=4):
        assert _get_vectorization_mode("Hopper-v5", 16) == "sync"


def test_small_and_simple_envs_are_unaffected():
    """The pre-existing rules still decide first — this change only narrows `async`."""
    with mock.patch("slm_lab.env.os.cpu_count", return_value=64):
        assert _get_vectorization_mode(CLASSIC, 16) == "sync"   # classic_control
        assert _get_vectorization_mode(MUJOCO, 4) == "sync"     # num_envs < 8


def test_the_override_wins_so_benchmarking_can_force_either_mode():
    """A guard with no way to measure the thing it guards against cannot be checked."""
    with mock.patch.dict("os.environ", {"VECTORIZATION_MODE": "async"}):
        with mock.patch("slm_lab.env.os.cpu_count", return_value=4):
            assert _get_vectorization_mode(MUJOCO, 16) == "async"
