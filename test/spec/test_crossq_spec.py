"""Tests for CrossQ spec loading and validation."""

import pytest

try:
    import torcharc

    HAS_TORCHARC = True
except ImportError:
    HAS_TORCHARC = False


@pytest.mark.skipif(not HAS_TORCHARC, reason="torcharc not installed")
class TestCrossQSpec:
    def test_spec_loads(self):
        from slm_lab.spec import spec_util

        spec = spec_util.get(
            "benchmark_arc/crossq/crossq_mujoco_arc.yaml", "crossq_halfcheetah_arc"
        )
        assert spec is not None
        assert spec["name"] == "crossq_halfcheetah_arc"

    def test_algorithm_name(self):
        from slm_lab.spec import spec_util

        spec = spec_util.get(
            "benchmark_arc/crossq/crossq_mujoco_arc.yaml", "crossq_halfcheetah_arc"
        )
        assert spec["agent"]["algorithm"]["name"] == "CrossQ"

    def test_training_iter_utd1(self):
        from slm_lab.spec import spec_util

        spec = spec_util.get(
            "benchmark_arc/crossq/crossq_mujoco_arc.yaml", "crossq_halfcheetah_arc"
        )
        assert spec["agent"]["algorithm"]["training_iter"] == 1

    def test_critic_net_type(self):
        from slm_lab.spec import spec_util

        spec = spec_util.get(
            "benchmark_arc/crossq/crossq_mujoco_arc.yaml", "crossq_halfcheetah_arc"
        )
        assert spec["agent"]["critic_net"]["type"] == "TorchArcNet"

    def test_spec_check_passes(self):
        from slm_lab.spec import spec_util

        spec = spec_util.get(
            "benchmark_arc/crossq/crossq_mujoco_arc.yaml", "crossq_halfcheetah_arc"
        )
        assert spec_util.check(spec)
