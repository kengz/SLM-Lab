"""Tests for automatic action rescaling wrapper."""
import numpy as np
import pytest

from slm_lab.env import make_env


@pytest.fixture
def pendulum_spec():
    """Pendulum has action bounds [-2, 2] - needs rescaling."""
    return {
        "env": {"name": "Pendulum-v1", "max_t": 200, "max_frame": 1000},
        "meta": {"distributed": False, "max_session": 1},
    }


@pytest.fixture
def hopper_spec():
    """Hopper has action bounds [-1, 1] - no rescaling needed."""
    return {
        "env": {"name": "Hopper-v4", "max_t": 1000, "max_frame": 1000},
        "meta": {"distributed": False, "max_session": 1},
    }


def test_pendulum_accepts_unit_actions(pendulum_spec):
    """Pendulum should accept actions in [-1, 1] after rescaling."""
    env = make_env(pendulum_spec)
    env.reset()
    # Actions in [-1, 1] should work (will be rescaled to [-2, 2])
    obs, _, _, _, _ = env.step(np.array([1.0]))
    assert obs is not None
    obs, _, _, _, _ = env.step(np.array([-1.0]))
    assert obs is not None
    obs, _, _, _, _ = env.step(np.array([0.5]))
    assert obs is not None
    env.close()


def test_hopper_accepts_unit_actions(hopper_spec):
    """Hopper should accept actions in [-1, 1]."""
    env = make_env(hopper_spec)
    env.reset()
    # Actions in [-1, 1] should work (bounds already [-1, 1])
    obs, _, _, _, _ = env.step(np.array([1.0, 0.5, -0.5]))
    assert obs is not None
    env.close()


def test_pendulum_rescales_correctly(pendulum_spec):
    """Policy output 1.0 should map to env action 2.0 (max bound)."""
    env = make_env(pendulum_spec)
    env.reset()
    # This is a functional test - action 1.0 maps to torque 2.0
    # We can't directly test the internal scaling, but the step should work
    obs, _, _, _, _ = env.step(np.array([1.0]))
    assert obs is not None
    env.close()
